//! IR-level optimization passes.
//!
//! - `DcePass`      — Dead Code Elimination: backward BFS from side-effecting
//!                    instructions; removes pure instructions with no live uses.
//! - `CsePass`      — Common Subexpression Elimination: per-block deduplication
//!                    of pure instructions with identical operation + operands.
//! - `OpExpandPass` — Op Expansion: replaces abstract `call @ReLU(x)` etc. with
//!                    concrete `tensorop.unary.relu(x)` instructions.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::PassError;
use crate::ir::block::IrBlock;
use crate::ir::function::IrFunction;
use crate::ir::instr::{BinOp, IrInstr, TensorOp};
use crate::ir::module::IrModule;
use crate::ir::value::ValueId;
use crate::pass::Pass;

// ===========================================================================
// DcePass
// ===========================================================================

/// Dead Code Elimination.
///
/// Removes pure instructions (BinOp, Const*, TensorOp, Load) whose result is
/// never used. Side-effecting instructions (Store, Call, terminators) are
/// always kept.
pub struct DcePass;

impl Pass for DcePass {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in &mut module.functions {
            dce_function(func);
        }
        Ok(())
    }
}

fn is_side_effecting(instr: &IrInstr) -> bool {
    matches!(
        instr,
        IrInstr::Store { .. }
            | IrInstr::Br { .. }
            | IrInstr::CondBr { .. }
            | IrInstr::Return { .. }
            | IrInstr::Call { .. }
            | IrInstr::SwitchVariant { .. }
            | IrInstr::Print { .. }
            | IrInstr::Panic { .. }
            | IrInstr::ArrayStore { .. }
            | IrInstr::ChanSend { .. }
            | IrInstr::Spawn { .. }
            | IrInstr::ParFor { .. }
            | IrInstr::AtomicStore { .. }
            | IrInstr::AtomicAdd { .. }
            | IrInstr::MutexUnlock { .. }
            | IrInstr::Barrier
            | IrInstr::ReadLine { .. }
            | IrInstr::ReadI64 { .. }
            | IrInstr::ReadF64 { .. }
            | IrInstr::ListPush { .. }
            | IrInstr::ListSet { .. }
            | IrInstr::ListPop { .. }
            | IrInstr::ListSort { .. }
            | IrInstr::MapSet { .. }
            | IrInstr::MapRemove { .. }
            | IrInstr::CallClosure { .. }
            | IrInstr::FileWriteAll { .. }
            | IrInstr::ProcessExit { .. }
            | IrInstr::CallExtern { .. }
            | IrInstr::Retain { .. }
            | IrInstr::Release { .. }
            | IrInstr::TcpConnect { .. }
            | IrInstr::TcpListen { .. }
            | IrInstr::TcpAccept { .. }
            | IrInstr::TcpRead { .. }
            | IrInstr::TcpWrite { .. }
            | IrInstr::TcpClose { .. }
    )
}

fn dce_function(func: &mut IrFunction) {
    // Build result → operands map for backward reachability.
    let mut result_ops: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for block in &func.blocks {
        for instr in &block.instrs {
            if let Some(r) = instr.result() {
                result_ops.insert(r, instr.operands());
            }
        }
    }

    // Seed the live set with operands of all side-effecting instructions.
    let mut live: HashSet<ValueId> = HashSet::new();
    let mut queue: VecDeque<ValueId> = VecDeque::new();
    for block in &func.blocks {
        for instr in &block.instrs {
            if is_side_effecting(instr) {
                for op in instr.operands() {
                    if live.insert(op) {
                        queue.push_back(op);
                    }
                }
            }
        }
    }

    // BFS: if a value is live, the values it depends on are also live.
    while let Some(vid) = queue.pop_front() {
        if let Some(ops) = result_ops.get(&vid) {
            for &op in ops {
                if live.insert(op) {
                    queue.push_back(op);
                }
            }
        }
    }

    // Remove dead pure instructions.
    for block in &mut func.blocks {
        block.instrs.retain(|instr| {
            is_side_effecting(instr) || instr.result().map_or(true, |r| live.contains(&r))
        });
    }
}

// ===========================================================================
// CsePass
// ===========================================================================

/// Common Subexpression Elimination.
///
/// Per-block CSE: if two instructions produce the same value (same operation
/// on the same operands), the second is eliminated and its result replaced
/// with the first's result throughout subsequent instructions.
///
/// Commutative BinOps (Add, Mul, CmpEq) have their operand IDs sorted so that
/// `a + b` and `b + a` are treated as identical.
pub struct CsePass;

impl Pass for CsePass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in &mut module.functions {
            let mut replacements: HashMap<ValueId, ValueId> = HashMap::new();
            for block in &mut func.blocks {
                // `known` is scoped per-block: cross-block CSE would require
                // dominance analysis; without it, a value defined in one branch
                // (e.g. else2) could incorrectly substitute uses in another
                // block (e.g. merge) that isn't dominated by that branch.
                let mut known: HashMap<CseKey, ValueId> = HashMap::new();
                cse_block(block, &mut known, &mut replacements);
            }
            // Remove stale type/def entries for eliminated values.
            for (old, _) in &replacements {
                func.value_types.remove(old);
                func.value_defs.remove(old);
            }
        }
        Ok(())
    }
}

/// A hashable key that uniquely identifies a pure instruction's computation.
#[derive(Hash, Eq, PartialEq)]
enum CseKey {
    BinOp {
        op: String,
        a: u32,
        b: u32,
    },
    UnaryOp {
        op: String,
        operand: u32,
    },
    ConstFloat {
        bits: u64,
        ty: String,
    },
    ConstInt {
        value: i64,
        ty: String,
    },
    ConstBool {
        value: bool,
    },
    TensorOp {
        op: String,
        inputs: Vec<u32>,
        ty: String,
    },
}

fn is_commutative(op: BinOp) -> bool {
    matches!(op, BinOp::Add | BinOp::Mul | BinOp::CmpEq | BinOp::CmpNe)
}

fn cse_key(instr: &IrInstr) -> Option<CseKey> {
    match instr {
        IrInstr::UnaryOp { op, operand, .. } => Some(CseKey::UnaryOp {
            op: format!("{}", op),
            operand: operand.0,
        }),
        IrInstr::BinOp { op, lhs, rhs, .. } => {
            let (mut a, mut b) = (lhs.0, rhs.0);
            if is_commutative(*op) && a > b {
                std::mem::swap(&mut a, &mut b);
            }
            Some(CseKey::BinOp {
                op: format!("{}", op),
                a,
                b,
            })
        }
        IrInstr::ConstFloat { value, ty, .. } => Some(CseKey::ConstFloat {
            bits: value.to_bits(),
            ty: ty.to_string(),
        }),
        IrInstr::ConstInt { value, ty, .. } => Some(CseKey::ConstInt {
            value: *value,
            ty: ty.to_string(),
        }),
        IrInstr::ConstBool { value, .. } => Some(CseKey::ConstBool { value: *value }),
        IrInstr::TensorOp {
            op,
            inputs,
            result_ty,
            ..
        } => {
            let op_str = match op {
                TensorOp::Einsum { notation } => format!("einsum:{}", notation),
                TensorOp::Unary { op } => format!("unary:{}", op),
                TensorOp::Reshape => "reshape".to_owned(),
                TensorOp::Transpose { axes } => format!("transpose:{:?}", axes),
                TensorOp::Reduce { op, axes, keepdims } => {
                    format!("reduce:{}:{:?}:{}", op, axes, keepdims)
                }
            };
            Some(CseKey::TensorOp {
                op: op_str,
                inputs: inputs.iter().map(|v| v.0).collect(),
                ty: result_ty.to_string(),
            })
        }
        // Side-effecting or complex — not CSE-able.
        _ => None,
    }
}

pub(crate) fn apply_replacements(instr: &mut IrInstr, reps: &HashMap<ValueId, ValueId>) {
    let replace = |v: &mut ValueId| {
        if let Some(&r) = reps.get(v) {
            *v = r;
        }
    };
    match instr {
        IrInstr::BinOp { lhs, rhs, .. } => {
            replace(lhs);
            replace(rhs);
        }
        IrInstr::UnaryOp { operand, .. } => {
            replace(operand);
        }
        IrInstr::TensorOp { inputs, .. } => {
            for v in inputs {
                replace(v);
            }
        }
        IrInstr::Load {
            tensor, indices, ..
        } => {
            replace(tensor);
            for v in indices {
                replace(v);
            }
        }
        IrInstr::Store {
            tensor,
            indices,
            value,
        } => {
            replace(tensor);
            replace(value);
            for v in indices {
                replace(v);
            }
        }
        IrInstr::Br { args, .. } => {
            for v in args {
                replace(v);
            }
        }
        IrInstr::CondBr {
            cond,
            then_args,
            else_args,
            ..
        } => {
            replace(cond);
            for v in then_args {
                replace(v);
            }
            for v in else_args {
                replace(v);
            }
        }
        IrInstr::Return { values } => {
            for v in values {
                replace(v);
            }
        }
        IrInstr::Call { args, .. } => {
            for v in args {
                replace(v);
            }
        }
        IrInstr::Cast { operand, .. } => {
            replace(operand);
        }
        // Constants have no operands.
        IrInstr::ConstFloat { .. } | IrInstr::ConstInt { .. } | IrInstr::ConstBool { .. } => {}
        IrInstr::MakeStruct { fields, .. } => {
            for v in fields {
                replace(v);
            }
        }
        IrInstr::GetField { base, .. } => {
            replace(base);
        }
        IrInstr::MakeVariant { fields, .. } => {
            for v in fields { replace(v); }
        }
        IrInstr::SwitchVariant { scrutinee, .. } => {
            replace(scrutinee);
        }
        IrInstr::ExtractVariantField { operand, .. } => {
            replace(operand);
        }
        IrInstr::MakeTuple { elements, .. } => {
            for v in elements {
                replace(v);
            }
        }
        IrInstr::GetElement { base, .. } => {
            replace(base);
        }
        IrInstr::AllocArray { init, .. } => {
            for v in init { replace(v); }
        }
        IrInstr::ArrayLoad { array, index, .. } => {
            replace(array);
            replace(index);
        }
        IrInstr::ArrayStore { array, index, value } => {
            replace(array);
            replace(index);
            replace(value);
        }
        IrInstr::ConstStr { .. } => {}
        IrInstr::StrLen { operand, .. } => {
            replace(operand);
        }
        IrInstr::StrConcat { lhs, rhs, .. } => {
            replace(lhs);
            replace(rhs);
        }
        IrInstr::Print { operand } => {
            replace(operand);
        }
        IrInstr::MakeClosure { captures, .. } => {
            for v in captures {
                replace(v);
            }
        }
        IrInstr::CallClosure { closure, args, .. } => {
            replace(closure);
            for v in args {
                replace(v);
            }
        }
        IrInstr::ParFor { start, end, args, .. } => { replace(start); replace(end); for v in args { replace(v); } }
        IrInstr::ChanNew { .. } => {}
        IrInstr::ChanSend { chan, value } => { replace(chan); replace(value); }
        IrInstr::ChanRecv { chan, .. } => { replace(chan); }
        IrInstr::Spawn { args, .. } => { for v in args { replace(v); } }
        IrInstr::AtomicNew { value, .. } => { replace(value); }
        IrInstr::AtomicLoad { atomic, .. } => { replace(atomic); }
        IrInstr::AtomicStore { atomic, value } => { replace(atomic); replace(value); }
        IrInstr::AtomicAdd { atomic, value, .. } => { replace(atomic); replace(value); }
        IrInstr::MutexNew { value, .. } => { replace(value); }
        IrInstr::MutexLock { mutex, .. } => { replace(mutex); }
        IrInstr::MutexUnlock { mutex } => { replace(mutex); }
        IrInstr::MakeSome { value, .. } => { replace(value); }
        IrInstr::MakeNone { .. } => {}
        IrInstr::IsSome { operand, .. } => { replace(operand); }
        IrInstr::OptionUnwrap { operand, .. } => { replace(operand); }
        IrInstr::MakeOk { value, .. } => { replace(value); }
        IrInstr::MakeErr { value, .. } => { replace(value); }
        IrInstr::IsOk { operand, .. } => { replace(operand); }
        IrInstr::ResultUnwrap { operand, .. } => { replace(operand); }
        IrInstr::ResultUnwrapErr { operand, .. } => { replace(operand); }
        IrInstr::Barrier => {}
        IrInstr::Sparsify { operand, .. } => { replace(operand); }
        IrInstr::Densify { operand, .. } => { replace(operand); }
        IrInstr::MakeGrad { value, tangent, .. } => { replace(value); replace(tangent); }
        IrInstr::GradValue { operand, .. } => { replace(operand); }
        IrInstr::GradTangent { operand, .. } => { replace(operand); }
        IrInstr::StrContains { haystack, needle, .. } => { replace(haystack); replace(needle); }
        IrInstr::StrStartsWith { haystack, prefix, .. } => { replace(haystack); replace(prefix); }
        IrInstr::StrEndsWith { haystack, suffix, .. } => { replace(haystack); replace(suffix); }
        IrInstr::StrToUpper { operand, .. } => { replace(operand); }
        IrInstr::StrToLower { operand, .. } => { replace(operand); }
        IrInstr::StrTrim { operand, .. } => { replace(operand); }
        IrInstr::StrRepeat { operand, count, .. } => { replace(operand); replace(count); }
        IrInstr::Panic { msg } => { replace(msg); }
        IrInstr::ValueToStr { operand, .. } => { replace(operand); }
        IrInstr::ReadLine { .. } => {}
        IrInstr::ReadI64 { .. } => {}
        IrInstr::ReadF64 { .. } => {}
        IrInstr::ParseI64 { operand, .. } => { replace(operand); }
        IrInstr::ParseF64 { operand, .. } => { replace(operand); }
        IrInstr::StrIndex { string, index, .. } => { replace(string); replace(index); }
        IrInstr::StrSlice { string, start, end, .. } => { replace(string); replace(start); replace(end); }
        IrInstr::StrFind { haystack, needle, .. } => { replace(haystack); replace(needle); }
        IrInstr::StrReplace { string, from, to, .. } => { replace(string); replace(from); replace(to); }
        IrInstr::ListNew { .. } => {}
        IrInstr::ListPush { list, value } => { replace(list); replace(value); }
        IrInstr::ListLen { list, .. } => { replace(list); }
        IrInstr::ListGet { list, index, .. } => { replace(list); replace(index); }
        IrInstr::ListSet { list, index, value } => { replace(list); replace(index); replace(value); }
        IrInstr::ListPop { list, .. } => { replace(list); }
        IrInstr::MapNew { .. } => {}
        IrInstr::MapSet { map, key, value } => { replace(map); replace(key); replace(value); }
        IrInstr::MapGet { map, key, .. } => { replace(map); replace(key); }
        IrInstr::MapContains { map, key, .. } => { replace(map); replace(key); }
        IrInstr::MapRemove { map, key } => { replace(map); replace(key); }
        IrInstr::MapLen { map, .. } => { replace(map); }
        // Phase 56: File I/O
        IrInstr::FileReadAll { path, .. } => { replace(path); }
        IrInstr::FileWriteAll { path, content, .. } => { replace(path); replace(content); }
        IrInstr::FileExists { path, .. } => { replace(path); }
        IrInstr::FileLines { path, .. } => { replace(path); }
        // Phase 58: Extended collections
        IrInstr::ListContains { list, value, .. } => { replace(list); replace(value); }
        IrInstr::ListSort { list } => { replace(list); }
        IrInstr::MapKeys { map, .. } => { replace(map); }
        IrInstr::MapValues { map, .. } => { replace(map); }
        IrInstr::ListConcat { lhs, rhs, .. } => { replace(lhs); replace(rhs); }
        IrInstr::ListSlice { list, start, end, .. } => { replace(list); replace(start); replace(end); }
        // Phase 59: Process / environment
        IrInstr::ProcessExit { code } => { replace(code); }
        IrInstr::ProcessArgs { .. } => {}
        IrInstr::EnvVar { name, .. } => { replace(name); }
        // Phase 61: Pattern matching helpers
        IrInstr::GetVariantTag { operand, .. } => { replace(operand); }
        IrInstr::StrEq { lhs, rhs, .. } => { replace(lhs); replace(rhs); }
        // Phase 81: FFI
        IrInstr::CallExtern { args, .. } => { for a in args { replace(a); } }
        // Phase 83: GC
        IrInstr::Retain { ptr } => { replace(ptr); }
        IrInstr::Release { ptr, .. } => { replace(ptr); }
        IrInstr::TcpConnect { host, port, .. } => { replace(host); replace(port); }
        IrInstr::TcpListen { port, .. } => { replace(port); }
        IrInstr::TcpAccept { listener, .. } => { replace(listener); }
        IrInstr::TcpRead { conn, .. } => { replace(conn); }
        IrInstr::TcpWrite { conn, data } => { replace(conn); replace(data); }
        IrInstr::TcpClose { conn } => { replace(conn); }
    }
}

fn cse_block(
    block: &mut IrBlock,
    known: &mut HashMap<CseKey, ValueId>,
    replacements: &mut HashMap<ValueId, ValueId>,
) {
    let mut new_instrs = Vec::new();
    for mut instr in block.instrs.drain(..) {
        apply_replacements(&mut instr, replacements);
        if let Some(key) = cse_key(&instr) {
            if let Some(&prev) = known.get(&key) {
                // Duplicate found: record replacement, drop the instruction.
                if let Some(result) = instr.result() {
                    replacements.insert(result, prev);
                }
                continue;
            } else if let Some(result) = instr.result() {
                known.insert(key, result);
            }
        }
        new_instrs.push(instr);
    }
    block.instrs = new_instrs;
}

// ===========================================================================
// OpExpandPass
// ===========================================================================

/// Op Expansion.
///
/// Replaces `call @ReLU(x)`, `call @Sigmoid(x)`, `call @Tanh(x)`, and
/// `call @GELU(x)` with the equivalent `TensorOp::Unary` instruction, so
/// that downstream passes (DCE, CSE, ShapeCheck) operate on concrete ops.
///
/// Ops that require weights or reductions (Dense, Linear, Softmax, Add, Concat)
/// are left as `Call` instructions.
pub struct OpExpandPass;

impl Pass for OpExpandPass {
    fn name(&self) -> &'static str {
        "op-expand"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in &mut module.functions {
            for block in &mut func.blocks {
                for instr in &mut block.instrs {
                    if let IrInstr::Call {
                        result: Some(result),
                        callee,
                        args,
                        result_ty: Some(ty),
                    } = instr
                    {
                        if let Some(unary_name) = elementwise_unary(callee) {
                            *instr = IrInstr::TensorOp {
                                result: *result,
                                op: TensorOp::Unary {
                                    op: unary_name.to_owned(),
                                },
                                inputs: args.clone(),
                                result_ty: ty.clone(),
                            };
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Maps known elementwise activation op names to their `TensorOp::Unary` op strings.
/// Returns `None` for ops that cannot be expanded (require weights, reductions, etc.).
fn elementwise_unary(callee: &str) -> Option<&'static str> {
    match callee {
        "ReLU" => Some("relu"),
        "Sigmoid" => Some("sigmoid"),
        "Tanh" => Some("tanh"),
        "GELU" => Some("gelu"),
        _ => None,
    }
}
