//! Constant folding and identity simplification pass for `IrModule`.
//!
//! `ConstFoldPass` performs a forward, single-pass walk of each function's
//! instructions and applies two categories of reduction:
//!
//! **A. Constant arithmetic** — when both operands of a `BinOp` are known
//! compile-time constants the instruction is replaced with a single `Const`:
//! - `ConstFloat op ConstFloat` → folded `ConstFloat`  (Add, Sub, Mul, Div)
//! - `ConstInt   op ConstInt`   → folded `ConstInt`    (Add, Sub, Mul)
//!
//! **B. Identity simplification** — when one operand is a neutral element the
//! result is replaced with the other operand and the instruction is dropped:
//! - `x + 0 → x`  |  `0 + x → x`
//! - `x * 1 → x`  |  `1 * x → x`
//!
//! Passes `DCE` and `CSE` run afterward and remove any constants whose results
//! become unused after folding.

use std::collections::HashMap;

use crate::error::PassError;
use crate::ir::function::IrFunction;
use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;
use crate::ir::value::ValueId;
use crate::pass::Pass;

pub struct ConstFoldPass;

impl Pass for ConstFoldPass {
    fn name(&self) -> &'static str {
        "const-fold"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in &mut module.functions {
            const_fold_func(func);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal representation of a known constant value
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum KnownVal {
    Int(i64),
    Float(f64),
}

// ---------------------------------------------------------------------------
// Per-function folding pass
// ---------------------------------------------------------------------------

fn const_fold_func(func: &mut IrFunction) {
    let mut known: HashMap<ValueId, KnownVal> = HashMap::new();
    let mut reps: HashMap<ValueId, ValueId> = HashMap::new();

    for block in &mut func.blocks {
        let mut new_instrs = Vec::new();
        for mut instr in block.instrs.drain(..) {
            // Apply pending value replacements to this instruction's operands.
            apply_reps(&mut instr, &reps);

            match &instr {
                IrInstr::ConstInt { result, value, .. } => {
                    known.insert(*result, KnownVal::Int(*value));
                    new_instrs.push(instr);
                }
                IrInstr::ConstFloat { result, value, .. } => {
                    known.insert(*result, KnownVal::Float(*value));
                    new_instrs.push(instr);
                }
                IrInstr::UnaryOp {
                    result,
                    op,
                    operand,
                    ty,
                } => {
                    if let Some(kv) = known.get(operand).cloned() {
                        let folded: Option<(IrInstr, KnownVal)> = match (op, &kv) {
                            (ScalarUnaryOp::Neg, KnownVal::Float(f)) => {
                                let v = -f;
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Neg, KnownVal::Int(i)) => {
                                let v = i.wrapping_neg();
                                Some((IrInstr::ConstInt { result: *result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
                            }
                            (ScalarUnaryOp::Sqrt, KnownVal::Float(f)) => {
                                let v = f.sqrt();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Abs, KnownVal::Float(f)) => {
                                let v = f.abs();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Abs, KnownVal::Int(i)) => {
                                let v = i.wrapping_abs();
                                Some((IrInstr::ConstInt { result: *result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
                            }
                            (ScalarUnaryOp::Floor, KnownVal::Float(f)) => {
                                let v = f.floor();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Ceil, KnownVal::Float(f)) => {
                                let v = f.ceil();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::BitNot, KnownVal::Int(i)) => {
                                let v = !i;
                                Some((IrInstr::ConstInt { result: *result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
                            }
                            // Phase 36 trig/transcendental folding
                            (ScalarUnaryOp::Sin, KnownVal::Float(f)) => {
                                let v = f.sin();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Cos, KnownVal::Float(f)) => {
                                let v = f.cos();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Tan, KnownVal::Float(f)) => {
                                let v = f.tan();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Exp, KnownVal::Float(f)) => {
                                let v = f.exp();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Log, KnownVal::Float(f)) => {
                                let v = f.ln();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Log2, KnownVal::Float(f)) => {
                                let v = f.log2();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Round, KnownVal::Float(f)) => {
                                let v = f.round();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Sign, KnownVal::Float(f)) => {
                                let v = f.signum();
                                Some((IrInstr::ConstFloat { result: *result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
                            }
                            (ScalarUnaryOp::Sign, KnownVal::Int(i)) => {
                                let v = i.signum();
                                Some((IrInstr::ConstInt { result: *result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
                            }
                            _ => None,
                        };
                        if let Some((folded_instr, folded_val)) = folded {
                            known.insert(*result, folded_val);
                            new_instrs.push(folded_instr);
                            continue;
                        }
                    }
                    new_instrs.push(instr);
                }

                IrInstr::BinOp {
                    result,
                    op,
                    lhs,
                    rhs,
                    ty,
                } => {
                    let lv = known.get(lhs).cloned();
                    let rv = known.get(rhs).cloned();

                    // Case A: both operands are known constants — fold.
                    if let (Some(lv), Some(rv)) = (lv, rv) {
                        if let Some((folded_instr, folded_val)) =
                            eval_binop(*op, *result, &lv, &rv, ty)
                        {
                            known.insert(*result, folded_val);
                            new_instrs.push(folded_instr);
                            continue;
                        }
                    }

                    // Case B: identity simplification — drop instr, record rep.
                    if let Some(rep) = identity_rep(*op, *lhs, *rhs, &known) {
                        // Chase existing replacements so the chain stays flat.
                        let canonical = *reps.get(&rep).unwrap_or(&rep);
                        reps.insert(*result, canonical);
                        continue;
                    }

                    new_instrs.push(instr);
                }

                _ => new_instrs.push(instr),
            }
        }
        block.instrs = new_instrs;
    }

    // Remove stale type/def entries for values that were replaced (like CsePass).
    for (old, _) in &reps {
        func.value_types.remove(old);
        func.value_defs.remove(old);
    }
}

// ---------------------------------------------------------------------------
// Constant arithmetic evaluation
// ---------------------------------------------------------------------------

/// Tries to evaluate `lhs op rhs` when both operands are known constants.
/// Returns the replacement instruction and the folded `KnownVal`, or `None`
/// if the operation cannot be folded (e.g. division by zero).
fn eval_binop(
    op: BinOp,
    result: ValueId,
    lv: &KnownVal,
    rv: &KnownVal,
    ty: &IrType,
) -> Option<(IrInstr, KnownVal)> {
    match (op, lv, rv) {
        // Float arithmetic
        (BinOp::Add, KnownVal::Float(a), KnownVal::Float(b)) => {
            let v = a + b;
            Some((
                IrInstr::ConstFloat {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Float(v),
            ))
        }
        (BinOp::Sub, KnownVal::Float(a), KnownVal::Float(b)) => {
            let v = a - b;
            Some((
                IrInstr::ConstFloat {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Float(v),
            ))
        }
        (BinOp::Mul, KnownVal::Float(a), KnownVal::Float(b)) => {
            let v = a * b;
            Some((
                IrInstr::ConstFloat {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Float(v),
            ))
        }
        (BinOp::Div, KnownVal::Float(a), KnownVal::Float(b)) if *b != 0.0 => {
            let v = a / b;
            Some((
                IrInstr::ConstFloat {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Float(v),
            ))
        }

        // Integer arithmetic
        (BinOp::Add, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = a.wrapping_add(*b);
            Some((
                IrInstr::ConstInt {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Int(v),
            ))
        }
        (BinOp::Sub, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = a.wrapping_sub(*b);
            Some((
                IrInstr::ConstInt {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Int(v),
            ))
        }
        (BinOp::Mul, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = a.wrapping_mul(*b);
            Some((
                IrInstr::ConstInt {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Int(v),
            ))
        }
        (BinOp::Mod, KnownVal::Int(a), KnownVal::Int(b)) if *b != 0 => {
            let v = a.wrapping_rem(*b);
            Some((
                IrInstr::ConstInt {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Int(v),
            ))
        }
        (BinOp::Mod, KnownVal::Float(a), KnownVal::Float(b)) if *b != 0.0 => {
            let v = a % b;
            Some((
                IrInstr::ConstFloat { result, value: v, ty: ty.clone() },
                KnownVal::Float(v),
            ))
        }
        // Math builtins: pow, min, max
        (BinOp::Pow, KnownVal::Float(a), KnownVal::Float(b)) => {
            let v = a.powf(*b);
            Some((IrInstr::ConstFloat { result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
        }
        (BinOp::Min, KnownVal::Float(a), KnownVal::Float(b)) => {
            let v = a.min(*b);
            Some((IrInstr::ConstFloat { result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
        }
        (BinOp::Max, KnownVal::Float(a), KnownVal::Float(b)) => {
            let v = a.max(*b);
            Some((IrInstr::ConstFloat { result, value: v, ty: ty.clone() }, KnownVal::Float(v)))
        }
        (BinOp::Pow, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = (*a as f64).powf(*b as f64) as i64;
            Some((IrInstr::ConstInt { result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
        }
        (BinOp::Min, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = *a.min(b);
            Some((IrInstr::ConstInt { result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
        }
        (BinOp::Max, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = *a.max(b);
            Some((IrInstr::ConstInt { result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
        }
        // Bitwise ops on integers
        (BinOp::BitAnd, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = a & b;
            Some((IrInstr::ConstInt { result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
        }
        (BinOp::BitOr, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = a | b;
            Some((IrInstr::ConstInt { result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
        }
        (BinOp::BitXor, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = a ^ b;
            Some((IrInstr::ConstInt { result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
        }
        (BinOp::Shl, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = a.wrapping_shl(*b as u32);
            Some((IrInstr::ConstInt { result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
        }
        (BinOp::Shr, KnownVal::Int(a), KnownVal::Int(b)) => {
            let v = a.wrapping_shr(*b as u32);
            Some((IrInstr::ConstInt { result, value: v, ty: ty.clone() }, KnownVal::Int(v)))
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Identity simplification
// ---------------------------------------------------------------------------

/// Returns the value that `result` should be replaced with if one operand is
/// a neutral element for `op`, or `None` if no simplification applies.
fn identity_rep(
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
    known: &HashMap<ValueId, KnownVal>,
) -> Option<ValueId> {
    let is_zero = |v: ValueId| {
        matches!(known.get(&v), Some(KnownVal::Float(f)) if *f == 0.0)
            || matches!(known.get(&v), Some(KnownVal::Int(i)) if *i == 0)
    };
    let is_one = |v: ValueId| {
        matches!(known.get(&v), Some(KnownVal::Float(f)) if *f == 1.0)
            || matches!(known.get(&v), Some(KnownVal::Int(i)) if *i == 1)
    };

    match op {
        BinOp::Add => {
            if is_zero(rhs) {
                return Some(lhs);
            }
            if is_zero(lhs) {
                return Some(rhs);
            }
        }
        BinOp::Mul => {
            if is_one(rhs) {
                return Some(lhs);
            }
            if is_one(lhs) {
                return Some(rhs);
            }
        }
        _ => {}
    }
    None
}

// ---------------------------------------------------------------------------
// Operand replacement (mirrors CsePass::apply_replacements)
// ---------------------------------------------------------------------------

fn apply_reps(instr: &mut IrInstr, reps: &HashMap<ValueId, ValueId>) {
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
        IrInstr::Cast { operand, .. } => {
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
        IrInstr::StrSplit { str_val, delim, .. } => { replace(str_val); replace(delim); }
        IrInstr::StrJoin { list_val, delim, .. } => { replace(list_val); replace(delim); }
    }
}
