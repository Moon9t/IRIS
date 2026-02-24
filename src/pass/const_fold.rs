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
                        match (op, kv) {
                            (ScalarUnaryOp::Neg, KnownVal::Float(f)) => {
                                let v = -f;
                                known.insert(*result, KnownVal::Float(v));
                                new_instrs.push(IrInstr::ConstFloat {
                                    result: *result,
                                    value: v,
                                    ty: ty.clone(),
                                });
                                continue;
                            }
                            (ScalarUnaryOp::Neg, KnownVal::Int(i)) => {
                                let v = i.wrapping_neg();
                                known.insert(*result, KnownVal::Int(v));
                                new_instrs.push(IrInstr::ConstInt {
                                    result: *result,
                                    value: v,
                                    ty: ty.clone(),
                                });
                                continue;
                            }
                            _ => {}
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
                IrInstr::ConstFloat {
                    result,
                    value: v,
                    ty: ty.clone(),
                },
                KnownVal::Float(v),
            ))
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
        IrInstr::MakeVariant { .. } => {}
        IrInstr::SwitchVariant { scrutinee, .. } => {
            replace(scrutinee);
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
    }
}
