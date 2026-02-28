//! Strength reduction optimizer: replaces expensive ops with cheaper equivalents.
//!
//! Transformations:
//! - `x * 2^n` → `x << n`  (multiply by power-of-2 → shift left)
//! - `x / 2^n` → `x >> n`  (integer divide by power-of-2 → logical shift right)
//! - `x - x`   → `0`       (subtraction of identical values)

use std::collections::HashMap;

use crate::error::PassError;
use crate::ir::function::IrFunction;
use crate::ir::instr::{BinOp, IrInstr};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;
use crate::ir::value::ValueId;
use crate::pass::Pass;

pub struct StrengthReducePass;

impl Pass for StrengthReducePass {
    fn name(&self) -> &'static str {
        "strength-reduce"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in &mut module.functions {
            strength_reduce_func(func);
        }
        Ok(())
    }
}

fn is_power_of_two(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn log2_exact(n: i64) -> i64 {
    debug_assert!(is_power_of_two(n));
    (n as u64).trailing_zeros() as i64
}

fn strength_reduce_func(func: &mut IrFunction) {
    // Map from ValueId → known integer constant.
    let mut known_consts: HashMap<ValueId, i64> = HashMap::new();

    // Use a local counter to avoid borrowing func while func.blocks is borrowed.
    let mut next_id = func.next_value;
    let mut fresh = || {
        let id = ValueId(next_id);
        next_id += 1;
        id
    };

    for block in &mut func.blocks {
        let mut new_instrs = Vec::new();
        for instr in block.instrs.drain(..) {
            match &instr {
                IrInstr::ConstInt { result, value, .. } => {
                    known_consts.insert(*result, *value);
                    new_instrs.push(instr);
                }
                IrInstr::BinOp { result, op, lhs, rhs, ty } => {
                    let lhs_const = known_consts.get(lhs).copied();
                    let rhs_const = known_consts.get(rhs).copied();
                    let ty = ty.clone();
                    let result = *result;
                    let lhs = *lhs;
                    let rhs = *rhs;

                    match op {
                        BinOp::Mul => {
                            // x * 2^n → x << n
                            if let Some(c) = rhs_const {
                                if is_power_of_two(c) {
                                    let shift = log2_exact(c);
                                    let shift_val = fresh();
                                    new_instrs.push(IrInstr::ConstInt {
                                        result: shift_val,
                                        value: shift,
                                        ty: IrType::Scalar(crate::ir::types::DType::I64),
                                    });
                                    new_instrs.push(IrInstr::BinOp {
                                        result,
                                        op: BinOp::Shl,
                                        lhs,
                                        rhs: shift_val,
                                        ty,
                                    });
                                    continue;
                                }
                            }
                            if let Some(c) = lhs_const {
                                if is_power_of_two(c) {
                                    let shift = log2_exact(c);
                                    let shift_val = fresh();
                                    new_instrs.push(IrInstr::ConstInt {
                                        result: shift_val,
                                        value: shift,
                                        ty: IrType::Scalar(crate::ir::types::DType::I64),
                                    });
                                    new_instrs.push(IrInstr::BinOp {
                                        result,
                                        op: BinOp::Shl,
                                        lhs: rhs, // commutative
                                        rhs: shift_val,
                                        ty,
                                    });
                                    continue;
                                }
                            }
                            new_instrs.push(IrInstr::BinOp { result, op: BinOp::Mul, lhs, rhs, ty });
                        }
                        BinOp::Div => {
                            // x / 2^n → x >> n  (integer divide by positive power-of-2)
                            if let Some(c) = rhs_const {
                                if is_power_of_two(c) {
                                    let shift = log2_exact(c);
                                    let shift_val = fresh();
                                    new_instrs.push(IrInstr::ConstInt {
                                        result: shift_val,
                                        value: shift,
                                        ty: IrType::Scalar(crate::ir::types::DType::I64),
                                    });
                                    new_instrs.push(IrInstr::BinOp {
                                        result,
                                        op: BinOp::Shr,
                                        lhs,
                                        rhs: shift_val,
                                        ty,
                                    });
                                    continue;
                                }
                            }
                            new_instrs.push(IrInstr::BinOp { result, op: BinOp::Div, lhs, rhs, ty });
                        }
                        BinOp::Sub => {
                            // x - x → 0
                            if lhs == rhs {
                                new_instrs.push(IrInstr::ConstInt {
                                    result,
                                    value: 0,
                                    ty,
                                });
                                continue;
                            }
                            new_instrs.push(IrInstr::BinOp { result, op: BinOp::Sub, lhs, rhs, ty });
                        }
                        _ => {
                            new_instrs.push(IrInstr::BinOp { result, op: *op, lhs, rhs, ty });
                        }
                    }
                }
                _ => new_instrs.push(instr),
            }
        }
        block.instrs = new_instrs;
    }

    // Sync the counter back so any subsequent passes see unique IDs.
    func.next_value = next_id;
}
