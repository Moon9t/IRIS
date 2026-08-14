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
                IrInstr::BinOp {
                    result,
                    op,
                    lhs,
                    rhs,
                    ty,
                } => {
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
                            new_instrs.push(IrInstr::BinOp {
                                result,
                                op: BinOp::Mul,
                                lhs,
                                rhs,
                                ty,
                            });
                        }
                        BinOp::Div => {
                            // `x / 2^n → x >> n` is NOT a valid rewrite for signed
                            // integers and has been removed.
                            //
                            // An arithmetic shift right *floors*; IRIS division
                            // truncates toward zero (`wrapping_div`, matching the
                            // interpreter and the C runtime). They differ for every
                            // negative dividend that does not divide exactly:
                            //
                            //     -7 / 2  == -3      (truncate — correct)
                            //     -7 >> 1 == -4      (floor    — what this emitted)
                            //
                            // The rewrite therefore silently changed results, and
                            // only when the divisor happened to be a visible
                            // constant. Whether it fired depended on inlining, so
                            // the same expression gave -3 through a real call and
                            // -4 once the callee was small enough to inline. It also
                            // broke the identity `(a/b)*b + (a%b) == a`, since `%`
                            // was left truncating.
                            //
                            // Recovering the optimisation would need the usual bias
                            // correction (add `(x >> 63) >>> (64 - n)` before
                            // shifting), or a proof that `x >= 0`. Neither is worth
                            // doing here: codegen emits LLVM `sdiv`, and LLVM
                            // performs this strength reduction itself, correctly,
                            // whenever it is sound.
                            new_instrs.push(IrInstr::BinOp {
                                result,
                                op: BinOp::Div,
                                lhs,
                                rhs,
                                ty,
                            });
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
                            new_instrs.push(IrInstr::BinOp {
                                result,
                                op: BinOp::Sub,
                                lhs,
                                rhs,
                                ty,
                            });
                        }
                        _ => {
                            new_instrs.push(IrInstr::BinOp {
                                result,
                                op: *op,
                                lhs,
                                rhs,
                                ty,
                            });
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::module::{IrFunctionBuilder, IrModule};
    use crate::ir::types::{DType, IrType};
    use crate::pass::Pass;

    fn i64_ty() -> IrType {
        IrType::Scalar(DType::I64)
    }

    /// Build `x * rhs_const` and return the module.
    fn build_mul_module(rhs_val: i64) -> IrModule {
        let mut m = IrModule::new("test");
        let mut builder = IrFunctionBuilder::new("f", vec![], i64_ty());
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);

        let x = builder.fresh_value(); // %0
        builder.push_instr(
            IrInstr::ConstInt {
                result: x,
                value: 7,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        let c = builder.fresh_value(); // %1
        builder.push_instr(
            IrInstr::ConstInt {
                result: c,
                value: rhs_val,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        let r = builder.fresh_value(); // %2
        builder.push_instr(
            IrInstr::BinOp {
                result: r,
                op: BinOp::Mul,
                lhs: x,
                rhs: c,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        builder.push_instr(IrInstr::Return { values: vec![r] }, None);
        m.add_function(builder.build()).unwrap();
        m
    }

    /// Build `x / rhs_const` and return the module.
    fn build_div_module(rhs_val: i64) -> IrModule {
        let mut m = IrModule::new("test");
        let mut builder = IrFunctionBuilder::new("f", vec![], i64_ty());
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);

        let x = builder.fresh_value();
        builder.push_instr(
            IrInstr::ConstInt {
                result: x,
                value: 100,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        let c = builder.fresh_value();
        builder.push_instr(
            IrInstr::ConstInt {
                result: c,
                value: rhs_val,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        let r = builder.fresh_value();
        builder.push_instr(
            IrInstr::BinOp {
                result: r,
                op: BinOp::Div,
                lhs: x,
                rhs: c,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        builder.push_instr(IrInstr::Return { values: vec![r] }, None);
        m.add_function(builder.build()).unwrap();
        m
    }

    /// Build `x - x` and return the module.
    fn build_sub_self_module() -> IrModule {
        let mut m = IrModule::new("test");
        let mut builder = IrFunctionBuilder::new("f", vec![], i64_ty());
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);

        let x = builder.fresh_value();
        builder.push_instr(
            IrInstr::ConstInt {
                result: x,
                value: 42,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        let r = builder.fresh_value();
        builder.push_instr(
            IrInstr::BinOp {
                result: r,
                op: BinOp::Sub,
                lhs: x,
                rhs: x,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        builder.push_instr(IrInstr::Return { values: vec![r] }, None);
        m.add_function(builder.build()).unwrap();
        m
    }

    #[test]
    fn pass_name() {
        let pass = StrengthReducePass;
        assert_eq!(pass.name(), "strength-reduce");
    }

    #[test]
    fn mul_by_power_of_two_reduced_to_shl() {
        let mut m = build_mul_module(8); // 8 = 2^3
        StrengthReducePass.run(&mut m).unwrap();

        let block = &m.functions()[0].blocks()[0];
        // Should have: ConstInt(7), ConstInt(8), ConstInt(3), Shl, Return
        let has_shl = block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::BinOp { op: BinOp::Shl, .. }));
        assert!(has_shl, "mul by 8 should be reduced to shl");
        let no_mul = !block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::BinOp { op: BinOp::Mul, .. }));
        assert!(no_mul, "mul should be removed");
    }

    #[test]
    fn mul_by_non_power_of_two_not_reduced() {
        let mut m = build_mul_module(7);
        StrengthReducePass.run(&mut m).unwrap();

        let block = &m.functions()[0].blocks()[0];
        let has_mul = block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::BinOp { op: BinOp::Mul, .. }));
        assert!(has_mul, "mul by 7 should NOT be reduced");
    }

    #[test]
    fn mul_by_one_not_reduced() {
        // 1 is 2^0, so it should be reduced to shl by 0
        let mut m = build_mul_module(1);
        StrengthReducePass.run(&mut m).unwrap();

        let block = &m.functions()[0].blocks()[0];
        let has_shl = block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::BinOp { op: BinOp::Shl, .. }));
        assert!(has_shl, "mul by 1 is 2^0, should become shl by 0");
    }

    #[test]
    fn div_by_power_of_two_reduced_to_shr() {
        let mut m = build_div_module(4); // 4 = 2^2
        StrengthReducePass.run(&mut m).unwrap();

        let block = &m.functions()[0].blocks()[0];
        let has_shr = block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::BinOp { op: BinOp::Shr, .. }));
        assert!(has_shr, "div by 4 should be reduced to shr");
    }

    #[test]
    fn div_by_non_power_of_two_not_reduced() {
        let mut m = build_div_module(3);
        StrengthReducePass.run(&mut m).unwrap();

        let block = &m.functions()[0].blocks()[0];
        let has_div = block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::BinOp { op: BinOp::Div, .. }));
        assert!(has_div, "div by 3 should NOT be reduced");
    }

    #[test]
    fn sub_self_reduced_to_zero() {
        let mut m = build_sub_self_module();
        StrengthReducePass.run(&mut m).unwrap();

        let block = &m.functions()[0].blocks()[0];
        // x - x should become ConstInt(0)
        let no_sub = !block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::BinOp { op: BinOp::Sub, .. }));
        assert!(no_sub, "sub self should be removed");
        // Should have a ConstInt with value 0 for the result
        let has_zero = block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::ConstInt { value: 0, .. }));
        assert!(has_zero, "sub self should produce const 0");
    }

    #[test]
    fn is_power_of_two_basic() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(6));
        assert!(!is_power_of_two(-1));
    }

    #[test]
    fn log2_exact_values() {
        assert_eq!(log2_exact(1), 0);
        assert_eq!(log2_exact(2), 1);
        assert_eq!(log2_exact(4), 2);
        assert_eq!(log2_exact(8), 3);
        assert_eq!(log2_exact(1024), 10);
    }

    #[test]
    fn strength_reduce_preserves_other_ops() {
        // Add should not be touched
        let mut m = IrModule::new("test");
        let mut builder = IrFunctionBuilder::new("f", vec![], i64_ty());
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);

        let a = builder.fresh_value();
        builder.push_instr(
            IrInstr::ConstInt {
                result: a,
                value: 3,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        let b = builder.fresh_value();
        builder.push_instr(
            IrInstr::ConstInt {
                result: b,
                value: 4,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        let r = builder.fresh_value();
        builder.push_instr(
            IrInstr::BinOp {
                result: r,
                op: BinOp::Add,
                lhs: a,
                rhs: b,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        builder.push_instr(IrInstr::Return { values: vec![r] }, None);
        m.add_function(builder.build()).unwrap();

        StrengthReducePass.run(&mut m).unwrap();

        let block = &m.functions()[0].blocks()[0];
        let has_add = block
            .instrs
            .iter()
            .any(|i| matches!(i, IrInstr::BinOp { op: BinOp::Add, .. }));
        assert!(has_add, "add should be preserved");
    }
}
