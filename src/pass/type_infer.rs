//! Type consistency checking pass.
//!
//! For bootstrap: validates that type constraints are locally consistent
//! without doing full unification. A full unification-based inference engine
//! is the natural next step once the IR stabilizes.

use crate::error::PassError;
use crate::ir::instr::{IrInstr, ScalarUnaryOp, TensorOp};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::pass::Pass;

/// Checks that tensor operation result types are consistent with their inputs,
/// and that binary operations do not mix incompatible types.
///
/// This pass runs after `ValidatePass`, so `IrType::Infer` is guaranteed to
/// have been eliminated already.
pub struct TypeInferPass;

impl Pass for TypeInferPass {
    fn name(&self) -> &'static str {
        "type-infer"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in module.functions() {
            for block in func.blocks() {
                for instr in &block.instrs {
                    match instr {
                        IrInstr::BinOp { lhs, rhs, .. } => {
                            let lhs_ty = func.value_type(*lhs);
                            let rhs_ty = func.value_type(*rhs);
                            match (lhs_ty, rhs_ty) {
                                (Some(l), Some(r)) if l != r => {
                                    // Allow bool result from comparison of same base types —
                                    // the lowerer already converts the result ty to Bool.
                                    // Here we check operands only.
                                    return Err(PassError::TypeError {
                                        func: func.name.clone(),
                                        detail: format!(
                                            "binary op on mismatched types {} and {}",
                                            l, r
                                        ),
                                    });
                                }
                                _ => {}
                            }
                        }

                        IrInstr::UnaryOp { op, operand, .. } => {
                            if let Some(ty) = func.value_type(*operand) {
                                match op {
                                    ScalarUnaryOp::Neg => {
                                        if !matches!(
                                            ty,
                                            IrType::Scalar(
                                                DType::F32
                                                    | DType::F64
                                                    | DType::I32
                                                    | DType::I64
                                            )
                                        ) {
                                            return Err(PassError::TypeError {
                                                func: func.name.clone(),
                                                detail: format!(
                                                    "neg operand must be a numeric scalar, got {}",
                                                    ty
                                                ),
                                            });
                                        }
                                    }
                                    ScalarUnaryOp::Not => {
                                        if !matches!(ty, IrType::Scalar(DType::Bool)) {
                                            return Err(PassError::TypeError {
                                                func: func.name.clone(),
                                                detail: format!(
                                                    "not operand must be bool, got {}",
                                                    ty
                                                ),
                                            });
                                        }
                                    }
                                }
                            }
                        }

                        IrInstr::TensorOp {
                            op: TensorOp::Einsum { notation: _ },
                            inputs,
                            result_ty,
                            ..
                        } => {
                            // Validate all inputs are tensor types.
                            for &input in inputs {
                                if let Some(ty) = func.value_type(input) {
                                    if !matches!(ty, IrType::Tensor { .. }) {
                                        return Err(PassError::TypeError {
                                            func: func.name.clone(),
                                            detail: format!(
                                                "einsum input {} must be a tensor, got {}",
                                                input, ty
                                            ),
                                        });
                                    }
                                }
                            }

                            // Result must also be a tensor.
                            if !matches!(result_ty, IrType::Tensor { .. }) {
                                return Err(PassError::TypeError {
                                    func: func.name.clone(),
                                    detail: format!(
                                        "einsum result type must be a tensor, got {}",
                                        result_ty
                                    ),
                                });
                            }
                        }

                        IrInstr::Load { result_ty, .. } => {
                            // Load result must be a scalar.
                            if !matches!(result_ty, IrType::Scalar(_)) {
                                return Err(PassError::TypeError {
                                    func: func.name.clone(),
                                    detail: format!(
                                        "load result must be a scalar, got {}",
                                        result_ty
                                    ),
                                });
                            }
                        }

                        IrInstr::Cast { to_ty, .. } => {
                            // Cast result must be a scalar type.
                            if !matches!(to_ty, IrType::Scalar(_)) {
                                return Err(PassError::TypeError {
                                    func: func.name.clone(),
                                    detail: format!(
                                        "cast target type must be a scalar, got {}",
                                        to_ty
                                    ),
                                });
                            }
                        }

                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}
