//! Shape consistency validation pass for `IrModule`.
//!
//! `ShapeCheckPass` walks every `TensorOp` instruction and verifies that the
//! input shapes are compatible with the declared output shape. It runs after
//! `OpExpandPass` so that expanded activations are checked as `TensorOp::Unary`
//! rather than as opaque `Call`s.
//!
//! Checks implemented:
//! - `Unary`     — input and output must have the same shape and dtype
//! - `Reshape`   — if all dims are concrete, total element count must be preserved
//! - `Transpose` — axes must be a valid permutation; output rank must equal input rank
//! - `Reduce`    — output rank must equal `input_rank − |axes|` (or same if keepdims)
//! - `Einsum`    — input count matches spec count; spec lengths match input ranks;
//!                  output indices are a subset of all input indices

use std::collections::HashSet;

use crate::error::PassError;
use crate::ir::instr::{IrInstr, TensorOp};
use crate::ir::module::IrModule;
use crate::ir::types::{Dim, IrType, Shape};
use crate::pass::Pass;

pub struct ShapeCheckPass;

impl Pass for ShapeCheckPass {
    fn name(&self) -> &'static str {
        "shape-check"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in module.functions() {
            for block in func.blocks() {
                for instr in &block.instrs {
                    if let IrInstr::TensorOp { op, inputs, result_ty, .. } = instr {
                        let input_tys: Vec<&IrType> =
                            inputs.iter().filter_map(|v| func.value_type(*v)).collect();
                        check_tensor_op(op, &input_tys, result_ty, &func.name)?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn check_tensor_op(
    op: &TensorOp,
    inputs: &[&IrType],
    result_ty: &IrType,
    func: &str,
) -> Result<(), PassError> {
    match op {
        TensorOp::Unary { .. } => {
            // Input and output must have the same shape and dtype.
            if let Some(first) = inputs.first() {
                if *first != result_ty && both_concrete(first, result_ty) {
                    return Err(PassError::ShapeMismatch {
                        func: func.to_owned(),
                        detail: format!(
                            "unary op: input type {} does not match output type {}",
                            first, result_ty
                        ),
                    });
                }
            }
        }

        TensorOp::Reshape => {
            // If all dimensions are literal, the total element count must be preserved.
            if let (
                Some(IrType::Tensor { shape: in_shape, .. }),
                IrType::Tensor { shape: out_shape, .. },
            ) = (inputs.first(), result_ty)
            {
                if in_shape.is_fully_concrete() && out_shape.is_fully_concrete() {
                    let in_n = shape_elements(in_shape);
                    let out_n = shape_elements(out_shape);
                    if in_n != out_n {
                        return Err(PassError::ShapeMismatch {
                            func: func.to_owned(),
                            detail: format!(
                                "reshape: input has {} elements but output has {}",
                                in_n, out_n
                            ),
                        });
                    }
                }
            }
        }

        TensorOp::Transpose { axes } => {
            if let Some(IrType::Tensor { shape, .. }) = inputs.first() {
                // Axes length must equal tensor rank.
                if axes.len() != shape.rank() {
                    return Err(PassError::ShapeMismatch {
                        func: func.to_owned(),
                        detail: format!(
                            "transpose: {} axes provided for rank-{} tensor",
                            axes.len(),
                            shape.rank()
                        ),
                    });
                }
                // Axes must form a valid permutation of 0..rank.
                let mut sorted = axes.clone();
                sorted.sort_unstable();
                let expected: Vec<usize> = (0..axes.len()).collect();
                if sorted != expected {
                    return Err(PassError::ShapeMismatch {
                        func: func.to_owned(),
                        detail: format!(
                            "transpose axes {:?} are not a valid permutation of 0..{}",
                            axes,
                            axes.len()
                        ),
                    });
                }
                // Output rank must match input rank.
                if let IrType::Tensor { shape: out_shape, .. } = result_ty {
                    if out_shape.rank() != shape.rank() {
                        return Err(PassError::ShapeMismatch {
                            func: func.to_owned(),
                            detail: format!(
                                "transpose: output rank {} != input rank {}",
                                out_shape.rank(),
                                shape.rank()
                            ),
                        });
                    }
                }
            }
        }

        TensorOp::Reduce { axes, keepdims, .. } => {
            if let (
                Some(IrType::Tensor { shape: in_shape, .. }),
                IrType::Tensor { shape: out_shape, .. },
            ) = (inputs.first(), result_ty)
            {
                let expected_rank = if *keepdims {
                    in_shape.rank()
                } else {
                    in_shape.rank().saturating_sub(axes.len())
                };
                if out_shape.rank() != expected_rank {
                    return Err(PassError::ShapeMismatch {
                        func: func.to_owned(),
                        detail: format!(
                            "reduce: expected output rank {}, got {}",
                            expected_rank,
                            out_shape.rank()
                        ),
                    });
                }
            }
        }

        TensorOp::Einsum { notation } => {
            // Validate the einsum notation string:
            // 1. Must contain exactly one "->".
            // 2. Input spec count must equal the number of tensor inputs.
            // 3. Each input spec length must match the corresponding input tensor rank.
            // 4. Every output index must appear in at least one input spec.
            let arrow = match notation.find("->") {
                Some(pos) => pos,
                None => {
                    return Err(PassError::ShapeMismatch {
                        func: func.to_owned(),
                        detail: format!(
                            "einsum notation {:?} is missing '->'",
                            notation
                        ),
                    });
                }
            };
            let input_part = &notation[..arrow];
            let output_part = &notation[arrow + 2..];

            // Split input specs on comma.
            let input_specs: Vec<&str> = if input_part.is_empty() {
                vec![]
            } else {
                input_part.split(',').collect()
            };

            // Check spec count matches actual input count.
            if input_specs.len() != inputs.len() {
                return Err(PassError::ShapeMismatch {
                    func: func.to_owned(),
                    detail: format!(
                        "einsum {:?}: notation has {} input specs but {} inputs provided",
                        notation,
                        input_specs.len(),
                        inputs.len()
                    ),
                });
            }

            // Check each input spec length matches the input tensor rank.
            let mut all_input_indices: HashSet<char> = HashSet::new();
            for (spec, inp_ty) in input_specs.iter().zip(inputs.iter()) {
                all_input_indices.extend(spec.chars());
                if let IrType::Tensor { shape, .. } = inp_ty {
                    if spec.len() != shape.rank() {
                        return Err(PassError::ShapeMismatch {
                            func: func.to_owned(),
                            detail: format!(
                                "einsum {:?}: input spec {:?} has {} indices but input has rank {}",
                                notation,
                                spec,
                                spec.len(),
                                shape.rank()
                            ),
                        });
                    }
                }
            }

            // Check every output index appears in at least one input spec.
            for ch in output_part.chars() {
                if !all_input_indices.contains(&ch) {
                    return Err(PassError::ShapeMismatch {
                        func: func.to_owned(),
                        detail: format!(
                            "einsum {:?}: output index {:?} does not appear in any input spec",
                            notation, ch
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Returns the total number of elements in a fully-concrete shape.
fn shape_elements(shape: &Shape) -> u64 {
    shape
        .0
        .iter()
        .map(|d| if let Dim::Literal(n) = d { *n } else { 1 })
        .product()
}

/// Returns `true` if both types are non-Infer tensors (worth comparing concretely).
fn both_concrete(a: &IrType, b: &IrType) -> bool {
    matches!(a, IrType::Tensor { .. }) && matches!(b, IrType::Tensor { .. })
}
