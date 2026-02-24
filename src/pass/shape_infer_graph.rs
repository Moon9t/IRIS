//! Shape inference for `GraphIr`.
//!
//! `infer_shapes` walks nodes in declaration order (inputs → layers → outputs)
//! and returns a map from each `NodeId` to its `IrType`. Because the graph is
//! a DAG with nodes in topological order, a single forward pass is sufficient.
//!
//! Registered ops (Phase 4 + Phase 6):
//! - `Dense`, `Linear`    — replaces last dim with `units` hyperparameter
//! - `Softmax`, `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Add` — identity (same shape out)
//! - `Concat(axis=N)`     — sums dim[N] across all inputs
//! - `BatchNorm`, `Dropout`, `LayerNorm` — identity (passthrough shape)
//! - `MaxPool(stride=S)`, `AvgPool(stride=S)` — divide the last 2 spatial dims by S (default 2)
//! - `GlobalAveragePool`, `GlobalMaxPool` — collapse last 2 spatial dims to 1
//! - `Flatten` — collapse all non-batch dims to a single product dim
//! - `Conv2D(filters, kernel_size, stride, padding)` — NCHW spatial shrink
//! - `Embedding(embed_dim)` — lookup table: [B, S] → [B, S, embed_dim]
//!
//! Unknown ops return `LowerError::UnknownOp`. Use `--emit graph` for models
//! with ops not yet registered.

use std::collections::HashMap;

use crate::error::LowerError;
use crate::ir::graph::{GraphIr, GraphNode, LayerParam, NodeId, ParamValue};
use crate::ir::types::{Dim, IrType, Shape};

/// Infer the output `IrType` for every node in `graph`.
///
/// Nodes are processed in `NodeId` order, which matches declaration order:
/// inputs first, then layers, then outputs.
pub fn infer_shapes(graph: &GraphIr) -> Result<HashMap<NodeId, IrType>, LowerError> {
    let mut shapes: HashMap<NodeId, IrType> = HashMap::new();

    for node in graph.nodes() {
        match node {
            GraphNode::Input { id, ty, .. } => {
                shapes.insert(*id, ty.clone());
            }
            GraphNode::Layer { id, op, params, inputs, .. } => {
                let input_tys: Vec<&IrType> =
                    inputs.iter().filter_map(|pid| shapes.get(pid)).collect();
                let output_ty = infer_op_output_ty(op, params, &input_tys)?;
                shapes.insert(*id, output_ty);
            }
            GraphNode::Output { id, from, .. } => {
                let ty = shapes.get(from).cloned().unwrap_or(IrType::Infer);
                shapes.insert(*id, ty);
            }
        }
    }

    Ok(shapes)
}

/// Returns the output type for a layer op given all its input types.
fn infer_op_output_ty(
    op: &str,
    params: &[LayerParam],
    input_tys: &[&IrType],
) -> Result<IrType, LowerError> {
    let first = input_tys.first().copied().cloned().unwrap_or(IrType::Infer);
    match op {
        "Dense" | "Linear" => {
            let units = params
                .iter()
                .find(|p| p.key == "units")
                .and_then(|p| {
                    if let ParamValue::Int(n) = p.value {
                        Some(n as u64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| LowerError::UnknownOp {
                    op: format!("{op}: missing required 'units' integer parameter"),
                })?;
            match &first {
                IrType::Tensor { dtype, shape } => {
                    // Replace the last dimension with `units`.
                    let mut dims = shape.0[..shape.0.len().saturating_sub(1)].to_vec();
                    dims.push(Dim::Literal(units));
                    Ok(IrType::Tensor {
                        dtype: *dtype,
                        shape: Shape(dims),
                    })
                }
                _ => Err(LowerError::UnknownOp {
                    op: format!("{op}: input must be a tensor type"),
                }),
            }
        }

        // Unary/n-ary elementwise ops: output shape == first input shape.
        "Softmax" | "ReLU" | "Sigmoid" | "Tanh" | "GELU" | "Add" => Ok(first),

        // New Phase 6 passthrough ops — output shape equals input shape.
        "BatchNorm" | "Dropout" | "LayerNorm" => Ok(first),

        // MaxPool: divide the last 2 spatial dims by `stride` (default 2).
        // Only Literal dims are reduced; Symbolic dims are left unchanged.
        "MaxPool" => {
            let stride = get_param_int(params, "stride").unwrap_or(2) as u64;
            match &first {
                IrType::Tensor { dtype, shape } => {
                    let rank = shape.rank();
                    let mut dims = shape.0.clone();
                    let start = rank.saturating_sub(2);
                    for d in dims[start..].iter_mut() {
                        if let Dim::Literal(n) = d {
                            *n = n.saturating_div(stride);
                        }
                    }
                    Ok(IrType::Tensor { dtype: *dtype, shape: Shape(dims) })
                }
                _ => Err(LowerError::UnknownOp {
                    op: "MaxPool: input must be a tensor type".into(),
                }),
            }
        }

        // Concat(axis=N): sums dim[N] across all inputs.
        "Concat" => {
            let axis = params
                .iter()
                .find(|p| p.key == "axis")
                .and_then(|p| {
                    if let ParamValue::Int(n) = p.value {
                        Some(n as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);
            match &first {
                IrType::Tensor { dtype, shape } => {
                    let mut dims = shape.0.clone();
                    for ty in &input_tys[1..] {
                        if let IrType::Tensor { shape: s, .. } = ty {
                            if let (Some(Dim::Literal(a)), Some(Dim::Literal(b))) =
                                (dims.get(axis).cloned(), s.0.get(axis))
                            {
                                dims[axis] = Dim::Literal(a + b);
                            }
                        }
                    }
                    Ok(IrType::Tensor { dtype: *dtype, shape: Shape(dims) })
                }
                _ => Err(LowerError::UnknownOp {
                    op: "Concat: input must be a tensor type".into(),
                }),
            }
        }

        // AvgPool: same spatial downsampling formula as MaxPool.
        "AvgPool" => {
            let stride = get_param_int(params, "stride").unwrap_or(2) as u64;
            match &first {
                IrType::Tensor { dtype, shape } => {
                    let rank = shape.rank();
                    let mut dims = shape.0.clone();
                    let start = rank.saturating_sub(2);
                    for d in dims[start..].iter_mut() {
                        if let Dim::Literal(n) = d {
                            *n = n.saturating_div(stride);
                        }
                    }
                    Ok(IrType::Tensor { dtype: *dtype, shape: Shape(dims) })
                }
                _ => Err(LowerError::UnknownOp {
                    op: "AvgPool: input must be a tensor type".into(),
                }),
            }
        }

        // GlobalAveragePool / GlobalMaxPool: collapse last 2 spatial dims to 1.
        "GlobalAveragePool" | "GlobalMaxPool" => match &first {
            IrType::Tensor { dtype, shape } if shape.rank() >= 2 => {
                let mut dims = shape.0.clone();
                let rank = dims.len();
                dims[rank - 2] = Dim::Literal(1);
                dims[rank - 1] = Dim::Literal(1);
                Ok(IrType::Tensor { dtype: *dtype, shape: Shape(dims) })
            }
            _ => Ok(first),
        },

        // Flatten: collapse all dims except dim[0] into a single product.
        // [N, d1, d2, ...] → [N, d1*d2*...]
        "Flatten" => match &first {
            IrType::Tensor { dtype, shape } if shape.rank() >= 2 => {
                let flat: u64 = shape.0[1..]
                    .iter()
                    .map(|d| if let Dim::Literal(n) = d { *n } else { 1 })
                    .product();
                Ok(IrType::Tensor {
                    dtype: *dtype,
                    shape: Shape(vec![shape.0[0].clone(), Dim::Literal(flat)]),
                })
            }
            _ => Ok(first),
        },

        // Conv2D: NCHW convention — [N, C_in, H, W] → [N, filters, H_out, W_out]
        // H_out = (H + 2*padding - kernel_size) / stride + 1
        "Conv2D" => {
            let kernel  = get_param_int(params, "kernel_size").unwrap_or(3) as u64;
            let stride  = get_param_int(params, "stride").unwrap_or(1).max(1) as u64;
            let padding = get_param_int(params, "padding").unwrap_or(0) as u64;
            match &first {
                IrType::Tensor { dtype, shape } if shape.rank() == 4 => {
                    let filters = get_param_int(params, "filters")
                        .map(|f| f as u64)
                        .unwrap_or_else(|| {
                            if let Dim::Literal(c) = shape.0[1] { c } else { 1 }
                        });
                    let spatial = |n: u64| {
                        (n + 2 * padding).saturating_sub(kernel) / stride + 1
                    };
                    let h_out = if let Dim::Literal(h) = shape.0[2] { spatial(h) } else { 0 };
                    let w_out = if let Dim::Literal(w) = shape.0[3] { spatial(w) } else { 0 };
                    Ok(IrType::Tensor {
                        dtype: *dtype,
                        shape: Shape(vec![
                            shape.0[0].clone(),
                            Dim::Literal(filters),
                            Dim::Literal(h_out),
                            Dim::Literal(w_out),
                        ]),
                    })
                }
                _ => Err(LowerError::UnknownOp {
                    op: "Conv2D: requires a 4-D NCHW tensor input".into(),
                }),
            }
        }

        // Embedding: [batch, seq_len] → [batch, seq_len, embed_dim]
        "Embedding" => {
            let embed_dim = get_param_int(params, "embed_dim").unwrap_or(64) as u64;
            match &first {
                IrType::Tensor { dtype, shape } if shape.rank() == 2 => {
                    Ok(IrType::Tensor {
                        dtype: *dtype,
                        shape: Shape(vec![
                            shape.0[0].clone(),
                            shape.0[1].clone(),
                            Dim::Literal(embed_dim),
                        ]),
                    })
                }
                _ => Err(LowerError::UnknownOp {
                    op: "Embedding: requires a 2-D [batch, seq_len] input".into(),
                }),
            }
        }

        other => Err(LowerError::UnknownOp {
            op: other.to_owned(),
        }),
    }
}

/// Returns the integer value of a named hyperparameter, or `None` if absent
/// or non-integer.
fn get_param_int(params: &[LayerParam], key: &str) -> Option<i64> {
    params.iter().find(|p| p.key == key).and_then(|p| {
        if let ParamValue::Int(n) = p.value { Some(n) } else { None }
    })
}
