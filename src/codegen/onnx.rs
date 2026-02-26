//! ONNX text export for `GraphIr`.
//!
//! `emit_onnx_text` produces a human-readable protobuf-text-style ONNX
//! representation. This is a structural stub — it faithfully represents the
//! graph topology and types without requiring a binary ONNX library.
//!
//! IRIS op → ONNX op_type mapping:
//! - `Dense`, `Linear` → `Gemm`
//! - `ReLU`            → `Relu`
//! - `GELU`            → `Gelu`
//! - `BatchNorm`       → `BatchNormalization`
//! - `LayerNorm`       → `LayerNormalization`
//! - others            → same name (Softmax, Sigmoid, Tanh, Add, Concat, MaxPool, Dropout)

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::CodegenError;
use crate::ir::graph::{GraphIr, GraphNode, NodeId, ParamValue};
use crate::ir::types::{DType, Dim, IrType};

/// Map an IRIS op name to the corresponding ONNX `op_type` string.
fn onnx_op(iris_op: &str) -> &str {
    match iris_op {
        "Dense" | "Linear" => "Gemm",
        "ReLU" => "Relu",
        "GELU" => "Gelu",
        "BatchNorm" => "BatchNormalization",
        "LayerNorm" => "LayerNormalization",
        "Conv2D" => "Conv",
        "AvgPool" => "AveragePool",
        "GlobalAveragePool" => "GlobalAveragePool",
        "GlobalMaxPool" => "GlobalMaxPool",
        "Flatten" => "Flatten",
        "Embedding" => "Gather",
        // MaxPool, Dropout, Softmax, Sigmoid, Tanh, Add, Concat: same name in ONNX
        other => other,
    }
}

/// Format an `IrType` as a compact ONNX type string.
fn fmt_onnx_type(ty: &IrType) -> String {
    match ty {
        IrType::Tensor { dtype, shape } => {
            let elem = dtype_to_onnx_elem(*dtype);
            let dims: Vec<String> = shape
                .0
                .iter()
                .map(|d| match d {
                    Dim::Literal(n) => format!("dim_value: {}", n),
                    Dim::Symbolic(s) => format!("dim_param: {:?}", s),
                })
                .collect();
            format!(
                "tensor_type {{ elem_type: {} shape {{ {} }} }}",
                elem,
                dims.join(" ")
            )
        }
        IrType::Scalar(dtype) => {
            format!(
                "tensor_type {{ elem_type: {} }}",
                dtype_to_onnx_elem(*dtype)
            )
        }
        IrType::Struct { name, .. } => format!("unknown_type {{ // struct {} }}", name),
        IrType::Enum { name, .. } => format!("unknown_type {{ // enum {} }}", name),
        IrType::Tuple(_) => "unknown_type { // tuple }".to_owned(),
        IrType::Str => "unknown_type { // str }".to_owned(),
        IrType::Array { elem, len } => format!("unknown_type {{ // array[{}; {}] }}", elem, len),
        IrType::Option(inner) => format!("unknown_type {{ // option<{}> }}", inner),
        IrType::ResultType(ok, err) => format!("unknown_type {{ // result<{},{}> }}", ok, err),
        IrType::Chan(elem) => format!("unknown_type {{ // chan<{}> }}", elem),
        IrType::Atomic(inner) => format!("unknown_type {{ // atomic<{}> }}", inner),
        IrType::Mutex(inner) => format!("unknown_type {{ // mutex<{}> }}", inner),
        IrType::Grad(inner) => format!("unknown_type {{ // grad<{}> }}", inner),
        IrType::Sparse(inner) => format!("unknown_type {{ // sparse<{}> }}", inner),
        IrType::List(inner) => format!("unknown_type {{ // list<{}> }}", inner),
        IrType::Map(k, v) => format!("unknown_type {{ // map<{}, {}> }}", k, v),
        IrType::Infer | IrType::Fn { .. } => "unknown_type {}".to_owned(),
    }
}

fn dtype_to_onnx_elem(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 1,
        DType::F64 => 11,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::Bool => 9,
    }
}

/// Emit a structural ONNX text representation of `graph`.
///
/// `shapes` must map every `NodeId` in `graph` to its `IrType`
/// (as produced by `infer_shapes`).
pub fn emit_onnx_text(
    graph: &GraphIr,
    shapes: &HashMap<NodeId, IrType>,
) -> Result<String, CodegenError> {
    let mut out = String::new();

    writeln!(out, "ir_version: 7")?;
    writeln!(out, "graph {{")?;
    writeln!(out, "  name: {:?}", graph.name)?;

    // One `node { ... }` block per layer.
    for node in graph.layers() {
        if let GraphNode::Layer {
            op,
            inputs,
            params,
            name,
            ..
        } = node
        {
            writeln!(out, "  node {{")?;
            writeln!(out, "    op_type: {:?}", onnx_op(op))?;

            // Resolve predecessor names for the input list.
            let input_names: Vec<&str> = inputs
                .iter()
                .filter_map(|pid| graph.nodes().iter().find(|n| n.id() == *pid))
                .map(|n| n.name())
                .collect();
            writeln!(out, "    input: {:?}", input_names)?;
            writeln!(out, "    output: {:?}", [name.as_str()])?;

            for p in params {
                let val_str = match &p.value {
                    ParamValue::Int(n) => format!("i: {}", n),
                    ParamValue::Float(v) => format!("f: {}", v),
                    ParamValue::Bool(b) => format!("i: {}", *b as i64),
                    ParamValue::Str(s) => format!("s: {:?}", s),
                };
                writeln!(out, "    attribute {{ name: {:?} {} }}", p.key, val_str)?;
            }
            writeln!(out, "  }}")?;
        }
    }

    // Graph-level input declarations.
    for node in graph.inputs() {
        if let GraphNode::Input { id, name, ty } = node {
            let ty_str = fmt_onnx_type(shapes.get(id).unwrap_or(ty));
            writeln!(out, "  input {{ name: {:?} type {{ {} }} }}", name, ty_str)?;
        }
    }

    // Graph-level output declarations.
    for node in graph.outputs() {
        if let GraphNode::Output { from, name, .. } = node {
            if let Some(ty) = shapes.get(from) {
                writeln!(
                    out,
                    "  output {{ name: {:?} type {{ {} }} }}",
                    name,
                    fmt_onnx_type(ty)
                )?;
            }
        }
    }

    writeln!(out, "}}")?;
    Ok(out)
}
