//! Binary ONNX protobuf emitter.
//!
//! Encodes an IRIS `GraphIr` as a minimal but valid ONNX `ModelProto` using
//! the hand-rolled protobuf encoder in `src/proto/mod.rs`.
//!
//! ONNX proto3 field numbers used:
//!   ModelProto:          ir_version=1, opset_import=7, graph=8
//!   OperatorSetIdProto:  domain=1, version=2
//!   GraphProto:          node=1, name=2, input=11, output=12
//!   NodeProto:           input=1, output=2, name=3, op_type=4, attribute=7
//!   ValueInfoProto:      name=1, type=2
//!   TypeProto:           tensor_type=1
//!   TypeProto::Tensor:   elem_type=1, shape=2
//!   TensorShapeProto:    dim=1
//!   TensorShapeProto::Dimension: dim_value=1, dim_param=2

use std::collections::HashMap;

use crate::error::CodegenError;
use crate::ir::graph::{GraphIr, GraphNode, NodeId, ParamValue};
use crate::ir::types::{Dim, DType, IrType};
use crate::proto::{encode_message_field, encode_string_field, encode_varint_field};

// ---------------------------------------------------------------------------
// ONNX element type codes (from onnx.proto TensorProto::DataType)
// ---------------------------------------------------------------------------

fn dtype_to_onnx_elem(dtype: DType) -> u64 {
    match dtype {
        DType::F32 => 1,
        DType::F64 => 11,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::Bool => 9,
    }
}

// ---------------------------------------------------------------------------
// ONNX op name mapping (same as text emitter)
// ---------------------------------------------------------------------------

fn onnx_op(iris_op: &str) -> &str {
    match iris_op {
        "Dense" | "Linear"  => "Gemm",
        "ReLU"              => "Relu",
        "GELU"              => "Gelu",
        "BatchNorm"         => "BatchNormalization",
        "LayerNorm"         => "LayerNormalization",
        "Conv2D"            => "Conv",
        "AvgPool"           => "AveragePool",
        "GlobalAveragePool" => "GlobalAveragePool",
        "GlobalMaxPool"     => "GlobalMaxPool",
        "Flatten"           => "Flatten",
        "Embedding"         => "Gather",
        other => other,
    }
}

// ---------------------------------------------------------------------------
// TypeProto encoding
// ---------------------------------------------------------------------------

/// Encode a TensorShapeProto::Dimension for a single dim.
fn encode_shape_dim(dim: &Dim) -> Vec<u8> {
    match dim {
        Dim::Literal(n) => encode_varint_field(1, *n),   // dim_value
        Dim::Symbolic(s) => encode_string_field(2, s),   // dim_param
    }
}

/// Encode a TensorShapeProto from a list of dims.
fn encode_tensor_shape(dims: &[Dim]) -> Vec<u8> {
    let mut out = Vec::new();
    for dim in dims {
        // Each dim is an embedded TensorShapeProto::Dimension (field 1).
        let dim_bytes = encode_shape_dim(dim);
        out.extend(encode_message_field(1, &dim_bytes));
    }
    out
}

/// Encode a TypeProto for a tensor type.
fn encode_type_proto(ty: &IrType) -> Vec<u8> {
    let tensor_inner = match ty {
        IrType::Tensor { dtype, shape } => {
            let mut inner = Vec::new();
            // elem_type = 1
            inner.extend(encode_varint_field(1, dtype_to_onnx_elem(*dtype)));
            // shape = 2
            if !shape.0.is_empty() {
                let shape_bytes = encode_tensor_shape(&shape.0);
                inner.extend(encode_message_field(2, &shape_bytes));
            }
            inner
        }
        IrType::Scalar(dtype) => {
            // Treat a scalar as a rank-0 tensor.
            encode_varint_field(1, dtype_to_onnx_elem(*dtype))
        }
        _ => Vec::new(),
    };
    // TypeProto.tensor_type = field 1 (embedded message)
    encode_message_field(1, &tensor_inner)
}

// ---------------------------------------------------------------------------
// ValueInfoProto
// ---------------------------------------------------------------------------

fn encode_value_info(name: &str, ty: &IrType) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(encode_string_field(1, name));   // name = 1
    let type_bytes = encode_type_proto(ty);
    out.extend(encode_message_field(2, &type_bytes)); // type = 2
    out
}

// ---------------------------------------------------------------------------
// AttributeProto (for node hyperparameters)
// ---------------------------------------------------------------------------

fn encode_attribute(key: &str, value: &ParamValue) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(encode_string_field(1, key)); // name = 1
    match value {
        ParamValue::Int(n) => {
            // type = 1 (INT), i = 4
            out.extend(encode_varint_field(20, 1)); // AttributeProto.type = 1 (INT)
            out.extend(encode_varint_field(4, *n as u64));
        }
        ParamValue::Float(v) => {
            // type = 2 (FLOAT), f = 4 (fixed32)
            // We encode floats as a fixed32 (wire type 5) — field 4
            out.extend(encode_varint_field(20, 2)); // type = 2 (FLOAT)
            let bits = (*v as f32).to_bits();
            let tag = ((4u64 << 3) | 5) as u8; // field 4, wire type 5 (32-bit)
            out.push(tag);
            out.extend_from_slice(&bits.to_le_bytes());
        }
        ParamValue::Bool(b) => {
            out.extend(encode_varint_field(20, 1)); // type = INT
            out.extend(encode_varint_field(4, *b as u64));
        }
        ParamValue::Str(s) => {
            out.extend(encode_varint_field(20, 8)); // type = STRING
            out.extend(encode_string_field(4, s));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// NodeProto
// ---------------------------------------------------------------------------

fn encode_node(
    op_type: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[(String, ParamValue)],
) -> Vec<u8> {
    let mut out = Vec::new();
    for inp in inputs {
        out.extend(encode_string_field(1, inp)); // input = 1
    }
    for outp in outputs {
        out.extend(encode_string_field(2, outp)); // output = 2
    }
    out.extend(encode_string_field(3, name));    // name = 3
    out.extend(encode_string_field(4, op_type)); // op_type = 4
    for (key, val) in attrs {
        let attr_bytes = encode_attribute(key, val);
        out.extend(encode_message_field(7, &attr_bytes)); // attribute = 7
    }
    out
}

// ---------------------------------------------------------------------------
// GraphProto
// ---------------------------------------------------------------------------

fn encode_graph(
    graph: &GraphIr,
    shapes: &HashMap<NodeId, IrType>,
) -> Vec<u8> {
    let mut out = Vec::new();

    // name = 2
    out.extend(encode_string_field(2, &graph.name));

    // node = 1 (repeated)
    for node in graph.layers() {
        if let GraphNode::Layer { op, inputs, params, name, .. } = node {
            let op_type = onnx_op(op);
            let input_names: Vec<&str> = inputs
                .iter()
                .filter_map(|pid| graph.nodes().iter().find(|n| n.id() == *pid))
                .map(|n| n.name())
                .collect();
            let attrs: Vec<(String, ParamValue)> = params
                .iter()
                .map(|p| (p.key.clone(), p.value.clone()))
                .collect();
            let node_bytes = encode_node(
                op_type,
                name,
                &input_names,
                &[name.as_str()],
                &attrs,
            );
            out.extend(encode_message_field(1, &node_bytes));
        }
    }

    // input = 11 (repeated)
    for node in graph.inputs() {
        if let GraphNode::Input { id, name, ty } = node {
            let resolved_ty = shapes.get(id).unwrap_or(ty);
            let vi = encode_value_info(name, resolved_ty);
            out.extend(encode_message_field(11, &vi));
        }
    }

    // output = 12 (repeated)
    for node in graph.outputs() {
        if let GraphNode::Output { from, name, .. } = node {
            if let Some(ty) = shapes.get(from) {
                let vi = encode_value_info(name, ty);
                out.extend(encode_message_field(12, &vi));
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// OperatorSetIdProto
// ---------------------------------------------------------------------------

fn encode_opset(domain: &str, version: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(encode_string_field(1, domain));     // domain = 1
    out.extend(encode_varint_field(2, version));    // version = 2
    out
}

// ---------------------------------------------------------------------------
// ModelProto — the top-level message
// ---------------------------------------------------------------------------

/// Encode `graph` as a binary ONNX `ModelProto`.
///
/// The result is valid ONNX protobuf and can be parsed by any conformant
/// ONNX runtime that supports opset 17.
pub fn emit_onnx_binary(
    graph: &GraphIr,
    shapes: &HashMap<NodeId, IrType>,
) -> Result<Vec<u8>, CodegenError> {
    let mut out = Vec::new();

    // ir_version = 1 (field 1, varint). ONNX IR version 7 = opset 17 era.
    out.extend(encode_varint_field(1, 7));

    // opset_import = 7 (field 7, repeated message). Use opset 17, domain="".
    let opset_bytes = encode_opset("", 17);
    out.extend(encode_message_field(7, &opset_bytes));

    // graph = 8 (field 8, embedded message).
    let graph_bytes = encode_graph(graph, shapes);
    out.extend(encode_message_field(8, &graph_bytes));

    Ok(out)
}
