//! Text emission for `GraphIr`.
//!
//! Output format (deterministic, NodeId order):
//!
//! ```text
//! // IRIS graph: Net
//!
//! model Net {
//!   input x: tensor<f32, [1, 784]>
//!
//!   layer h1  = Dense(x) { units=128 }
//!   layer out = Softmax(h1)
//!
//!   output out
//! }
//! ```

use std::fmt::Write;

use crate::codegen::CodegenError;
use crate::ir::graph::{GraphIr, GraphNode};
use crate::ir::types::IrType;

pub fn emit_graph_text(graph: &GraphIr) -> Result<String, CodegenError> {
    let mut out = String::new();

    writeln!(out, "// IRIS graph: {}", graph.name)?;
    writeln!(out)?;
    writeln!(out, "model {} {{", graph.name)?;

    // Inputs
    for node in graph.inputs() {
        if let GraphNode::Input { name, ty, .. } = node {
            writeln!(out, "  input {}: {}", name, format_ir_type(ty))?;
        }
    }

    // Layers
    let has_inputs = graph.inputs().count() > 0;
    if has_inputs {
        writeln!(out)?;
    }
    for node in graph.layers() {
        if let GraphNode::Layer { name, op, params, inputs, .. } = node {
            let input_names: Vec<&str> = inputs
                .iter()
                .filter_map(|id| graph.node(*id).map(|n| n.name()))
                .collect();
            let args = input_names.join(", ");
            if params.is_empty() {
                writeln!(out, "  layer {} = {}({})", name, op, args)?;
            } else {
                let param_str: Vec<String> = params
                    .iter()
                    .map(|p| format!("{}={}", p.key, p.value))
                    .collect();
                writeln!(
                    out,
                    "  layer {} = {}({}) {{ {} }}",
                    name,
                    op,
                    args,
                    param_str.join(", ")
                )?;
            }
        }
    }

    // Outputs
    let has_layers = graph.layers().count() > 0;
    if has_layers || has_inputs {
        writeln!(out)?;
    }
    for node in graph.outputs() {
        if let GraphNode::Output { name, .. } = node {
            writeln!(out, "  output {}", name)?;
        }
    }

    writeln!(out, "}}")?;

    Ok(out)
}

fn format_ir_type(ty: &IrType) -> String {
    ty.to_string()
}
