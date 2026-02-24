//! Lowers a `GraphIr` + pre-computed shape map into an `IrFunction`.
//!
//! Each input node becomes a block parameter on the entry block. Each layer
//! node becomes an `IrInstr::Call` whose result type is taken from `shapes`.
//! Output nodes determine the `Return` values and the function's return type.
//!
//! The produced `IrFunction` uses the model name as the function name, so it
//! drops cleanly into an `IrModule` via `module.add_function()`.

use std::collections::HashMap;

use crate::error::LowerError;
use crate::ir::function::{IrFunction, Param};
use crate::ir::graph::{GraphIr, GraphNode, NodeId};
use crate::ir::instr::IrInstr;
use crate::ir::module::IrFunctionBuilder;
use crate::ir::types::IrType;
use crate::ir::value::ValueId;

/// Lower a `GraphIr` to an `IrFunction`.
///
/// `shapes` must map every `NodeId` in `graph` to its `IrType`
/// (as produced by `infer_shapes`).
pub fn lower_graph_to_ir(
    graph: &GraphIr,
    shapes: &HashMap<NodeId, IrType>,
) -> Result<IrFunction, LowerError> {
    // Determine the function return type from the first output node's source.
    let first_output = graph.outputs().next().ok_or_else(|| LowerError::UnknownOp {
        op: "model has no outputs".into(),
    })?;
    let return_ty = match first_output {
        GraphNode::Output { from, .. } => {
            shapes.get(from).cloned().unwrap_or(IrType::Infer)
        }
        _ => unreachable!(),
    };

    // Build function parameter list from input nodes (in NodeId order).
    let params: Vec<Param> = graph
        .inputs()
        .map(|n| Param {
            name: n.name().to_owned(),
            ty: shapes.get(&n.id()).cloned().unwrap_or(IrType::Infer),
        })
        .collect();

    let mut builder = IrFunctionBuilder::new(&graph.name, params, return_ty);
    let entry = builder.create_block(Some("entry"));

    // Add one block parameter per input node and record the mapping.
    let mut node_values: HashMap<NodeId, ValueId> = HashMap::new();
    for input in graph.inputs() {
        let ty = shapes.get(&input.id()).cloned().unwrap_or(IrType::Infer);
        let v = builder.add_block_param(entry, Some(input.name()), ty);
        node_values.insert(input.id(), v);
    }

    builder.set_current_block(entry);

    // Lower each layer to a Call instruction.
    for layer in graph.layers() {
        if let GraphNode::Layer { id, op, inputs, .. } = layer {
            let result_ty = shapes.get(id).cloned().unwrap_or(IrType::Infer);
            let args: Vec<ValueId> = inputs
                .iter()
                .filter_map(|pid| node_values.get(pid).copied())
                .collect();
            let result = builder.fresh_value();
            builder.push_instr(
                IrInstr::Call {
                    result: Some(result),
                    callee: op.clone(),
                    args,
                    result_ty: Some(result_ty.clone()),
                },
                Some(result_ty),
            );
            node_values.insert(*id, result);
        }
    }

    // Collect return values from output nodes.
    let return_vals: Vec<ValueId> = graph
        .outputs()
        .filter_map(|n| {
            if let GraphNode::Output { from, .. } = n {
                node_values.get(from).copied()
            } else {
                None
            }
        })
        .collect();

    builder.push_instr(IrInstr::Return { values: return_vals }, None);

    Ok(builder.build())
}
