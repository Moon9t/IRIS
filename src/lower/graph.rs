//! Lowers an `AstModel` to a `GraphIr`.
//!
//! Data flow is sequential: each layer's input is the preceding layer's output
//! (or the last input if no layer has been declared yet). Output names must
//! reference a declared layer or input node.

use std::collections::HashMap;

use crate::error::LowerError;
use crate::ir::graph::{GraphIr, GraphNode, LayerParam, NodeId, ParamValue};
use crate::lower::lower_type;
use crate::parser::ast::{AstExpr, AstLayerParam, AstModel};

/// Lower an `AstModel` to a `GraphIr`.
pub fn lower_model(model: &AstModel) -> Result<GraphIr, LowerError> {
    let mut graph = GraphIr::new(&model.name.name);
    // Maps declared name → NodeId for output resolution.
    let mut name_to_node: HashMap<String, NodeId> = HashMap::new();
    // The NodeId of the most recently declared node (used as implicit input to layers).
    let mut prev_node: Option<NodeId> = None;

    // --- Inputs ---
    for input in &model.inputs {
        let ty = lower_type(&input.ty);
        let node = GraphNode::Input {
            id: NodeId(0), // placeholder; add_node assigns the real id
            name: input.name.name.clone(),
            ty,
        };
        let id = graph.add_node(node).map_err(|_| LowerError::DuplicateNode {
            name: input.name.name.clone(),
            span: input.name.span,
        })?;
        // Fix up id in the stored node.
        if let GraphNode::Input { id: stored_id, .. } = graph.nodes.last_mut().unwrap() {
            *stored_id = id;
        }
        name_to_node.insert(input.name.name.clone(), id);
        prev_node = Some(id);
    }

    // --- Layers ---
    for layer in &model.layers {
        let params = lower_layer_params(&layer.params)?;
        // If the layer declares explicit input refs, resolve them; otherwise fall
        // back to the implicit sequential predecessor.
        let inputs = if !layer.input_refs.is_empty() {
            layer
                .input_refs
                .iter()
                .map(|r| {
                    name_to_node.get(&r.name).copied().ok_or_else(|| LowerError::UndefinedLayer {
                        name: r.name.clone(),
                        span: r.span,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            match prev_node {
                Some(p) => vec![p],
                None => vec![],
            }
        };
        let node = GraphNode::Layer {
            id: NodeId(0),
            name: layer.name.name.clone(),
            op: layer.op.name.clone(),
            params,
            inputs,
        };
        let id = graph.add_node(node).map_err(|_| LowerError::DuplicateNode {
            name: layer.name.name.clone(),
            span: layer.name.span,
        })?;
        if let GraphNode::Layer { id: stored_id, .. } = graph.nodes.last_mut().unwrap() {
            *stored_id = id;
        }
        name_to_node.insert(layer.name.name.clone(), id);
        prev_node = Some(id);
    }

    // --- Outputs ---
    // Output nodes are terminal: nothing can reference them, so they are pushed
    // directly without inserting into node_index (avoiding name conflicts with
    // the layer they reference).
    for output in &model.outputs {
        let from = name_to_node
            .get(&output.name.name)
            .copied()
            .ok_or_else(|| LowerError::UndefinedLayer {
                name: output.name.name.clone(),
                span: output.name.span,
            })?;
        let id = NodeId(graph.nodes.len() as u32);
        graph.nodes.push(GraphNode::Output {
            id,
            name: output.name.name.clone(),
            from,
        });
    }

    Ok(graph)
}

fn lower_layer_params(params: &[AstLayerParam]) -> Result<Vec<LayerParam>, LowerError> {
    params.iter().map(lower_layer_param).collect()
}

fn lower_layer_param(param: &AstLayerParam) -> Result<LayerParam, LowerError> {
    let value = match &param.value {
        AstExpr::IntLit { value, .. } => ParamValue::Int(*value),
        AstExpr::FloatLit { value, .. } => ParamValue::Float(*value),
        AstExpr::BoolLit { value, .. } => ParamValue::Bool(*value),
        AstExpr::StringLit { value, .. } => ParamValue::Str(value.clone()),
        other => {
            return Err(LowerError::InvalidLayerParam {
                detail: "layer hyperparameters must be literals".to_owned(),
                span: other.span(),
            })
        }
    };
    Ok(LayerParam {
        key: param.key.name.clone(),
        value,
    })
}
