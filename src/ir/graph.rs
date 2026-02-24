//! High-level computation graph IR for model definitions.
//!
//! `GraphIr` sits above `IrModule` in the compiler hierarchy. It represents a
//! model as an ordered list of named nodes (inputs → layers → outputs), with
//! sequential data flow. Lowering GraphIr → IrModule is deferred to Phase 3.

use std::collections::HashMap;

use crate::ir::types::IrType;

/// Opaque node identifier (index into `GraphIr::nodes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// A literal hyperparameter value attached to a layer.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Int(n) => write!(f, "{}", n),
            ParamValue::Float(v) => write!(f, "{}", v),
            ParamValue::Bool(b) => write!(f, "{}", b),
            ParamValue::Str(s) => write!(f, "\"{}\"", s),
        }
    }
}

/// A single `key = value` hyperparameter on a layer.
#[derive(Debug, Clone)]
pub struct LayerParam {
    pub key: String,
    pub value: ParamValue,
}

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub enum GraphNode {
    /// A model input: `input <name>: <ty>`
    Input {
        id: NodeId,
        name: String,
        ty: IrType,
    },
    /// A layer: `layer <name> <Op>(<params>)`
    Layer {
        id: NodeId,
        name: String,
        op: String,
        params: Vec<LayerParam>,
        /// Sequential: always exactly one element for Phase 2 models.
        inputs: Vec<NodeId>,
    },
    /// A model output: `output <name>`
    Output {
        id: NodeId,
        name: String,
        from: NodeId,
    },
}

impl GraphNode {
    pub fn id(&self) -> NodeId {
        match self {
            GraphNode::Input { id, .. } => *id,
            GraphNode::Layer { id, .. } => *id,
            GraphNode::Output { id, .. } => *id,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            GraphNode::Input { name, .. } => name,
            GraphNode::Layer { name, .. } => name,
            GraphNode::Output { name, .. } => name,
        }
    }

    pub fn set_id(&mut self, new_id: NodeId) {
        match self {
            GraphNode::Input { id, .. } => *id = new_id,
            GraphNode::Layer { id, .. } => *id = new_id,
            GraphNode::Output { id, .. } => *id = new_id,
        }
    }
}

/// High-level computation graph for a model.
///
/// Invariants:
/// - Node names are unique within the graph.
/// - `NodeId(n)` indexes `nodes[n]`.
/// - Input nodes appear first; Output nodes appear last.
pub struct GraphIr {
    pub name: String,
    pub(crate) nodes: Vec<GraphNode>,
    pub(crate) node_index: HashMap<String, NodeId>,
}

impl GraphIr {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            node_index: HashMap::new(),
        }
    }

    /// Add a node to the graph. Returns the assigned `NodeId`.
    /// Returns `Err` if a node with the same name already exists.
    pub fn add_node(&mut self, node: GraphNode) -> Result<NodeId, String> {
        let name = node.name().to_owned();
        if self.node_index.contains_key(&name) {
            return Err(format!("duplicate node name '{}'", name));
        }
        let id = NodeId(self.nodes.len() as u32);
        self.node_index.insert(name, id);
        self.nodes.push(node);
        Ok(id)
    }

    pub fn node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id.0 as usize)
    }

    pub fn node_by_name(&self, name: &str) -> Option<&GraphNode> {
        let id = self.node_index.get(name)?;
        self.nodes.get(id.0 as usize)
    }

    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    pub fn inputs(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes
            .iter()
            .filter(|n| matches!(n, GraphNode::Input { .. }))
    }

    pub fn layers(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes
            .iter()
            .filter(|n| matches!(n, GraphNode::Layer { .. }))
    }

    pub fn outputs(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes
            .iter()
            .filter(|n| matches!(n, GraphNode::Output { .. }))
    }
}
