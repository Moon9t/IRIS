//! Dead-node elimination for `GraphIr`.
//!
//! `DeadNodePass` performs a backward BFS from every `Output` node, collecting
//! all reachable ("live") nodes. Unreachable layer nodes are dropped. The
//! surviving nodes are renumbered sequentially and all `NodeId` references
//! (layer `inputs` and output `from` fields) are remapped accordingly.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::PassError;
use crate::ir::graph::{GraphIr, GraphNode, NodeId};
use crate::pass::graph_pass::GraphPass;

pub struct DeadNodePass;

impl GraphPass for DeadNodePass {
    fn name(&self) -> &'static str {
        "DeadNodeElim"
    }

    fn run(&mut self, graph: &mut GraphIr) -> Result<(), PassError> {
        // 1. Backward BFS from output nodes to collect live NodeIds.
        let mut live: HashSet<NodeId> = HashSet::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();

        for node in &graph.nodes {
            if let GraphNode::Output { id, from, .. } = node {
                live.insert(*id);
                queue.push_back(*from);
            }
        }
        while let Some(nid) = queue.pop_front() {
            if live.insert(nid) {
                // Find the node and enqueue its predecessors.
                if let Some(GraphNode::Layer { inputs, .. }) =
                    graph.nodes.iter().find(|n| n.id() == nid)
                {
                    for pred in inputs {
                        queue.push_back(*pred);
                    }
                }
            }
        }

        // 2. Drain nodes, keeping only live ones in original order.
        let live_nodes: Vec<GraphNode> =
            graph.nodes.drain(..).filter(|n| live.contains(&n.id())).collect();

        // 3. Assign new sequential NodeIds and build the remap table.
        let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
        let mut new_nodes: Vec<GraphNode> = Vec::with_capacity(live_nodes.len());
        for (new_idx, mut node) in live_nodes.into_iter().enumerate() {
            let old_id = node.id();
            let new_id = NodeId(new_idx as u32);
            remap.insert(old_id, new_id);
            node.set_id(new_id);
            new_nodes.push(node);
        }

        // 4. Remap all NodeId references using the remap table.
        for node in &mut new_nodes {
            match node {
                GraphNode::Layer { inputs, .. } => {
                    for inp in inputs {
                        *inp = remap[inp];
                    }
                }
                GraphNode::Output { from, .. } => {
                    *from = remap[from];
                }
                GraphNode::Input { .. } => {}
            }
        }

        // 5. Rebuild the graph.
        graph.nodes = new_nodes;
        graph.node_index.clear();
        for node in &graph.nodes {
            // Output nodes are not in node_index (they share names with their source).
            if !matches!(node, GraphNode::Output { .. }) {
                graph.node_index.insert(node.name().to_owned(), node.id());
            }
        }

        Ok(())
    }
}
