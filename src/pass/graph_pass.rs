//! `GraphPass` trait and `GraphPassManager` for passes that operate on `GraphIr`.
//!
//! Mirrors `Pass` / `PassManager` in `mod.rs`, but the unit of transformation is
//! a `GraphIr` rather than an `IrModule`. Graph passes run before IR lowering.

use crate::error::PassError;
use crate::ir::graph::GraphIr;

/// A compiler pass that operates on a `GraphIr` in place.
pub trait GraphPass {
    /// Human-readable name, used in error messages.
    fn name(&self) -> &'static str;

    /// Run the pass on the graph.
    fn run(&mut self, graph: &mut GraphIr) -> Result<(), PassError>;
}

/// Manages and executes an ordered sequence of graph passes.
pub struct GraphPassManager {
    passes: Vec<Box<dyn GraphPass>>,
}

impl GraphPassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Appends a pass to the end of the pipeline.
    pub fn add_pass(&mut self, pass: impl GraphPass + 'static) {
        self.passes.push(Box::new(pass));
    }

    /// Runs all passes in registration order on `graph`.
    ///
    /// Returns `Err((pass_name, error))` at the first failure.
    pub fn run(&mut self, graph: &mut GraphIr) -> Result<(), (String, PassError)> {
        for pass in &mut self.passes {
            pass.run(graph).map_err(|e| (pass.name().to_owned(), e))?;
        }
        Ok(())
    }
}

impl Default for GraphPassManager {
    fn default() -> Self {
        Self::new()
    }
}
