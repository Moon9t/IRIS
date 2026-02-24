pub mod const_fold;
pub mod dead_node;
pub mod graph_pass;
pub mod opt;
pub mod shape_check;
pub mod shape_infer_graph;
pub mod type_infer;
pub mod validate;

pub use const_fold::ConstFoldPass;
pub use dead_node::DeadNodePass;
pub use graph_pass::{GraphPass, GraphPassManager};
pub use opt::{CsePass, DcePass, OpExpandPass};
pub use shape_check::ShapeCheckPass;
pub use shape_infer_graph::infer_shapes;

use crate::error::PassError;
use crate::ir::module::IrModule;

/// A compiler pass that operates on an `IrModule` in place.
///
/// Passes must be deterministic: given the same `IrModule`, the transformed
/// output must be identical across runs (no global mutable state, no randomness).
pub trait Pass {
    /// Human-readable name, used in error messages and diagnostics.
    fn name(&self) -> &'static str;

    /// Run the pass on the module.
    ///
    /// On success, the module is in a valid state for the next pass.
    /// On error, the module state is unspecified — the pipeline aborts.
    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError>;
}

/// Manages and executes an ordered sequence of compiler passes.
///
/// Passes run in the order they were registered. The pipeline aborts at the
/// first error. A failed validation pass means subsequent passes may produce
/// incorrect results, so aborting early is correct.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    /// If set, dumps IR text to stderr after the pass with this name completes.
    dump_after: Option<String>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new(), dump_after: None }
    }

    /// Appends a pass to the end of the pipeline.
    pub fn add_pass(&mut self, pass: impl Pass + 'static) {
        self.passes.push(Box::new(pass));
    }

    /// Configures the manager to dump IR to stderr after the named pass completes.
    pub fn set_dump_after(&mut self, pass_name: impl Into<String>) {
        self.dump_after = Some(pass_name.into());
    }

    /// Runs all passes in registration order on `module`.
    ///
    /// Returns `Err((pass_name, error))` at the first failure.
    pub fn run(&mut self, module: &mut IrModule) -> Result<(), (String, PassError)> {
        for pass in &mut self.passes {
            pass.run(module).map_err(|e| (pass.name().to_owned(), e))?;
            if let Some(ref target) = self.dump_after {
                if pass.name() == target.as_str() {
                    use crate::codegen::printer::emit_ir_text;
                    if let Ok(text) = emit_ir_text(module) {
                        eprintln!("--- IR after {} ---\n{}", pass.name(), text);
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns the names of all registered passes in pipeline order.
    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|p| p.name()).collect()
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}
