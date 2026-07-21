pub mod const_fold;
pub mod copy_prop;
pub mod dead_node;
pub mod effect_checker;
pub mod effect_registry;
pub mod exhaustive;
pub mod gc_annotate;
pub mod graph_pass;
pub mod inline;
pub mod licm;
pub mod lint;
pub mod loop_unroll;
pub mod opt;
pub mod shape_check;
pub mod shape_infer_graph;
pub mod strength_reduce;
pub mod type_infer;
pub mod type_infer_hm;
pub mod validate;
pub mod ast_exhaustive;
pub mod variance_checker;

pub use const_fold::ConstFoldPass;
pub use copy_prop::CopyPropPass;
pub use dead_node::DeadNodePass;
pub use exhaustive::ExhaustivePass;
pub use gc_annotate::GcAnnotatePass;
pub use graph_pass::{GraphPass, GraphPassManager};
pub use inline::InlinePass;
pub use licm::LicmPass;
pub use lint::{find_unused_vars, IrWarning};
pub use loop_unroll::LoopUnrollPass;
pub use opt::{CsePass, DcePass, OpExpandPass};
pub use shape_check::ShapeCheckPass;
pub use shape_infer_graph::infer_shapes;
pub use strength_reduce::StrengthReducePass;
pub use type_infer_hm::HmTypeInferPass;
pub use ast_exhaustive::AstExhaustivenessPass;

use std::collections::HashSet;
use crate::error::PassError;
use crate::ir::module::IrModule;
use crate::ir::value::ValueId;

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
        Self {
            passes: Vec::new(),
            dump_after: None,
        }
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
            // Debug: verify all Br/CondBr args are defined after each pass.
            if let Err(e) = verify_uses_defined(module, pass.name()) {
                return Err((pass.name().to_owned(), PassError::TypeError {
                    func: "verify_uses_defined".into(),
                    detail: e,
                }));
            }
            if pass.name() == "GcAnnotate" {
                if let Err(e) = verify_release_dominance(module) {
                    return Err((pass.name().to_owned(), PassError::TypeError {
                        func: "verify_release_dominance".into(),
                        detail: e,
                    }));
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::ir::module::IrModule;

    /// A no-op pass that always succeeds.
    struct NoopPass;
    impl Pass for NoopPass {
        fn name(&self) -> &'static str {
            "noop"
        }
        fn run(&mut self, _module: &mut IrModule) -> Result<(), PassError> {
            Ok(())
        }
    }

    /// A pass that always fails with a custom error.
    struct FailPass;
    impl Pass for FailPass {
        fn name(&self) -> &'static str {
            "fail"
        }
        fn run(&mut self, _module: &mut IrModule) -> Result<(), PassError> {
            Err(PassError::TypeError {
                func: "fail".into(),
                detail: "intentional failure".into(),
            })
        }
    }

    /// A pass that increments a counter on each invocation to verify mutation.
    #[allow(dead_code)]
    struct CountPass {
        count: usize,
    }
    impl Pass for CountPass {
        fn name(&self) -> &'static str {
            "count"
        }
        fn run(&mut self, _module: &mut IrModule) -> Result<(), PassError> {
            self.count += 1;
            Ok(())
        }
    }

    #[test]
    fn pass_manager_empty_pipeline() {
        let mut pm = PassManager::new();
        let mut module = IrModule::new("test");
        assert!(pm.run(&mut module).is_ok());
    }

    #[test]
    fn pass_manager_runs_all_passes() {
        let mut pm = PassManager::new();
        pm.add_pass(NoopPass);
        pm.add_pass(NoopPass);
        pm.add_pass(NoopPass);
        let mut module = IrModule::new("test");
        assert!(pm.run(&mut module).is_ok());
    }

    #[test]
    fn pass_manager_aborts_on_failure() {
        let mut pm = PassManager::new();
        pm.add_pass(NoopPass);
        pm.add_pass(FailPass);
        pm.add_pass(NoopPass); // should not run
        let mut module = IrModule::new("test");
        let result = pm.run(&mut module);
        assert!(result.is_err());
        let (name, _err) = result.unwrap_err();
        assert_eq!(name, "fail");
    }

    #[test]
    fn pass_manager_pass_names() {
        let mut pm = PassManager::new();
        pm.add_pass(NoopPass);
        pm.add_pass(FailPass);
        assert_eq!(pm.pass_names(), vec!["noop", "fail"]);
    }

    #[test]
    fn standard_pass_names() {
        // Verify that all the production passes have distinct names
        let names: Vec<&str> = vec![
            ConstFoldPass.name(),
            DeadNodePass.name(),
            ExhaustivePass.name(),
            GcAnnotatePass.name(),
            InlinePass::default().name(),
            LoopUnrollPass::default().name(),
            StrengthReducePass.name(),
            ShapeCheckPass.name(),
        ];
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(names.len(), unique.len(), "pass names must be unique");
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Standard Pipeline Factory
// ---------------------------------------------------------------------------

/// Builds the formalized 15-pass SSA pipeline sequence.
pub fn build_standard_pipeline() -> PassManager {
    let mut pm = PassManager::new();
    pm.add_pass(crate::pass::validate::ValidatePass);
    pm.add_pass(HmTypeInferPass);
    pm.add_pass(crate::pass::type_infer::TypeInferPass);
    pm.add_pass(ShapeCheckPass);
    pm.add_pass(InlinePass::default());
    pm.add_pass(ConstFoldPass);
    pm.add_pass(CopyPropPass);
    // GraphPass (DeadNodePass) is skipped in this IR pipeline
    pm.add_pass(LicmPass);
    pm.add_pass(LoopUnrollPass::default());
    pm.add_pass(StrengthReducePass);
    pm.add_pass(crate::pass::opt::OptPass);
    pm.add_pass(ExhaustivePass);
    pm.add_pass(GcAnnotatePass);
    pm.add_pass(crate::pass::lint::IrLintPass);
    pm
}

// ---------------------------------------------------------------------------
// Debug verification: check that all uses of value IDs in Br/CondBr are defined
// ---------------------------------------------------------------------------

/// Verifies every function in the module: all value IDs referenced in `Br`
/// and `CondBr` block arguments must exist as a result of some instruction
/// or as a block parameter in the same function.
/// Returns `Err` with a description on violation.
pub fn verify_uses_defined(module: &IrModule, after_pass: &str) -> Result<(), String> {
    for func in &module.functions {
        // Collect ALL defined values: block params + instruction results.
        let mut defined: HashSet<ValueId> = HashSet::new();
        for block in &func.blocks {
            for param in &block.params {
                defined.insert(param.id);
            }
            for instr in &block.instrs {
                if let Some(r) = instr.result() {
                    defined.insert(r);
                }
            }
        }
        for (bi, block) in func.blocks.iter().enumerate() {
            for (ii, instr) in block.instrs.iter().enumerate() {
                let check = |label: &str, args: &[ValueId]| -> Result<(), String> {
                    for (i, v) in args.iter().enumerate() {
                        if !defined.contains(v) {
                            let defs: Vec<u32> = defined.iter().map(|v| v.0).collect::<Vec<_>>();
                            return Err(format!(
                                "[after {}] {}: block{} instr{} {}[{}] = %{} not defined in function '{}'. defined IDs: {:?}",
                                after_pass, func.name, bi, ii, label, i, v.0, func.name, defs
                            ));
                        }
                    }
                    Ok(())
                };
                match instr {
                    crate::ir::instr::IrInstr::Br { args, .. } => {
                        check("Br", args)?;
                    }
                    crate::ir::instr::IrInstr::CondBr {
                        then_args,
                        else_args,
                        ..
                    } => {
                        check("CondBr.then", then_args)?;
                        check("CondBr.else", else_args)?;
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

/// Verifies that every `Release(v)` is dominated by the instruction or block parameter
/// that defines `v`.
pub fn verify_release_dominance(module: &IrModule) -> Result<(), String> {
    for func in &module.functions {
        if func.blocks.is_empty() {
            continue;
        }

        // 1. Find definition block for every value
        let mut def_blocks: std::collections::HashMap<crate::ir::value::ValueId, crate::ir::block::BlockId> = std::collections::HashMap::new();
        for block in &func.blocks {
            for param in &block.params {
                def_blocks.insert(param.id, block.id);
            }
            for instr in &block.instrs {
                if let Some(r) = instr.result() {
                    def_blocks.insert(r, block.id);
                }
            }
        }

        // 2. Compute dominators
        let block_ids: Vec<crate::ir::block::BlockId> = func.blocks.iter().map(|b| b.id).collect();
        let all_ids: HashSet<crate::ir::block::BlockId> = block_ids.iter().cloned().collect();
        let mut preds: std::collections::HashMap<crate::ir::block::BlockId, Vec<crate::ir::block::BlockId>> =
            block_ids.iter().map(|&b| (b, Vec::new())).collect();
            
        for block in &func.blocks {
            // Find successors
            let mut succs = Vec::new();
            for instr in &block.instrs {
                match instr {
                    crate::ir::instr::IrInstr::Br { target, .. } => succs.push(*target),
                    crate::ir::instr::IrInstr::CondBr { then_block, else_block, .. } => {
                        succs.push(*then_block);
                        succs.push(*else_block);
                    }
                    crate::ir::instr::IrInstr::SwitchVariant { arms, default_block, .. } => {
                        for (_, t) in arms { succs.push(*t); }
                        if let Some(def) = default_block { succs.push(*def); }
                    }
                    _ => {}
                }
            }
            for succ in succs {
                if let Some(p) = preds.get_mut(&succ) {
                    p.push(block.id);
                }
            }
        }

        let entry_id = block_ids[0];
        let mut dom: std::collections::HashMap<crate::ir::block::BlockId, HashSet<crate::ir::block::BlockId>> = std::collections::HashMap::new();
        let mut entry_set = HashSet::new();
        entry_set.insert(entry_id);
        dom.insert(entry_id, entry_set);
        for &bid in &block_ids[1..] {
            dom.insert(bid, all_ids.clone());
        }

        let mut changed = true;
        while changed {
            changed = false;
            for &bid in &block_ids[1..] {
                let preds_list = preds[&bid].clone();
                if preds_list.is_empty() {
                    continue;
                }
                let mut new_dom: HashSet<crate::ir::block::BlockId> = all_ids.clone();
                for p in &preds_list {
                    if let Some(pd) = dom.get(p) {
                        new_dom = new_dom.intersection(pd).cloned().collect();
                    }
                }
                new_dom.insert(bid);
                if new_dom != dom[&bid] {
                    dom.insert(bid, new_dom);
                    changed = true;
                }
            }
        }

        // 3. Check every Release
        for block in &func.blocks {
            for instr in &block.instrs {
                if let crate::ir::instr::IrInstr::Release { ptr, .. } = instr {
                    if let Some(def_block) = def_blocks.get(ptr) {
                        if let Some(dominators) = dom.get(&block.id) {
                            if !dominators.contains(def_block) {
                                return Err(format!("Release(v{}) in block {} is NOT dominated by its def in block {}", ptr.0, block.id.0, def_block.0));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

