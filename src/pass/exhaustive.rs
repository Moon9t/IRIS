/// Phase 90b: Exhaustiveness checking pass.
///
/// Scans all `SwitchVariant` instructions and verifies that every variant
/// of the enum type is covered (or a `default_block` is present).
/// Reports a `PassError::TypeError` for non-exhaustive matches.
use crate::error::PassError;
use crate::ir::instr::IrInstr;
use crate::ir::module::IrModule;
use crate::ir::types::IrType;
use crate::pass::Pass;

pub struct ExhaustivePass;

impl Pass for ExhaustivePass {
    fn name(&self) -> &'static str { "exhaustive" }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in module.functions() {
            for block in func.blocks() {
                for instr in &block.instrs {
                    if let IrInstr::SwitchVariant { scrutinee, arms, default_block } = instr {
                        if default_block.is_some() {
                            // A default arm covers all unspecified variants.
                            continue;
                        }
                        // Look up the enum type from the scrutinee's value type.
                        let enum_ty = func.value_type(*scrutinee);
                        let num_variants = match enum_ty {
                            Some(IrType::Enum { variants, .. }) => variants.len(),
                            _ => continue, // can't determine — skip
                        };
                        let covered: std::collections::HashSet<usize> =
                            arms.iter().map(|(idx, _)| *idx).collect();
                        let missing: Vec<usize> = (0..num_variants)
                            .filter(|v| !covered.contains(v))
                            .collect();
                        if !missing.is_empty() {
                            return Err(PassError::TypeError {
                                func: func.name.clone(),
                                detail: format!(
                                    "non-exhaustive match: variant indices {:?} not covered",
                                    missing
                                ),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
