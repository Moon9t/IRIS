//! SSA validation pass.
//!
//! Checks structural correctness of an `IrModule` before any transformations.
//! This pass is intentionally conservative: it rejects anything it cannot
//! prove correct. Subsequent passes may relax constraints.

use std::collections::HashSet;

use crate::error::PassError;
use crate::ir::module::IrModule;
use crate::ir::types::IrType;
use crate::ir::value::ValueId;
use crate::pass::Pass;

/// Validates SSA invariants across the entire module.
///
/// Checks:
/// 1. Every value used in an instruction is defined before its first use
///    (linear scan within each function — sufficient for the block-param SSA
///    style the lowerer emits, where blocks appear in topological order).
/// 2. Every value is defined exactly once.
/// 3. Every block ends with exactly one terminator as its last instruction.
/// 4. No `IrType::Infer` values remain in the type map.
pub struct ValidatePass;

impl Pass for ValidatePass {
    fn name(&self) -> &'static str {
        "validate"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in module.functions() {
            let func_name = &func.name;

            // Check for unresolved Infer types anywhere in the value_types map.
            for (_value_id, ty) in &func.value_types {
                if matches!(ty, IrType::Infer) {
                    return Err(PassError::UnresolvedInfer {
                        func: func_name.clone(),
                    });
                }
            }

            // Track all defined ValueIds in program order (params then instrs,
            // block by block). This works because the lowerer emits blocks in
            // topological order and uses block-param SSA (no backward edges for
            // non-loop constructs).
            let mut defined: HashSet<ValueId> = HashSet::new();

            for block in func.blocks() {
                let block_label = block
                    .name
                    .as_deref()
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| format!("bb{}", block.id.0));

                // Block params are defined at block entry.
                for param in &block.params {
                    if !defined.insert(param.id) {
                        return Err(PassError::MultipleDefinition {
                            func: func_name.clone(),
                            value: format!("{}", param.id),
                        });
                    }
                }

                let n = block.instrs.len();
                for (i, instr) in block.instrs.iter().enumerate() {
                    // Terminator must be the last instruction.
                    if instr.is_terminator() && i != n - 1 {
                        return Err(PassError::MissingTerminator {
                            func: func_name.clone(),
                            block: block_label.clone(),
                        });
                    }

                    // All operands must be defined before this instruction.
                    for operand in instr.operands() {
                        if !defined.contains(&operand) {
                            return Err(PassError::UseBeforeDef {
                                func: func_name.clone(),
                                value: format!("{}", operand),
                            });
                        }
                    }

                    // Register this instruction's result as defined.
                    if let Some(result) = instr.result() {
                        if !defined.insert(result) {
                            return Err(PassError::MultipleDefinition {
                                func: func_name.clone(),
                                value: format!("{}", result),
                            });
                        }
                    }
                }

                // Block must end with a terminator.
                if !block.is_sealed() {
                    return Err(PassError::MissingTerminator {
                        func: func_name.clone(),
                        block: block_label,
                    });
                }
            }
        }
        Ok(())
    }
}
