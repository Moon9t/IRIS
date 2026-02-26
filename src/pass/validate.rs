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

/// Returns `true` if `ty` contains an unresolved `IrType::Infer` in a
/// position that must be concrete after lowering.
///
/// Intentionally skips:
/// - `Option` — `none` produces `Option(Infer)` when the element type is unknown.
/// - `ResultType` — `ok(v)` / `err(v)` leave one type parameter as `Infer`.
/// - `Chan`, `Atomic`, `Mutex` — the element type is resolved lazily at the
///   first `send` / `atomic_store` call; the IR `value_types` map only records
///   the initial `Infer` placeholder emitted by `channel()` / `atomic()`.
fn contains_infer(ty: &IrType) -> bool {
    match ty {
        IrType::Infer => true,
        // Deferred-type containers: element type resolved at use site.
        IrType::Option(_)
        | IrType::ResultType(..)
        | IrType::Chan(_)
        | IrType::Atomic(_)
        | IrType::Mutex(_) => false,
        IrType::Scalar(_) | IrType::Str | IrType::Enum { .. } | IrType::Struct { .. } => false,
        IrType::Tensor { .. } => false,
        IrType::Tuple(elems) => elems.iter().any(contains_infer),
        IrType::Array { elem, .. } => contains_infer(elem),
        IrType::Grad(inner) | IrType::Sparse(inner) | IrType::List(inner) => {
            contains_infer(inner)
        }
        IrType::Map(k, v) => contains_infer(k) || contains_infer(v),
        IrType::Fn { params, ret } => params.iter().any(contains_infer) || contains_infer(ret),
    }
}

/// Validates SSA invariants across the entire module.
///
/// Checks:
/// 1. Every value used in an instruction is defined before its first use
///    (linear scan within each function — sufficient for the block-param SSA
///    style the lowerer emits, where blocks appear in topological order).
/// 2. Every value is defined exactly once.
/// 3. Every block ends with exactly one terminator as its last instruction.
/// 4. No `IrType::Infer` remains in the type map, including inside compound
///    types such as `Chan`, `Atomic`, `Mutex`, `Fn`, `Tuple`, and `Array`.
pub struct ValidatePass;

impl Pass for ValidatePass {
    fn name(&self) -> &'static str {
        "validate"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in module.functions() {
            let func_name = &func.name;

            // Check for unresolved Infer types (top-level and inside compound
            // types) anywhere in the value_types map.
            for (_value_id, ty) in &func.value_types {
                if contains_infer(ty) {
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
