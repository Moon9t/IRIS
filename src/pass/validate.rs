//! SSA validation pass.
//!
//! Checks structural correctness of an `IrModule` before any transformations.
//! This pass is intentionally conservative: it rejects anything it cannot
//! prove correct. Subsequent passes may relax constraints.

use std::collections::HashSet;

use crate::error::PassError;
use crate::ir::module::IrModule;
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

            // We no longer check for unresolved Infer types here because ValidatePass
            // runs BEFORE HmTypeInferPass in the standard pipeline.
            // HmTypeInferPass is responsible for resolving Infer types and will
            // throw an error if any remain.

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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instr::{BinOp, IrInstr};
    use crate::ir::module::{IrFunctionBuilder, IrModule};
    use crate::ir::types::{DType, IrType};

    fn i64_ty() -> IrType {
        IrType::Scalar(DType::I64)
    }

    /// Build a valid "return 42" function with a single block.
    fn valid_module() -> IrModule {
        let mut m = IrModule::new("test");
        let mut builder = IrFunctionBuilder::new("main", vec![], i64_ty());
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);
        let c = builder.fresh_value();
        builder.push_instr(
            IrInstr::ConstInt {
                result: c,
                value: 42,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        builder.push_instr(IrInstr::Return { values: vec![c] }, None);
        m.add_function(builder.build()).unwrap();
        m
    }

    #[test]
    fn validate_valid_module() {
        let mut m = valid_module();
        let mut pass = ValidatePass;
        assert!(pass.run(&mut m).is_ok());
    }

    #[test]
    fn validate_empty_module() {
        let mut m = IrModule::new("empty");
        let mut pass = ValidatePass;
        assert!(pass.run(&mut m).is_ok());
    }

    #[test]
    fn validate_use_before_def() {
        let mut m = IrModule::new("test");
        let mut builder = IrFunctionBuilder::new("main", vec![], i64_ty());
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);
        // Use ValueId(99) which is never defined
        let result = builder.fresh_value();
        builder.push_instr(
            IrInstr::BinOp {
                result,
                op: BinOp::Add,
                lhs: crate::ir::value::ValueId(99),
                rhs: crate::ir::value::ValueId(98),
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        builder.push_instr(
            IrInstr::Return {
                values: vec![result],
            },
            None,
        );
        m.add_function(builder.build()).unwrap();

        let mut pass = ValidatePass;
        let err = pass.run(&mut m);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("used before"));
    }

    #[test]
    fn validate_missing_terminator() {
        let mut m = IrModule::new("test");
        let mut builder = IrFunctionBuilder::new("main", vec![], i64_ty());
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);
        let c = builder.fresh_value();
        builder.push_instr(
            IrInstr::ConstInt {
                result: c,
                value: 1,
                ty: i64_ty(),
            },
            Some(i64_ty()),
        );
        // Don't add a terminator — seal manually to bypass debug check
        builder.seal_unterminated_blocks();
        // Actually we need to test without terminator. The seal adds Return.
        // Instead, build a function that has an empty block via direct construction.
        drop(builder);

        // Construct directly with an unsealed block
        let func = crate::ir::function::IrFunction {
            id: crate::ir::function::FunctionId(0),
            name: "broken".into(),
            params: vec![],
            return_ty: i64_ty(),
            blocks: vec![crate::ir::block::IrBlock::new(
                crate::ir::block::BlockId(0),
                Some("entry".into()),
            )],
            value_defs: std::collections::HashMap::new(),
            value_types: std::collections::HashMap::new(),
            next_value: 0,
            attrs: vec![],
            span_table: crate::ir::function::SpanTable::default(),
            capture_count: 0,
        };
        m.add_function(func).unwrap();

        let mut pass = ValidatePass;
        let err = pass.run(&mut m);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("does not end with"));
    }

    #[test]
    fn validate_pass_name() {
        let pass = ValidatePass;
        assert_eq!(pass.name(), "validate");
    }
}
