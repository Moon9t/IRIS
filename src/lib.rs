//! IRIS: Intermediate Representation for Intelligent Systems.
//!
//! Compiler pipeline:
//!
//! ```text
//! source (.iris) → Lexer → [Tokens] → Parser → [AST]
//!   → Lowerer → [IrModule] → PassManager → Codegen → output
//! ```
//!
//! Passes (in order):
//! 1. `ValidatePass`   — SSA structural correctness
//! 2. `TypeInferPass`  — type consistency
//! 3. `ConstFoldPass`  — constant arithmetic + identity simplification
//! 4. `OpExpandPass`   — expand elementwise calls to TensorOp::Unary
//! 5. `DcePass`        — dead code elimination
//! 6. `CsePass`        — common subexpression elimination
//! 7. `ShapeCheckPass` — tensor shape consistency

pub mod cli;
pub mod codegen;
pub mod diagnostics;
pub mod error;
pub mod interp;
pub mod ir;
pub mod lower;
pub mod parser;
pub mod pass;
pub mod proto;

pub use error::Error;

/// Controls what the `compile()` function emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitKind {
    /// Pretty-printed IRIS IR text.
    Ir,
    /// Scalar LLVM IR with full arithmetic, comparison, and control-flow bodies.
    Llvm,
    /// High-level computation graph text (for model definitions).
    Graph,
    /// Structural ONNX text stub (protobuf-text-style, no binary).
    Onnx,
    /// Execute the first function with no arguments and return the result as text.
    Eval,
    /// Binary ONNX protobuf (valid ModelProto bytes, base64-encoded for string return).
    OnnxBinary,
}

/// Compiles an IRIS source string through the full pipeline.
///
/// Returns the emitted output as a `String`, or an `Error` if any
/// stage fails. The pipeline aborts at the first error.
pub fn compile(source: &str, module_name: &str, emit: EmitKind) -> Result<String, Error> {
    use crate::codegen::graph_printer::emit_graph_text;
    use crate::codegen::llvm_stub::emit_llvm_stub;
    use crate::codegen::onnx::emit_onnx_text;
    use crate::codegen::onnx_binary::emit_onnx_binary;
    use crate::codegen::printer::emit_ir_text;
    use crate::lower::{lower, lower_graph_to_ir, lower_model};
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;
    use crate::pass::infer_shapes;
    use crate::pass::type_infer::TypeInferPass;
    use crate::pass::validate::ValidatePass;
    use crate::pass::{ConstFoldPass, CsePass, DcePass, DeadNodePass, GraphPassManager, OpExpandPass, PassManager, ShapeCheckPass};

    // 1. Lex
    let tokens = Lexer::new(source).tokenize()?;

    // 2. Parse
    let ast_module = Parser::new(&tokens).parse_module()?;

    // 3. Graph emit path (model DSL) — short-circuit before IR lowering.
    if emit == EmitKind::Graph {
        let mut out = String::new();
        for model in &ast_module.models {
            let graph = lower_model(model)?;
            out.push_str(&emit_graph_text(&graph)?);
        }
        return Ok(out);
    }

    // 4. ONNX emit paths — run dead-node elimination then emit text or binary.
    if emit == EmitKind::Onnx || emit == EmitKind::OnnxBinary {
        let mut out = String::new();
        for model in &ast_module.models {
            let mut graph = lower_model(model)?;
            let mut gpm = GraphPassManager::new();
            gpm.add_pass(DeadNodePass);
            gpm.run(&mut graph).map_err(|(_, e)| Error::Pass(e))?;
            let shapes = infer_shapes(&graph)?;
            if emit == EmitKind::OnnxBinary {
                let bytes = emit_onnx_binary(&graph, &shapes)?;
                // Encode as hex string for the text return value.
                let hex: String =
                    bytes.iter().map(|b| format!("{:02x}", b)).collect();
                out.push_str(&hex);
            } else {
                out.push_str(&emit_onnx_text(&graph, &shapes)?);
            }
        }
        return Ok(out);
    }

    // 5. Lower all fn definitions to IrFunctions.
    let mut ir_module = lower(&ast_module, module_name)?;

    // 6. Lower model definitions: shape-infer then convert to IrFunctions.
    for model in &ast_module.models {
        let graph = lower_model(model)?;
        let shapes = infer_shapes(&graph)?;
        let func = lower_graph_to_ir(&graph, &shapes)?;
        ir_module
            .add_function(func)
            .map_err(|_| crate::error::LowerError::DuplicateFunction {
                name: model.name.name.clone(),
                span: model.name.span,
            })?;
    }

    // 7. Run pass pipeline.
    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    pm.add_pass(ConstFoldPass);
    pm.add_pass(OpExpandPass);
    pm.add_pass(DcePass);
    pm.add_pass(CsePass);
    pm.add_pass(ShapeCheckPass);
    pm.run(&mut ir_module).map_err(|(_, e)| Error::Pass(e))?;

    // 8. Emit.
    match emit {
        EmitKind::Ir => Ok(emit_ir_text(&ir_module)?),
        EmitKind::Llvm => Ok(emit_llvm_stub(&ir_module)?),
        EmitKind::Graph | EmitKind::Onnx | EmitKind::OnnxBinary => unreachable!("handled above"),
        EmitKind::Eval => {
            let func = ir_module.functions().first().ok_or_else(|| {
                Error::Interp(crate::error::InterpError::Unsupported {
                    detail: "no functions in module to evaluate".into(),
                })
            })?;
            let results = interp::eval_function_in_module(&ir_module, func, &[])?;
            let mut out = String::new();
            for val in &results {
                out.push_str(&format!("{}\n", val));
            }
            Ok(out)
        }
    }
}
