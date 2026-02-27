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
    /// Complete LLVM IR: named struct types, typed calls, alloca for fixed arrays.
    LlvmComplete,
    /// CUDA/NVPTX LLVM IR: kernel functions, thread/block IDs, !nvvm.annotations.
    Cuda,
    /// SIMD-annotated LLVM IR: <N x T> vector types, AVX2 target, !llvm.loop metadata.
    Simd,
    /// JIT compilation: compile via clang subprocess (or interpreter fallback) and run.
    Jit,
    /// PGO instrumented IR: block counters, @__llvm_profile_instrument_target.
    PgoInstrument,
    /// PGO optimized IR: branch weights from profile, hot/cold annotations.
    PgoOptimize,
    /// High-level computation graph text (for model definitions).
    Graph,
    /// Structural ONNX text stub (protobuf-text-style, no binary).
    Onnx,
    /// Execute the first function with no arguments and return the result as text.
    Eval,
    /// Binary ONNX protobuf (valid ModelProto bytes, base64-encoded for string return).
    OnnxBinary,
    /// Native binary: emit LLVM IR text intended for clang compilation via `build_binary()`.
    /// `compile()` returns the LLVM IR text; use `codegen::build_binary()` to produce an exe.
    Binary,
}

/// Compiles multiple IRIS source strings together, supporting `bring module_name`
/// to import public definitions from other modules.
///
/// `sources` is a slice of `(module_name, source_code)` pairs.
/// `main_module` is the name of the entry-point module (the one with the top-level `f()`).
pub fn compile_multi(sources: &[(&str, &str)], main_module: &str, emit: EmitKind) -> Result<String, Error> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;
    use std::collections::HashMap;

    // Parse all modules.
    let mut parsed: HashMap<&str, crate::parser::ast::AstModule> = HashMap::new();
    for (name, src) in sources {
        let tokens = Lexer::new(src).tokenize()?;
        let ast = Parser::new(&tokens).parse_module()?;
        parsed.insert(name, ast);
    }

    // Remove the main module and merge imported public definitions into it.
    let mut main_ast = parsed.remove(main_module)
        .ok_or_else(|| Error::Parse(crate::error::ParseError::UnexpectedToken {
            expected: format!("module named '{}'", main_module),
            found: "not found".to_owned(),
            span: crate::parser::lexer::Span::at(0),
        }))?;

    // Collect the bring list before mutably borrowing main_ast.
    let import_names: Vec<String> = main_ast.imports.clone();
    for mod_name in &import_names {
        if let Some(imported) = parsed.get(mod_name.as_str()) {
            for func in &imported.functions {
                if func.is_pub {
                    main_ast.functions.push(func.clone());
                }
            }
            // Always import structs, enums, type aliases, traits, impls, consts
            // (they are inherently public in this simple module system).
            main_ast.structs.extend(imported.structs.iter().cloned());
            main_ast.enums.extend(imported.enums.iter().cloned());
            main_ast.consts.extend(imported.consts.iter().cloned());
            main_ast.type_aliases.extend(imported.type_aliases.iter().cloned());
            main_ast.traits.extend(imported.traits.iter().cloned());
            main_ast.impls.extend(imported.impls.iter().cloned());
        }
    }

    compile_ast(&main_ast, main_module, emit, 1_000_000, 500, None)
}

/// Internal: compile a pre-built `AstModule` through the full pipeline.
fn compile_ast(
    ast_module: &crate::parser::ast::AstModule,
    module_name: &str,
    emit: EmitKind,
    max_steps: usize,
    max_depth: usize,
    dump_ir_after: Option<&str>,
) -> Result<String, Error> {
    use crate::codegen::cuda::emit_cuda;
    use crate::codegen::graph_printer::emit_graph_text;
    use crate::codegen::jit::emit_jit;
    use crate::codegen::llvm_ir::emit_llvm_ir;
    use crate::codegen::onnx::emit_onnx_text;
    use crate::codegen::onnx_binary::emit_onnx_binary;
    use crate::codegen::pgo::{emit_pgo_instrument, emit_pgo_optimize};
    use crate::codegen::printer::emit_ir_text;
    use crate::codegen::simd::emit_simd;
    use crate::lower::{lower, lower_graph_to_ir, lower_model};
    use crate::pass::infer_shapes;
    use crate::pass::type_infer::TypeInferPass;
    use crate::pass::validate::ValidatePass;
    use crate::pass::{ConstFoldPass, CsePass, DcePass, DeadNodePass, GraphPassManager, OpExpandPass, PassManager, ShapeCheckPass};

    if emit == EmitKind::Graph {
        let mut out = String::new();
        for model in &ast_module.models {
            let graph = lower_model(model)?;
            out.push_str(&emit_graph_text(&graph)?);
        }
        return Ok(out);
    }

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
                let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
                out.push_str(&hex);
            } else {
                out.push_str(&emit_onnx_text(&graph, &shapes)?);
            }
        }
        return Ok(out);
    }

    let mut ir_module = lower(ast_module, module_name)?;

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

    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    pm.add_pass(ConstFoldPass);
    pm.add_pass(OpExpandPass);
    pm.add_pass(DcePass);
    pm.add_pass(CsePass);
    pm.add_pass(ShapeCheckPass);
    if let Some(pass_name) = dump_ir_after {
        pm.set_dump_after(pass_name);
    }
    pm.run(&mut ir_module).map_err(|(_, e)| Error::Pass(e))?;

    match emit {
        EmitKind::Ir => Ok(emit_ir_text(&ir_module)?),
        EmitKind::Llvm | EmitKind::LlvmComplete | EmitKind::Binary => Ok(emit_llvm_ir(&ir_module)?),
        EmitKind::Cuda => Ok(emit_cuda(&ir_module)?),
        EmitKind::Simd => Ok(emit_simd(&ir_module)?),
        EmitKind::Jit => Ok(emit_jit(&ir_module)?),
        EmitKind::PgoInstrument => Ok(emit_pgo_instrument(&ir_module)?),
        EmitKind::PgoOptimize => Ok(emit_pgo_optimize(&ir_module, "")?),
        EmitKind::Graph | EmitKind::Onnx | EmitKind::OnnxBinary => unreachable!(),
        EmitKind::Eval => {
            let func = ir_module
                .functions()
                .iter()
                .find(|f| f.params.is_empty())
                .ok_or_else(|| {
                    Error::Interp(crate::error::InterpError::Unsupported {
                        detail: "no zero-argument function in module to evaluate".into(),
                    })
                })?;
            let opts = interp::InterpOptions { max_steps, max_depth };
            let results = interp::eval_function_in_module_opts(&ir_module, func, &[], opts)?;
            let mut out = String::new();
            for val in &results {
                // Skip unit/sentinel returns — programs that use print() for output
                // shouldn't also emit a spurious "0" from a `main() -> i64` sentinel.
                if matches!(val, interp::IrValue::Unit) {
                    continue;
                }
                out.push_str(&format!("{}\n", val));
            }
            Ok(out)
        }
    }
}

/// Compiles an IRIS source string through the full pipeline.
///
/// Returns the emitted output as a `String`, or an `Error` if any
/// stage fails. The pipeline aborts at the first error.
pub fn compile(source: &str, module_name: &str, emit: EmitKind) -> Result<String, Error> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;

    let tokens = Lexer::new(source).tokenize()?;
    let ast_module = Parser::new(&tokens).parse_module()?;
    compile_ast(&ast_module, module_name, emit, 1_000_000, 500, None)
}

/// Like [`compile`] but with configurable interpreter limits for `--emit eval`.
pub fn compile_with_opts(
    source: &str,
    module_name: &str,
    emit: EmitKind,
    max_steps: usize,
    max_depth: usize,
) -> Result<String, Error> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;

    let tokens = Lexer::new(source).tokenize()?;
    let ast_module = Parser::new(&tokens).parse_module()?;
    compile_ast(&ast_module, module_name, emit, max_steps, max_depth, None)
}

/// Like [`compile_with_opts`] but also supports `--dump-ir-after`.
pub fn compile_with_full_opts(
    source: &str,
    module_name: &str,
    emit: EmitKind,
    max_steps: usize,
    max_depth: usize,
    dump_ir_after: Option<&str>,
) -> Result<String, Error> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;

    let tokens = Lexer::new(source).tokenize()?;
    let ast_module = Parser::new(&tokens).parse_module()?;
    compile_ast(&ast_module, module_name, emit, max_steps, max_depth, dump_ir_after)
}
