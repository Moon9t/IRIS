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
pub mod compiler;
pub mod dap;
pub mod debugger;
pub mod diagnostics;
pub mod error;
pub mod interp;
pub mod ir;
pub mod lower;
pub mod lsp;
pub mod parser;
pub mod pass;
pub mod proto;
pub mod repl;
pub mod stdlib;

pub use codegen::ir_serial::{deserialize_module, serialize_module};
pub use compiler::FileCompiler;
pub use debugger::{DebugSession, TraceEntry};
pub use error::Error;
pub use ir::module::IrModule;
pub use lsp::{LspDiagnostic, LspState};
pub use parser::ast::{AstBring, BringPath};
pub use pass::{ExhaustivePass, GcAnnotatePass, HmTypeInferPass, InlinePass, IrWarning, LoopUnrollPass, StrengthReducePass};
pub use repl::ReplState;

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

/// Compiles multiple IRIS source strings together, supporting `bring module_name`,
/// `bring "file.iris"`, and `bring std.name` to import public definitions.
///
/// `sources` is a slice of `(module_name, source_code)` pairs.
/// `main_module` is the name of the entry-point module.
pub fn compile_multi(sources: &[(&str, &str)], main_module: &str, emit: EmitKind) -> Result<String, Error> {
    let main_ast = compile_multi_to_ast(sources, main_module)?;
    compile_ast(&main_ast, main_module, emit, 1_000_000, 500, None)
}

/// Internal: parse+merge all brought modules into a single merged `AstModule`.
pub fn compile_multi_to_ast(
    sources: &[(&str, &str)],
    main_module: &str,
) -> Result<crate::parser::ast::AstModule, Error> {
    use crate::parser::ast::BringPath;
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;
    use std::collections::{HashMap, HashSet, VecDeque};

    // Parse all provided modules.
    let mut parsed: HashMap<&str, crate::parser::ast::AstModule> = HashMap::new();
    for (name, src) in sources {
        let tokens = Lexer::new(src).tokenize()?;
        let ast = Parser::new(&tokens).parse_module()?;
        parsed.insert(name, ast);
    }

    // Remove the main module.
    let mut main_ast = parsed.remove(main_module)
        .ok_or_else(|| Error::Parse(crate::error::ParseError::UnexpectedToken {
            expected: format!("module named '{}'", main_module),
            found: "not found".to_owned(),
            span: crate::parser::lexer::Span::at(0),
        }))?;

    // BFS over brings; handles transitivity.
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    // Seed from main's brings.
    for bring in &main_ast.brings {
        let key = bring_key(&bring.path);
        if visited.insert(key.clone()) {
            queue.push_back(key);
        }
    }

    while let Some(key) = queue.pop_front() {
        // Try to resolve: first by File stem (look up in `parsed`), then by Stdlib.
        let dep_ast_opt: Option<crate::parser::ast::AstModule> = if key.starts_with("std:") {
            let lib_name = &key["std:".len()..];
            crate::stdlib::stdlib_source(lib_name)
                .map(|src| -> Result<_, Error> {
                    let tokens = Lexer::new(src).tokenize()?;
                    Ok(Parser::new(&tokens).parse_module()?)
                })
                .transpose()?
        } else {
            // Key is the stem name (e.g., "utils" from "utils.iris" or legacy "utils").
            parsed.remove(key.as_str())
        };

        if let Some(dep) = dep_ast_opt {
            // Enqueue dep's own brings.
            for bring in &dep.brings {
                let dep_key = bring_key(&bring.path);
                if visited.insert(dep_key.clone()) {
                    queue.push_back(dep_key);
                }
            }
            // Merge pub functions.
            for func in &dep.functions {
                if func.is_pub {
                    main_ast.functions.push(func.clone());
                }
            }
            // Merge structs/enums/consts/type_aliases/traits/impls (backward compat).
            main_ast.structs.extend(dep.structs.iter().cloned());
            main_ast.enums.extend(dep.enums.iter().cloned());
            main_ast.consts.extend(dep.consts.iter().cloned());
            main_ast.type_aliases.extend(dep.type_aliases.iter().cloned());
            main_ast.traits.extend(dep.traits.iter().cloned());
            main_ast.impls.extend(dep.impls.iter().cloned());
        }
    }

    Ok(main_ast)
}

/// Compute a lookup key from a `BringPath`.
fn bring_key(path: &crate::parser::ast::BringPath) -> String {
    use crate::parser::ast::BringPath;
    match path {
        BringPath::File(p) => {
            // Strip .iris extension to get the stem (module name).
            p.trim_end_matches(".iris").to_owned()
        }
        BringPath::Stdlib(name) => format!("std:{}", name),
    }
}

/// Internal: compile a pre-built `AstModule` through the full pipeline to an `IrModule`.
/// Used when building native binaries so we can pass the module to `build_binary`.
pub fn compile_ast_to_module(
    ast_module: &crate::parser::ast::AstModule,
    module_name: &str,
    dump_ir_after: Option<&str>,
) -> Result<IrModule, Error> {
    use crate::lower::{lower, lower_graph_to_ir, lower_model};
    use crate::pass::infer_shapes;
    use crate::pass::type_infer::TypeInferPass;
    use crate::pass::validate::ValidatePass;
    use crate::pass::{ConstFoldPass, CsePass, DcePass, OpExpandPass, PassManager, ShapeCheckPass, StrengthReducePass};

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
    pm.add_pass(StrengthReducePass);
    pm.add_pass(OpExpandPass);
    pm.add_pass(DcePass);
    pm.add_pass(CsePass);
    pm.add_pass(ShapeCheckPass);
    if let Some(pass_name) = dump_ir_after {
        pm.set_dump_after(pass_name);
    }
    pm.run(&mut ir_module).map_err(|(_, e)| Error::Pass(e))?;
    Ok(ir_module)
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
    use crate::pass::{ConstFoldPass, CsePass, DcePass, DeadNodePass, GraphPassManager, OpExpandPass, PassManager, ShapeCheckPass, StrengthReducePass};

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
    pm.add_pass(StrengthReducePass);
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
                // Str values are printed without surrounding quotes in eval output.
                match val {
                    interp::IrValue::Str(s) => out.push_str(&format!("{}\n", s)),
                    _ => out.push_str(&format!("{}\n", val)),
                }
            }
            Ok(out)
        }
    }
}

/// Compiles an IRIS source string to a fully-optimized `IrModule`.
///
/// Runs all standard passes (validate, type-infer, const-fold, strength-reduce,
/// op-expand, DCE, CSE, shape-check).  Useful before calling `serialize_module`.
pub fn compile_to_module(source: &str, module_name: &str) -> Result<IrModule, Error> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;

    let tokens = Lexer::new(source).tokenize()?;
    let ast_module = Parser::new(&tokens).parse_module()?;
    let ir = crate::lower::lower(&ast_module, module_name)?;
    // Run passes identical to compile_ast.
    use crate::pass::type_infer::TypeInferPass;
    use crate::pass::validate::ValidatePass;
    use crate::pass::{ConstFoldPass, CsePass, DcePass, OpExpandPass, PassManager, ShapeCheckPass, StrengthReducePass};
    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    pm.add_pass(ConstFoldPass);
    pm.add_pass(StrengthReducePass);
    pm.add_pass(OpExpandPass);
    pm.add_pass(DcePass);
    pm.add_pass(CsePass);
    pm.add_pass(ShapeCheckPass);
    let mut ir = ir;
    pm.run(&mut ir).map_err(|(_, e)| Error::Pass(e))?;
    Ok(ir)
}

/// Evaluates a pre-built `IrModule` without re-running passes.
///
/// Finds the first zero-argument function and runs the interpreter on it.
pub fn eval_ir_module(module: &IrModule) -> Result<String, Error> {
    let func = module
        .functions()
        .iter()
        .find(|f| f.params.is_empty())
        .ok_or_else(|| {
            Error::Interp(crate::error::InterpError::Unsupported {
                detail: "no zero-argument function in module".into(),
            })
        })?;
    let opts = interp::InterpOptions { max_steps: 1_000_000, max_depth: 500 };
    let results = interp::eval_function_in_module_opts(module, func, &[], opts)?;
    let mut out = String::new();
    for val in &results {
        if !matches!(val, interp::IrValue::Unit) {
            out.push_str(&format!("{}\n", val));
        }
    }
    Ok(out)
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

/// Compiles an IRIS source string and also returns dead-variable warnings.
///
/// Returns `(output, warnings)` on success, or an `Error` on failure.
pub fn compile_with_warnings(source: &str, module_name: &str, emit: EmitKind) -> Result<(String, Vec<IrWarning>), Error> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;

    let tokens = Lexer::new(source).tokenize()?;
    let ast_module = Parser::new(&tokens).parse_module()?;
    let warnings = pass::find_unused_vars(&ast_module);
    let output = compile_ast(&ast_module, module_name, emit, 1_000_000, 500, None)?;
    Ok((output, warnings))
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

/// Compiles an IRIS source string and on error returns a human-readable
/// diagnostic with source context (line number, source excerpt, caret pointer).
///
/// On success returns `Ok(output)`.  On failure returns `Err(diagnostic_string)`
/// instead of a structured `Error`, making it easy to display to end-users.
pub fn compile_with_diagnostics(source: &str, module_name: &str, emit: EmitKind) -> Result<String, String> {
    compile(source, module_name, emit)
        .map_err(|e| diagnostics::render_error(source, &e))
}

/// Compiles an `.iris` file from disk, resolving all `bring` declarations
/// relative to the file's directory (and optional extra search paths).
///
/// Uses `FileCompiler` from `src/compiler.rs` internally.
pub fn compile_file(path: &std::path::Path, emit: EmitKind) -> Result<String, Error> {
    let main_ast = compiler::FileCompiler::new().compile_file_to_ast(path, &[])?;
    let module_name = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    compile_ast(&main_ast, module_name, emit, 1_000_000, 500, None)
}

/// Like [`compile_file`] but returns the merged `IrModule` for further processing.
pub fn compile_file_to_module(path: &std::path::Path) -> Result<IrModule, Error> {
    let main_ast = compiler::FileCompiler::new().compile_file_to_ast(path, &[])?;
    let module_name = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    compile_ast_to_module(&main_ast, module_name, None)
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
