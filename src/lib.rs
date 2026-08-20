//! IRIS: Intermediate Representation for Intelligent Systems.
//!
#![allow(clippy::collapsible_match)]
#![allow(clippy::useless_conversion)]

//! Compiler pipeline:
//!
//! ```text
//! source (.iris) → Lexer → [Tokens] → Parser → [AST]
//!   → Lowerer → [IrModule] → PassManager → Codegen → output
//! ```
//!
//! Passes (in order):
//!  1. `HmTypeInferPass`   — resolve remaining Infer placeholders (union-find)
//!  2. `ValidatePass`      — SSA structural correctness
//!  3. `TypeInferPass`     — type consistency
//!  4. `ConstFoldPass`     — constant arithmetic + identity simplification
//!  5. `StrengthReducePass`— strength reduction (pow→mul, div-by-const, etc.)
//!  6. `CopyPropPass`      — copy propagation
//!  7. `OpExpandPass`      — expand elementwise calls to TensorOp::Unary
//!  8. `LicmPass`          — loop-invariant code motion
//!  9. `InlinePass`        — inline small single-block callees
//! 10. `LoopUnrollPass`    — unroll constant-bound loops (≤8 iterations)
//! 11. `ExhaustivePass`    — exhaustive match checking for enums
//! 12. `DcePass`           — dead code elimination
//! 13. `CsePass`           — common subexpression elimination
//! 14. `ShapeCheckPass`    — tensor shape consistency
//! 15. `GcAnnotatePass`    — insert Retain/Release for heap-allocated values

pub mod bench;
pub mod cache;
pub mod cli;
pub mod codegen;
pub mod compiler;
pub mod dap;
pub mod debugger;
pub mod diagnostics;
pub mod docs;
pub mod error;
pub mod explain;
pub mod formatter;
pub mod interp;
pub mod ir;
pub mod lower;
pub mod lsp;
pub mod parser;
pub mod pass;
pub mod pkg;
pub mod package_manager;
pub mod preprocessor;
pub mod profiler;
pub mod proto;
pub mod repl;
pub mod runtime_bindings;
pub mod security;
pub mod setup;
pub mod stdlib;
pub mod test_runner;
pub mod upgrade;

pub mod agent;
pub mod inference;
pub mod rl;

pub use codegen::ir_serial::{deserialize_module, serialize_module};
pub use compiler::FileCompiler;
pub use debugger::{DebugSession, TraceEntry};
pub use error::Error;
pub use ir::module::IrModule;
pub use lsp::{LspDiagnostic, LspState};
pub use parser::ast::{AstBring, BringPath};
pub use pass::{
    CopyPropPass, ExhaustivePass, GcAnnotatePass, HmTypeInferPass, InlinePass, IrWarning, LicmPass,
    LoopUnrollPass, StrengthReducePass,
};
pub use repl::ReplState;

/// Compiles an IRIS source string with error recovery, returning a partial AST
/// and all accumulated parse errors. Useful for IDE/LSP workflows where you
/// want diagnostics for *every* error, not just the first.
pub fn compile_with_recovery(
    source: &str,
) -> (crate::parser::ast::AstModule, Vec<crate::error::ParseError>) {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;

    match Lexer::new(source).tokenize() {
        Ok(tokens) => {
            let mut parser = Parser::new(&tokens);
            parser.parse_module_recovering()
        }
        Err(e) => {
            // Lexer error — return empty module + the lex error.
            (
                crate::parser::ast::AstModule {
            private_items: std::collections::HashSet::new(),
                    enums: vec![],
                    structs: vec![],
                    functions: vec![],
                    models: vec![],
                    consts: vec![],
                    type_aliases: vec![],
                    traits: vec![],
                    impls: vec![],
                    effects: vec![],
                    brings: vec![],
                    extern_fns: vec![],
                    modules: vec![],
                    macros: vec![],
                },
                vec![e],
            )
        }
    }
}

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
    /// Compiled PTX text generated from the CUDA/NVPTX backend via clang.
    CudaPtx,
    /// SIMD-annotated LLVM IR: <N x T> vector types, AVX2 target, !llvm.loop metadata.
    Simd,
    /// JIT compilation: compile via LLVM/clang and run natively.
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
    /// TensorRT backend compiler target.
    TensorRt,
}

/// Compiles multiple IRIS source strings together, supporting `bring module_name`,
/// `bring "file.iris"`, and `bring std.name` to import public definitions.
///
/// `sources` is a slice of `(module_name, source_code)` pairs.
/// `main_module` is the name of the entry-point module.
pub fn compile_multi(
    sources: &[(&str, &str)],
    main_module: &str,
    emit: EmitKind,
) -> Result<String, Error> {
    let mut main_ast = compile_multi_to_ast(sources, main_module)?;
    compile_ast(&mut main_ast, main_module, emit, 1_000_000, 0, None)
}

/// Internal: parse+merge all brought modules into a single merged `AstModule`.
pub fn compile_multi_to_ast(
    sources: &[(&str, &str)],
    main_module: &str,
) -> Result<crate::parser::ast::AstModule, Error> {
    use std::collections::{HashMap, HashSet, VecDeque};

    // Parse all provided modules with error recovery.
    let mut parsed: HashMap<&str, crate::parser::ast::AstModule> = HashMap::new();
    for (name, src) in sources {
        let ast = parse_recovering(src)?;
        parsed.insert(name, ast);
    }

    // Remove the main module.
    let mut main_ast = parsed.remove(main_module).ok_or_else(|| {
        Error::Parse(crate::error::ParseError::UnexpectedToken {
            expected: format!("module named '{}'", main_module),
            found: "not found".to_owned(),
            span: crate::parser::lexer::Span::at(0),
        })
    })?;

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
        let dep_ast_opt: Option<crate::parser::ast::AstModule> =
            if let Some(lib_name) = key.strip_prefix("std:") {
                crate::stdlib::stdlib_source(lib_name)
                    .map(parse_recovering)
                    .transpose()?
            } else {
                // Key is the stem name (e.g., "utils" from "utils.iris" or legacy "utils").
                parsed.remove(key.as_str())
            };

        if let Some(mut dep) = dep_ast_opt {
            let mod_name = if let Some(lib_name) = key.strip_prefix("std:") {
                lib_name.replace(['.', '-'], "_")
            } else {
                key.replace(['.', '-'], "_")
            };
            crate::compiler::mangle_module_symbols(&mut dep, &mod_name);

            // Enqueue dep's own brings.
            for bring in &dep.brings {
                let dep_key = bring_key(&bring.path);
                if visited.insert(dep_key.clone()) {
                    queue.push_back(dep_key);
                }
            }
            // Merge all functions (including internal ones) and other definitions
            main_ast.private_items.extend(dep.private_items);
            main_ast.extern_fns.extend(dep.extern_fns);
            main_ast.functions.extend(dep.functions);
            main_ast.structs.extend(dep.structs);
            main_ast.enums.extend(dep.enums);
            main_ast.consts.extend(dep.consts);
            main_ast.type_aliases.extend(dep.type_aliases);
            main_ast.traits.extend(dep.traits);
            main_ast.impls.extend(dep.impls);
            main_ast.models.extend(dep.models);
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

/// Process-wide flag for strict effect checking.
///
/// Set by `--strict-effects` on the CLI (via [`set_strict_effects`]) or by the
/// `IRIS_STRICT_EFFECTS` environment variable. A static is used rather than a
/// parameter because `compile_ast` has many call sites across the CLI, REPL,
/// LSP and test harness, and the flag is a whole-process compilation mode.
static STRICT_EFFECTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enable strict effect checking for this process. Called by the CLI when
/// `--strict-effects` is passed.
pub fn set_strict_effects(on: bool) {
    STRICT_EFFECTS.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// True when strict effect checking is on, from either the flag or the env var.
pub fn strict_effects_enabled() -> bool {
    if STRICT_EFFECTS.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var("IRIS_STRICT_EFFECTS")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// In strict mode, an effect violation fails the build.
///
/// Effect diagnostics were previously printed to stderr and then ignored, so a
/// violating program still compiled and exited 0. A safety property that does
/// not fail the build cannot gate CI and cannot be cited as a guarantee, which
/// is the entire point of having it. Warnings (which do not start with
/// `error[`) remain advisory.
fn check_effect_gate(strict: bool, errors: &[String]) -> Result<(), Error> {
    if !strict {
        return Ok(());
    }
    let hard = errors.iter().filter(|e| e.starts_with("error[")).count();
    if hard == 0 {
        return Ok(());
    }
    Err(Error::Pass(crate::error::PassError::TypeError {
        func: "<effect checking>".into(),
        detail: format!(
            "{} effect violation{} found (strict mode)",
            hard,
            if hard == 1 { "" } else { "s" }
        ),
    }))
}

/// Internal: compile a pre-built `AstModule` through the full pipeline to an `IrModule`.
/// Used when building native binaries so we can pass the module to `build_binary`.
pub fn compile_ast_to_module(
    ast_module: &mut crate::parser::ast::AstModule,
    module_name: &str,
    dump_ir_after: Option<&str>,
) -> Result<IrModule, Error> {
    use crate::lower::{lower, lower_graph_to_ir, lower_model};
    use crate::pass::infer_shapes;
    use crate::pass::ast_exhaustive::AstExhaustivenessPass;
    use crate::pass::variance_checker::VarianceChecker;

    // Flatten inline modules before passes.
    crate::compiler::flatten_inline_modules(ast_module);

    // Inject default method bodies from trait definitions into impls.
    crate::compiler::inject_default_impl_methods(ast_module);

    // Desugar yield into list accumulator pattern before passes.
    crate::compiler::desugar_yield(ast_module);

    // Expand macro calls before any passes or lowering.
    crate::compiler::expand_macros(ast_module);

    // AST-level exhaustiveness checking before lowering.
    AstExhaustivenessPass::new().run(ast_module).map_err(Error::Pass)?;
    VarianceChecker::new().run(ast_module).map_err(Error::Pass)?;

    // Borrow checker: validate reference safety at compile time.
    {
        let mut checker = crate::pass::borrow_checker::BorrowChecker::new();
        checker.check_module(ast_module);
        if checker.has_errors() {
            checker.print_errors();
            return Err(Error::Pass(crate::error::PassError::TypeError {
                func: "<borrow checking>".into(),
                detail: format!("{} borrow error(s) found", checker.errors().len()),
            }));
        }
    }

    // Effect checker (non-strict by default; emits warnings only).
    // `--strict-effects` / IRIS_STRICT_EFFECTS=1 requires explicit `effect`
    // clauses on effectful functions AND that each clause covers what the body
    // actually does — and makes a violation fail the build.
    let strict_effects = strict_effects_enabled();
    {
        let mut effect_checker = crate::pass::effect_checker::EffectChecker::new(strict_effects);
        effect_checker.run(ast_module);
        for err in &effect_checker.errors {
            eprintln!("{}", err);
        }
        check_effect_gate(strict_effects, &effect_checker.errors)?;
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
    let mut pm = crate::pass::build_standard_pipeline();
    if let Some(pass_name) = dump_ir_after {
        pm.set_dump_after(pass_name);
    }
    pm.run(&mut ir_module).map_err(|(_, e)| Error::Pass(e))?;
    Ok(ir_module)
}

/// Internal: compile a pre-built `AstModule` through the full pipeline.
/// Size of the stack the compiler pipeline runs on.
///
/// 64 MiB. Lowering, monomorphisation and the AST passes all recurse over
/// program structure, and on real inputs that goes deeper than a small stack
/// allows. The effect was that the *same program* compiled from the CLI, whose
/// main thread gets 8 MiB, and crashed with STATUS_STACK_OVERFLOW inside a Rust
/// test thread, whose default is far smaller — so a defect appeared or vanished
/// depending on who called the compiler. Anyone embedding `iris::compile` from a
/// worker thread hit the same cliff.
const COMPILER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Default interpreter call-depth limit.
///
/// Empirical: a tail-recursive `sum_to` survives 300 frames and overflows
/// between 300 and 400 on a 64 MiB stack. Set below that so the guard produces
/// a diagnostic rather than letting the process die.
const INTERP_DEFAULT_MAX_DEPTH: usize = 250;

/// Runs the compiler pipeline on a thread with a guaranteed stack.
///
/// rustc does the same thing for the same reason. This makes available stack
/// depth a property of the compiler rather than of the caller; reducing the
/// recursion itself is still worthwhile, but no longer load-bearing for
/// correctness.
///
/// A scoped thread is used so the closure can borrow its arguments, and a panic
/// is re-raised on the calling thread so behaviour is unchanged.
fn with_compiler_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        // `spawn_scoped` consumes the closure, so a failed spawn leaves no way to
        // run it inline. Spawning a thread only fails when the OS is out of
        // resources, which nothing here could recover from anyway.
        let handle = std::thread::Builder::new()
            .stack_size(COMPILER_STACK_BYTES)
            .name("iris-compile".to_owned())
            .spawn_scoped(scope, f)
            .expect("iris: could not spawn the compiler thread");
        match handle.join() {
            Ok(value) => value,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

fn compile_ast(
    ast_module: &mut crate::parser::ast::AstModule,
    module_name: &str,
    emit: EmitKind,
    max_steps: usize,
    max_depth: usize,
    dump_ir_after: Option<&str>,
) -> Result<String, Error> {
    with_compiler_stack(|| {
        compile_ast_inner(ast_module, module_name, emit, max_steps, max_depth, dump_ir_after)
    })
}

fn compile_ast_inner(
    ast_module: &mut crate::parser::ast::AstModule,
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
    use crate::pass::{DeadNodePass, GraphPassManager};

    // Flatten inline modules before any processing.
    crate::compiler::flatten_inline_modules(ast_module);

    // Inject default method bodies from trait definitions into impls.
    crate::compiler::inject_default_impl_methods(ast_module);

    // Desugar yield into list accumulator pattern before passes.
    crate::compiler::desugar_yield(ast_module);

    // Expand macro calls before any passes or lowering.
    crate::compiler::expand_macros(ast_module);

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

    let mut ir_module = {
        // AST-level exhaustiveness checking before lowering
        use crate::pass::ast_exhaustive::AstExhaustivenessPass;
        use crate::pass::variance_checker::VarianceChecker;
        AstExhaustivenessPass::new().run(ast_module).map_err(Error::Pass)?;
        VarianceChecker::new().run(ast_module).map_err(Error::Pass)?;

        // Borrow checker: validate reference safety at compile time.
        {
            let mut checker = crate::pass::borrow_checker::BorrowChecker::new();
            checker.check_module(ast_module);
            if checker.has_errors() {
                checker.print_errors();
                return Err(Error::Pass(crate::error::PassError::TypeError {
                    func: "<borrow checking>".into(),
                    detail: format!("{} borrow error(s) found", checker.errors().len()),
                }));
            }
        }

        // Effect checker — see the note at the other call site.
        let strict_effects = strict_effects_enabled();
        {
            let mut effect_checker = crate::pass::effect_checker::EffectChecker::new(strict_effects);
            effect_checker.run(ast_module);
            for err in &effect_checker.errors {
                eprintln!("{}", err);
            }
            check_effect_gate(strict_effects, &effect_checker.errors)?;
        }
        lower(ast_module, module_name)?
    };

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

    let mut pm = crate::pass::build_standard_pipeline();
    if let Some(pass_name) = dump_ir_after {
        pm.set_dump_after(pass_name);
    }
    pm.run(&mut ir_module).map_err(|(_, e)| Error::Pass(e))?;

    match emit {
        EmitKind::Ir => Ok(emit_ir_text(&ir_module)?),
        EmitKind::Llvm | EmitKind::LlvmComplete | EmitKind::Binary => Ok(emit_llvm_ir(&ir_module)?),
        EmitKind::Cuda => Ok(emit_cuda(&ir_module)?),
        EmitKind::CudaPtx => Ok(crate::codegen::cuda::emit_cuda_ptx(&ir_module)?),
        EmitKind::Simd => Ok(emit_simd(&ir_module)?),
        EmitKind::Jit => Ok(emit_jit(&ir_module)?),
        EmitKind::PgoInstrument => Ok(emit_pgo_instrument(&ir_module)?),
        EmitKind::PgoOptimize => Ok(emit_pgo_optimize(&ir_module, "")?),
        EmitKind::Graph | EmitKind::Onnx | EmitKind::OnnxBinary => unreachable!(),
        EmitKind::Eval => eval_ir_module_internal(&ir_module, max_steps, max_depth),
        EmitKind::TensorRt => Ok(crate::codegen::tensorrt::emit_tensorrt(&ir_module)?),
    }
}

/// Compiles an IRIS source string to a fully-optimized `IrModule`.
///
/// Runs all standard passes (validate, type-infer, const-fold, strength-reduce,
/// op-expand, DCE, CSE, shape-check).  Useful before calling `serialize_module`.
pub fn compile_to_module(source: &str, module_name: &str) -> Result<IrModule, Error> {
    let ast_module = parse_recovering(source)?;
    let ir = crate::lower::lower(&ast_module, module_name)?;

    let mut pm = crate::pass::build_standard_pipeline();
    let mut ir = ir;
    pm.run(&mut ir).map_err(|(_, e)| Error::Pass(e))?;
    Ok(ir)
}

/// Compiles an IRIS source string to a `IrModule` suitable for debugging.
///
/// Runs standard passes but skips heavy optimizations (inlining, DCE, loop unrolling)
/// to preserve call frames, loop structures, and all local variables for the debugger.
pub fn compile_to_module_debug(source: &str, module_name: &str) -> Result<IrModule, Error> {
    let ast_module = parse_recovering(source)?;
    let ir = crate::lower::lower(&ast_module, module_name)?;
    use crate::pass::type_infer::TypeInferPass;
    use crate::pass::validate::ValidatePass;
    use crate::pass::{
        ConstFoldPass, CopyPropPass, ExhaustivePass, GcAnnotatePass, HmTypeInferPass, LicmPass,
        OpExpandPass, PassManager, ShapeCheckPass, StrengthReducePass,
    };
    let mut pm = PassManager::new();
    pm.add_pass(HmTypeInferPass);
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    pm.add_pass(ConstFoldPass);
    pm.add_pass(StrengthReducePass);
    pm.add_pass(CopyPropPass);
    pm.add_pass(OpExpandPass);
    // Tail-call elimination before LICM, so a self-recursive function has
    // become a loop by the time loop-invariant motion runs on it.
    pm.add_pass(crate::pass::tail_call::TailCallPass);
    pm.add_pass(LicmPass);
    pm.add_pass(ExhaustivePass);
    pm.add_pass(ShapeCheckPass);
    pm.add_pass(GcAnnotatePass);
    let mut ir = ir;
    pm.run(&mut ir).map_err(|(_, e)| Error::Pass(e))?;
    Ok(ir)
}

/// Evaluates a pre-built `IrModule` without re-running passes.
///
/// Finds the first zero-argument function and executes it via the native LLVM
/// pipeline, capturing stdout.
pub fn eval_ir_module(module: &IrModule) -> Result<String, Error> {
    eval_ir_module_internal(module, 0, 0)
}

/// `max_steps` / `max_depth` of 0 mean "use the defaults".
fn eval_ir_module_internal(
    module: &IrModule,
    max_steps: usize,
    max_depth: usize,
) -> Result<String, Error> {
    // Opt-in shortcut: skip building a native binary altogether. Compiling and
    // linking one program per evaluation dominates test-suite runtime, and the
    // interpreter produces the same answer, so a suite can trade native coverage
    // for a very large speedup.
    if codegen::build::force_interpreter() {
        return interpret_module_for_eval(module, max_steps, max_depth);
    }
    match codegen::execute_binary_for_eval(module) {
        Ok(s) => Ok(s),
        Err(e) => {
            eprintln!(
                "JIT/Native execution failed (falling back to interpreter): {:?}",
                e
            );
            match e {
                crate::error::CodegenError::Unsupported { backend, .. }
                    if backend == "native" || backend == "binary" =>
                {
                    interpret_module_for_eval(module, max_steps, max_depth)
                }
                other => Err(Error::Codegen(other)),
            }
        }
    }
}

/// Interpret a module's entry function for `--emit eval`.
///
/// Reproduces the native path's output shape exactly — printed lines first, then
/// the returned value(s) — so callers cannot tell which path ran. Previously this
/// returned only the return value, silently discarding everything the program
/// printed.
fn interpret_module_for_eval(
    module: &IrModule,
    max_steps: usize,
    max_depth: usize,
) -> Result<String, Error> {
    let func = module
        .functions()
        .iter()
        .find(|f| f.name == "main" && f.params.is_empty())
        .or_else(|| module.functions().iter().find(|f| f.params.is_empty()))
        .ok_or_else(|| {
            Error::Codegen(crate::error::CodegenError::Unsupported {
                backend: "native".into(),
                detail: "no zero-argument function found for eval".into(),
            })
        })?;
    let opts = crate::interp::InterpOptions {
        max_steps: if max_steps == 0 { 10_000_000 } else { max_steps },
        // Honour the caller's limit, and default it to something the stack can
        // actually take.
        //
        // This was hardcoded to 5_000 while the interpreter overflows its
        // 64 MiB stack at roughly 350 frames — about 190 KB of Rust stack per
        // IRIS call. The guard could therefore never fire: deep recursion
        // aborted the whole process with STATUS_STACK_OVERFLOW instead of
        // returning the "call depth exceeded" error that exists for exactly
        // this case. A crash where a diagnostic was already written.
        //
        // The safe depth depends on how much state each frame holds, so no
        // constant is right for every program; this is deliberately
        // conservative and `--max-depth` raises it. The real fix is to stop
        // consuming a Rust frame per IRIS frame — see known-issues #25.
        max_depth: if max_depth == 0 { INTERP_DEFAULT_MAX_DEPTH } else { max_depth },
    };
    let (result, printed) =
        crate::interp::eval_function_in_module_opts_capturing(module, func, &[], opts);
    let vals = result.map_err(Error::Interp)?;
    // `printed` already ends each line with a newline, so the return value lands
    // on its own line, matching the native binary's stdout.
    let mut out = printed;
    out.push_str(
        &vals
            .into_iter()
            .map(|v| match v {
                // Render a returned string bare, exactly as `print` does. The
                // `Display` impl quotes strings, and a native binary never emits
                // those quotes, so using it here made every str-returning
                // program disagree between the two paths.
                crate::interp::IrValue::Str(s) => s,
                other => format!("{}", other),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    );
    // Terminate the last line. A native binary's final `print` emits its own
    // newline, so its stdout always ends with one; this path did not, and the
    // two therefore differed by a single byte on *every* program that returns a
    // value. The backend-agreement gate reported the whole corpus as divergent
    // for that reason -- the same shape of finding as the `iris_codegen:` lines
    // on stdout (#62), and the reason this function's contract is "callers
    // cannot tell which path ran".
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

/// Parse source text with full error recovery, printing all errors to stderr
/// and returning the first as `Error::Parse`. Used by all in-memory compile paths.
fn parse_recovering(source: &str) -> Result<crate::parser::ast::AstModule, Error> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;
    let pp = crate::preprocessor::Preprocessor::new();
    let source = pp.process(source, "<source>").map_err(Error::Preprocessor)?;
    let tokens = Lexer::new(&source).tokenize()?;
    let mut parser = Parser::new(&tokens);
    let (module, errors) = parser.parse_module_recovering();
    if errors.is_empty() {
        return Ok(module);
    }
    for e in &errors {
        eprintln!("\x1b[1;31merror\x1b[0m: {}", e);
    }
    if errors.len() > 1 {
        eprintln!(
            "\x1b[1;31merror\x1b[0m: aborting due to {} parse error(s)",
            errors.len()
        );
    }
    Err(Error::Parse(
        errors
            .into_iter()
            .next()
            .expect("errors is non-empty, checked above"),
    ))
}

/// Compiles an IRIS source string through the full pipeline.
///
/// Returns the emitted output as a `String`, or an `Error` if any
/// stage fails. The pipeline aborts at the first error.
pub fn compile(source: &str, module_name: &str, emit: EmitKind) -> Result<String, Error> {
    let mut ast_module = parse_recovering(source)?;
    compile_ast(&mut ast_module, module_name, emit, 1_000_000, 0, None)
}

/// Compiles an IRIS source string and also returns dead-variable warnings.
///
/// Returns `(output, warnings)` on success, or an `Error` on failure.
pub fn compile_with_warnings(
    source: &str,
    module_name: &str,
    emit: EmitKind,
) -> Result<(String, Vec<IrWarning>), Error> {
    let mut ast_module = parse_recovering(source)?;
    let warnings = pass::find_unused_vars(&ast_module);
    let output = compile_ast(&mut ast_module, module_name, emit, 1_000_000, 0, None)?;
    Ok((output, warnings))
}

/// Like [`compile`] but with legacy execution guardrails for interpreter-based
/// tooling. Native outputs ignore `max_steps` and `max_depth`.
pub fn compile_with_opts(
    source: &str,
    module_name: &str,
    emit: EmitKind,
    max_steps: usize,
    max_depth: usize,
) -> Result<String, Error> {
    let mut ast_module = parse_recovering(source)?;
    compile_ast(&mut ast_module, module_name, emit, max_steps, max_depth, None)
}

/// Compiles an IRIS source string and on error returns a human-readable
/// diagnostic with source context (line number, source excerpt, caret pointer).
///
/// On success returns `Ok(output)`.  On failure returns `Err(diagnostic_string)`
/// instead of a structured `Error`, making it easy to display to end-users.
pub fn compile_with_diagnostics(
    source: &str,
    module_name: &str,
    emit: EmitKind,
) -> Result<String, String> {
    compile(source, module_name, emit).map_err(|e| diagnostics::render_error(source, &e))
}

/// Compiles an `.iris` file from disk, resolving all `bring` declarations
/// relative to the file's directory (and optional extra search paths).
///
/// Uses `FileCompiler` from `src/compiler.rs` internally.
pub fn compile_file(path: &std::path::Path, emit: EmitKind) -> Result<String, Error> {
    let mut main_ast = compiler::FileCompiler::new().compile_file_to_ast(path, &[])?;
    let module_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("main");
    compile_ast(&mut main_ast, module_name, emit, 1_000_000, 0, None)
}

/// Compiles an `.iris` file with bring resolution, using the provided `source`
/// text for the main file instead of reading it from disk.  Brings are still
/// resolved from disk relative to `file_path`'s directory.
pub fn compile_file_text(
    source: &str,
    file_path: &std::path::Path,
    emit: EmitKind,
) -> Result<String, Error> {
    let mut main_ast =
        compiler::FileCompiler::new().compile_file_to_ast_with_text(file_path, source, &[])?;
    let module_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("main");
    compile_ast(&mut main_ast, module_name, emit, 1_000_000, 0, None)
}

/// Like [`compile_file`] but returns the merged `IrModule` for further processing.
pub fn compile_file_to_module(path: &std::path::Path) -> Result<IrModule, Error> {
    let mut main_ast = compiler::FileCompiler::new().compile_file_to_ast(path, &[])?;
    let module_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("main");
    compile_ast_to_module(&mut main_ast, module_name, None)
}

/// Like [`compile_file`] but passes through all options including `dump_ir_after`.
pub fn compile_file_with_full_opts(
    path: &std::path::Path,
    emit: EmitKind,
    max_steps: usize,
    max_depth: usize,
    dump_ir_after: Option<&str>,
) -> Result<String, Error> {
    let mut main_ast = compiler::FileCompiler::new().compile_file_to_ast(path, &[])?;
    let module_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("main");
    compile_ast(
        &mut main_ast,
        module_name,
        emit,
        max_steps,
        max_depth,
        dump_ir_after,
    )
}

/// Like [`compile_file_to_module`] but passes through `dump_ir_after`.
pub fn compile_file_to_module_with_opts(
    path: &std::path::Path,
    dump_ir_after: Option<&str>,
) -> Result<IrModule, Error> {
    let mut main_ast = compiler::FileCompiler::new().compile_file_to_ast(path, &[])?;
    let module_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("main");
    compile_ast_to_module(&mut main_ast, module_name, dump_ir_after)
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
    let mut ast_module = parse_recovering(source)?;
    compile_ast(
        &mut ast_module,
        module_name,
        emit,
        max_steps,
        max_depth,
        dump_ir_after,
    )
}
