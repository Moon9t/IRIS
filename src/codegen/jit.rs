//! JIT compilation backend for IRIS.
//!
//! Phase 52: Compiles IRIS IR to native machine code at runtime.
//!
//! Architecture
//! ─────────────
//! The JIT has three tiers:
//!
//! 1. **Native tier** (preferred): emits LLVM IR text via the `LlvmComplete`
//!    backend, then invokes an external `clang` process to compile it to a
//!    shared library (`.so`/`.dll`). The library is loaded with `dlopen`/
//!    `LoadLibrary` and the target function is called via `dlsym`/`GetProcAddress`.
//!    This tier requires `clang` to be in `PATH`.
//!
//! 2. **Interpreter tier** (fallback): uses the IRIS tree-walking interpreter
//!    directly. No native code is produced; pure Rust execution.
//!
//! 3. **Cached tier**: once a function has been JIT-compiled (either natively
//!    or via interpreter), results are cached in a `JitCache` for reuse within
//!    the same process.
//!
//! Usage
//! ──────
//! ```text
//! iris --emit jit program.iris
//! ```
//! This evaluates the first zero-argument function and prints the result,
//! choosing the fastest available tier automatically.
//!
//! JIT cache key: (module_name, function_name, ir_hash)

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::error::CodegenError;
use crate::ir::module::IrModule;
use crate::codegen::build::{
    find_clang,
    msys2_ucrt64_lib, msys2_ucrt64_include, msys2_gcc_lib,
    RUNTIME_H_SRC, RUNTIME_C_SRC,
};

// ---------------------------------------------------------------------------
// JIT cache
// ---------------------------------------------------------------------------

/// Identifier for a JIT-compiled function.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JitKey {
    pub module_name: String,
    pub function_name: String,
    /// Hash of the serialised IR for cache invalidation.
    pub ir_hash: u64,
}

/// Result of a JIT evaluation: the output text and the tier used.
#[derive(Debug, Clone)]
pub struct JitResult {
    pub output: String,
    pub tier: JitTier,
}

/// Which execution tier was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitTier {
    /// Native code via clang subprocess.
    Native,
    /// IRIS tree-walking interpreter.
    Interpreter,
}

impl std::fmt::Display for JitTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitTier::Native => f.write_str("native"),
            JitTier::Interpreter => f.write_str("interpreter"),
        }
    }
}

// ---------------------------------------------------------------------------
// JIT compiler
// ---------------------------------------------------------------------------

/// The JIT compiler — manages compilation and caching.
pub struct JitCompiler {
    cache: HashMap<JitKey, JitResult>,
}

impl JitCompiler {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    /// Compile and execute the first zero-argument function in `module`.
    ///
    /// Returns the output as a string (same format as `EmitKind::Eval`).
    pub fn compile_and_run(&mut self, module: &IrModule) -> Result<JitResult, CodegenError> {
        // Find the first zero-argument function.
        let func = module
            .functions()
            .iter()
            .find(|f| f.params.is_empty())
            .ok_or_else(|| CodegenError::Unsupported {
                backend: "jit".into(),
                detail: "no zero-argument function found to JIT-compile".into(),
            })?;

        let key = JitKey {
            module_name: module.name.clone(),
            function_name: func.name.clone(),
            ir_hash: hash_module(module),
        };

        // Cache hit.
        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }

        // Try native tier first.
        let result = if is_clang_available() {
            self.compile_native(module, &func.name)?
        } else {
            self.compile_interpreter(module)?
        };

        self.cache.insert(key, result.clone());
        Ok(result)
    }

    /// Native tier: emit LLVM IR → compile runtime + module → link → run.
    ///
    /// Mirrors the full `iris build` pipeline from `codegen::build` but writes
    /// output to a temp directory and executes the resulting binary, capturing
    /// its stdout.
    ///
    /// Unlike `iris build`, the JIT wrapper's `main()` **prints** the entry
    /// function's return value so stdout matches the interpreter tier.
    fn compile_native(&self, module: &IrModule, fn_name: &str) -> Result<JitResult, CodegenError> {
        use crate::codegen::llvm_ir::emit_llvm_ir;
        use crate::ir::types::{IrType, DType};
        use std::process::Command;

        // Use emit_llvm_ir (no binary wrapper) so functions keep their names.
        let mut ir_text = emit_llvm_ir(module)?;

        // Find the entry function's return type to generate correct print call.
        let entry_func = module
            .functions()
            .iter()
            .find(|f| f.name == fn_name)
            .or_else(|| module.functions().iter().find(|f| f.params.is_empty()));

        // Append a JIT-specific main() that calls the entry, prints its result,
        // and exits with code 0.
        if let Some(func) = entry_func {
            let (llvm_ret, print_call) = match &func.return_ty {
                IrType::Scalar(DType::I64) | IrType::Scalar(DType::I32)
                | IrType::Scalar(DType::U32) | IrType::Scalar(DType::U64)
                | IrType::Scalar(DType::USize)
                | IrType::Scalar(DType::I8) | IrType::Scalar(DType::U8) => {
                    ("i64", "  call void @iris_print_i64(i64 %r)\n")
                }
                IrType::Scalar(DType::F64) => {
                    ("double", "  call void @iris_print_f64(double %r)\n")
                }
                IrType::Scalar(DType::F32) => {
                    // Runtime prints f64; extend f32 → f64 first.
                    ("float", "  %rd = fpext float %r to double\n  call void @iris_print_f64(double %rd)\n")
                }
                IrType::Scalar(DType::Bool) => {
                    ("i1", "  call void @iris_print_bool(i1 %r)\n")
                }
                IrType::Str => {
                    ("ptr", "  call void @iris_print_str(ptr %r)\n")
                }
                _ => ("i64", "  call void @iris_print_i64(i64 %r)\n"),
            };

            ir_text.push_str(&format!(
                "\ndefine i32 @main(i32 %argc, ptr %argv) {{\n\
                 {}\
                 %r = call {} @{}()\n\
                 {}\
                 ret i32 0\n\
                 }}\n",
                "  call void @iris_set_argv(i32 %argc, ptr %argv)\n",
                llvm_ret, func.name, print_call
            ));
        }

        // Per-process + per-hash directory so parallel tests don't collide.
        let unique = format!("iris_jit_{}_{}", std::process::id(), hash_module(module));
        let tmp_dir = std::env::temp_dir().join(&unique);
        let _ = std::fs::create_dir_all(&tmp_dir);

        let ir_path = tmp_dir.join("module.ll");
        let exe_ext = if cfg!(target_os = "windows") { "exe" } else { "out" };
        let out_path = tmp_dir.join(format!("module.{}", exe_ext));

        std::fs::write(&ir_path, &ir_text).map_err(|e| CodegenError::Unsupported {
            backend: "jit-native".into(),
            detail: format!("cannot write temp IR file: {}", e),
        })?;

        // Write the embedded IRIS runtime (iris_runtime.h + iris_runtime.c)
        // which provides iris_print_i64, iris_set_argv, etc.
        let h_path = tmp_dir.join("iris_runtime.h");
        let c_path = tmp_dir.join("iris_runtime.c");
        std::fs::write(&h_path, RUNTIME_H_SRC).map_err(|e| CodegenError::Unsupported {
            backend: "jit-native".into(),
            detail: format!("cannot write runtime header: {}", e),
        })?;
        std::fs::write(&c_path, RUNTIME_C_SRC).map_err(|e| CodegenError::Unsupported {
            backend: "jit-native".into(),
            detail: format!("cannot write runtime C source: {}", e),
        })?;

        // Locate clang and set up environment.
        let clang = find_clang();
        let msys2_lib = msys2_ucrt64_lib();
        let msys2_inc = msys2_ucrt64_include();
        let gcc_lib = msys2_gcc_lib();

        // Common target triple for all clang invocations on Windows.
        let target_args: Vec<&str> = if cfg!(target_os = "windows") {
            vec!["-target", "x86_64-w64-windows-gnu"]
        } else {
            vec![]
        };

        // Step 1: Compile iris_runtime.c → iris_runtime.o (using clang).
        let rt_obj = tmp_dir.join("iris_runtime.o");
        let mut rt_cmd = Command::new(&clang);
        rt_cmd.args(&target_args);
        rt_cmd.args([
            "-O2", "-c",
            c_path.to_str().unwrap_or(""),
            "-o", rt_obj.to_str().unwrap_or(""),
            "-I", tmp_dir.to_str().unwrap_or(""),
            "-Wno-pragma-pack",
        ]);
        if let Some(ref inc) = msys2_inc {
            rt_cmd.arg("-I").arg(inc);
        }
        let rt_result = rt_cmd.stderr(std::process::Stdio::null()).output();
        if !matches!(&rt_result, Ok(o) if o.status.success()) {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return self.compile_interpreter(module);
        }

        // Step 2: Compile LLVM IR → module.o (using clang).
        let mod_obj = tmp_dir.join("module.o");
        let mut ir_cmd = Command::new(&clang);
        ir_cmd.args(&target_args);
        ir_cmd.args([
            "-O2", "-c",
            ir_path.to_str().unwrap_or(""),
            "-o", mod_obj.to_str().unwrap_or(""),
            "-Wno-override-module",
        ]);
        let ir_result = ir_cmd.stderr(std::process::Stdio::null()).output();
        if !matches!(&ir_result, Ok(o) if o.status.success()) {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return self.compile_interpreter(module);
        }

        // Step 3: Link module.o + iris_runtime.o → executable (using clang + lld).
        let mut link_cmd = Command::new(&clang);
        link_cmd.args(&target_args);
        link_cmd.args([
            "-fuse-ld=lld",
            "-O2",
            mod_obj.to_str().unwrap_or(""),
            rt_obj.to_str().unwrap_or(""),
            "-o", out_path.to_str().unwrap_or(""),
            "-lm", "-lpthread",
        ]);
        if let Some(ref lib) = msys2_lib {
            link_cmd.arg(format!("-L{}", lib));
        }
        if let Some(ref lib) = gcc_lib {
            link_cmd.arg(format!("-L{}", lib));
        }
        let link_output = link_cmd.stderr(std::process::Stdio::null()).output();
        if !matches!(&link_output, Ok(o) if o.status.success()) {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return self.compile_interpreter(module);
        }

        // Step 4: Run the compiled executable and capture stdout.
        let run_path = if cfg!(target_os = "windows") {
            std::fs::canonicalize(&out_path).unwrap_or_else(|_| out_path.clone())
        } else {
            out_path.clone()
        };

        let run_output = Command::new(&run_path)
            .output()
            .map_err(|e| CodegenError::Unsupported {
                backend: "jit-native".into(),
                detail: format!("cannot run compiled binary: {}", e),
            })?;

        let stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
        let _ = std::fs::remove_dir_all(&tmp_dir);

        Ok(JitResult {
            output: stdout,
            tier: JitTier::Native,
        })
    }

    /// Interpreter tier: use the IRIS tree-walking interpreter.
    fn compile_interpreter(&self, module: &IrModule) -> Result<JitResult, CodegenError> {
        use crate::interp::eval_function_in_module;

        let func = module
            .functions()
            .iter()
            .find(|f| f.params.is_empty())
            .ok_or_else(|| CodegenError::Unsupported {
                backend: "jit-interp".into(),
                detail: "no zero-argument function found".into(),
            })?;

        let results = eval_function_in_module(module, func, &[])
            .map_err(|e| CodegenError::Unsupported {
                backend: "jit-interp".into(),
                detail: format!("interpreter error: {:?}", e),
            })?;

        let mut output = String::new();
        for val in &results {
            output.push_str(&format!("{}\n", val));
        }

        Ok(JitResult { output, tier: JitTier::Interpreter })
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JIT module report
// ---------------------------------------------------------------------------

/// Generate a JIT compilation report for the module.
///
/// This is emitted when `--emit jit` is used. It shows:
/// - The IR hash (cache key).
/// - Which tier would be used.
/// - The function signatures available.
/// - The JIT execution result.
pub fn emit_jit(module: &IrModule) -> Result<String, CodegenError> {
    let mut compiler = JitCompiler::new();
    let result = compiler.compile_and_run(module)?;

    let mut out = String::new();
    use std::fmt::Write;
    writeln!(out, "; IRIS JIT — phase 52")?;
    writeln!(out, "; Module: {}", module.name)?;
    writeln!(out, "; IR hash: {:016x}", hash_module(module))?;
    writeln!(out, "; Execution tier: {}", result.tier)?;
    writeln!(out, "; clang available: {}", is_clang_available())?;
    writeln!(out, ";")?;
    writeln!(out, "; Functions available for JIT:")?;
    for func in module.functions() {
        let params: Vec<String> = func.params
            .iter()
            .map(|p| format!("{}: {}", p.name, p.ty))
            .collect();
        writeln!(out, ";   {} ({}) -> {}", func.name, params.join(", "), func.return_ty)?;
    }
    writeln!(out, ";")?;
    writeln!(out, "; Execution output:")?;
    for line in result.output.lines() {
        writeln!(out, ";   {}", line)?;
    }
    writeln!(out)?;
    out.push_str(&result.output);
    Ok(out)
}

// ---------------------------------------------------------------------------
// JIT IR description (for documentation/testing)
// ---------------------------------------------------------------------------

/// Emit a description of what the JIT would produce, for use in tests.
///
/// Unlike `emit_jit`, this does not actually execute code — it describes
/// the compilation plan.
pub fn emit_jit_plan(module: &IrModule) -> Result<String, CodegenError> {
    let mut out = String::new();
    use std::fmt::Write;

    writeln!(out, "; IRIS JIT compilation plan — phase 52")?;
    writeln!(out, "; Module: {}", module.name)?;
    writeln!(out, "; IR hash: {:016x}", hash_module(module))?;
    writeln!(out)?;

    let tier_str = if is_clang_available() {
        "native (clang subprocess)"
    } else {
        "interpreter (fallback)"
    };
    writeln!(out, "; Preferred tier: {}", tier_str)?;
    writeln!(out)?;

    writeln!(out, "; JIT pipeline:")?;
    writeln!(out, ";   1. IRIS IR → LLVM IR (emit_llvm_ir)")?;
    writeln!(out, ";   2. LLVM IR → machine code (clang -O2)")?;
    writeln!(out, ";   3. Load shared library (dlopen)")?;
    writeln!(out, ";   4. dlsym(entry_fn) → call with zero args")?;
    writeln!(out, ";   5. Capture stdout → return as string")?;
    writeln!(out)?;

    writeln!(out, "; Cache key:")?;
    writeln!(out, ";   module_name = {}", module.name)?;
    writeln!(out, ";   ir_hash     = {:016x}", hash_module(module))?;
    writeln!(out)?;

    writeln!(out, "; Functions compiled:")?;
    for func in module.functions() {
        if func.params.is_empty() {
            writeln!(out, ";   [ENTRY] {} () -> {}", func.name, func.return_ty)?;
        } else {
            let params: Vec<String> = func.params
                .iter()
                .map(|p| format!("{}: {}", p.name, p.ty))
                .collect();
            writeln!(out, ";   {} ({}) -> {}", func.name, params.join(", "), func.return_ty)?;
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true if `clang` is available in PATH.
fn is_clang_available() -> bool {
    std::process::Command::new("clang")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compute a stable hash of the module's IR text for cache invalidation.
fn hash_module(module: &IrModule) -> u64 {
    use crate::codegen::printer::emit_ir_text;
    let ir = emit_ir_text(module).unwrap_or_default();
    hash_str(&ir)
}

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}
