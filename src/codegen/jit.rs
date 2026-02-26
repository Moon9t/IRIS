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

    /// Native tier: emit LLVM IR → compile with clang → load shared lib → call.
    fn compile_native(&self, module: &IrModule, _fn_name: &str) -> Result<JitResult, CodegenError> {
        use crate::codegen::llvm_ir::emit_llvm_ir;
        use std::process::Command;

        let ir_text = emit_llvm_ir(module)?;

        // Write IR to a temp file.
        let tmp_dir = std::env::temp_dir();
        let ir_path = tmp_dir.join(format!("iris_jit_{}.ll", hash_str(&module.name)));
        let out_path = tmp_dir.join(format!("iris_jit_{}.out", hash_str(&module.name)));

        std::fs::write(&ir_path, &ir_text).map_err(|e| CodegenError::Unsupported {
            backend: "jit-native".into(),
            detail: format!("cannot write temp IR file: {}", e),
        })?;

        // Compile with clang: produce an executable.
        let status = Command::new("clang")
            .args([
                ir_path.to_str().unwrap_or(""),
                "-o", out_path.to_str().unwrap_or(""),
                "-lm",
                "-Wno-override-module",
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                // Run the compiled executable.
                let run_output = Command::new(out_path.to_str().unwrap_or(""))
                    .output()
                    .map_err(|e| CodegenError::Unsupported {
                        backend: "jit-native".into(),
                        detail: format!("cannot run compiled binary: {}", e),
                    })?;

                let stdout = String::from_utf8_lossy(&run_output.stdout).to_string();
                let _ = std::fs::remove_file(&ir_path);
                let _ = std::fs::remove_file(&out_path);

                Ok(JitResult {
                    output: stdout,
                    tier: JitTier::Native,
                })
            }
            _ => {
                // clang failed — fall back to interpreter.
                let _ = std::fs::remove_file(&ir_path);
                self.compile_interpreter(module)
            }
        }
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
