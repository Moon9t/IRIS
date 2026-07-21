//! JIT compilation backend for IRIS.
//!
//! Phase 52: Compiles IRIS IR to native machine code at runtime.
//!
//! Architecture
//! ─────────────
//! The JIT is native-only:
//!
//! 1. **Native tier**: emits LLVM IR text, compiles it with `clang`, and
//!    executes the resulting native binary.
//!
//! 2. **Cached tier**: once a function has been JIT-compiled natively, results
//!    are cached in a `JitCache` for reuse within the same process.
//!
//! Usage
//! ──────
//! ```text
//! iris --emit jit program.iris
//! ```
//! This evaluates the first zero-argument function and prints the result,
//! using the LLVM/native pipeline.
//!
//! JIT cache key: (module_name, function_name, ir_hash)

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::error::CodegenError;
use crate::ir::module::IrModule;

use crate::codegen::build::find_clang;
#[cfg(target_os = "windows")]
use crate::codegen::build::{msys2_gcc_lib, msys2_ucrt64_include, msys2_ucrt64_lib};

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
}

impl std::fmt::Display for JitTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitTier::Native => f.write_str("native"),
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
        Self {
            cache: HashMap::new(),
        }
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

        if !is_native_jit_available() {
            return Err(CodegenError::Unsupported {
                backend: "jit".into(),
                detail: "native JIT requires a working clang/LLVM toolchain; install clang or set IRIS_CLANG"
                    .into(),
            });
        }

        let result = self.compile_native(module)?;

        self.cache.insert(key, result.clone());
        Ok(result)
    }

    /// Native tier: compile and execute using the same LLVM pipeline as
    /// `EmitKind::Eval`, capturing stdout.
    fn compile_native(&self, module: &IrModule) -> Result<JitResult, CodegenError> {
        let stdout = crate::codegen::execute_binary_for_eval(module)?;
        Ok(JitResult {
            output: stdout,
            tier: JitTier::Native,
        })
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
    writeln!(
        out,
        "; native toolchain available: {}",
        is_native_jit_available()
    )?;
    writeln!(out, ";")?;
    writeln!(out, "; Functions available for JIT:")?;
    for func in module.functions() {
        let params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, p.ty))
            .collect();
        writeln!(
            out,
            ";   {} ({}) -> {}",
            func.name,
            params.join(", "),
            func.return_ty
        )?;
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

    let tier_str = if is_native_jit_available() {
        "native (clang subprocess)"
    } else {
        "native unavailable"
    };
    writeln!(out, "; Preferred tier: {}", tier_str)?;
    writeln!(out)?;

    writeln!(out, "; JIT pipeline:")?;
    writeln!(out, ";   1. IRIS IR → LLVM IR (eval wrapper)")?;
    writeln!(out, ";   2. LLVM IR → native binary (clang)")?;
    writeln!(out, ";   3. Execute native binary")?;
    writeln!(out, ";   4. Capture stdout → return as string")?;
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
            let params: Vec<String> = func
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, p.ty))
                .collect();
            writeln!(
                out,
                ";   {} ({}) -> {}",
                func.name,
                params.join(", "),
                func.return_ty
            )?;
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true if `clang` is available in PATH.
fn is_native_jit_available() -> bool {
    // clang must be present.
    let clang_ok = std::process::Command::new(find_clang())
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !clang_ok {
        return false;
    }
    // On Windows targeting MinGW/gnu, the JIT needs MSYS2/ucrt64 toolchain
    // (headers + libraries + GCC CRT). For MSVC targets, clang auto-discovers
    // the MSVC toolchain, so MSYS2 is not required.
    #[cfg(target_os = "windows")]
    {
        let target = crate::codegen::llvm_ir::native_target_triple();
        if !target.contains("msvc") {
            if msys2_ucrt64_include().is_none()
                || msys2_ucrt64_lib().is_none()
                || msys2_gcc_lib().is_none()
            {
                return false;
            }
        }
    }
    true
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
