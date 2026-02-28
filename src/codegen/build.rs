//! Native binary build pipeline for IRIS.
//!
//! Phase 54 — takes an `IrModule`, emits LLVM IR text, writes the embedded C
//! runtime to a temp dir, and invokes `clang` to produce a native executable.
//!
//! Build steps
//! -----------
//! 1. Emit LLVM IR from the module via `emit_llvm_ir`.
//! 2. Write `module.ll` to `$TMPDIR/iris_build_<PID>/`.
//! 3. Write the embedded `iris_runtime.h` + `iris_runtime.c` to the same dir.
//! 4. `clang -O2 -c iris_runtime.c -o iris_runtime.o -lpthread`
//! 5. `clang module.ll iris_runtime.o -o <output> -lm -lpthread`
//! 6. Return the path to the output binary.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::CodegenError;
use crate::ir::module::IrModule;

// ---------------------------------------------------------------------------
// Embedded runtime sources (compiled into the IRIS Rust binary itself)
// ---------------------------------------------------------------------------

/// The C runtime header, embedded at compile time.
pub const RUNTIME_H_SRC: &str = include_str!("../runtime/iris_runtime.h");

/// The C runtime implementation, embedded at compile time.
pub const RUNTIME_C_SRC: &str = include_str!("../runtime/iris_runtime.c");

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compile an `IrModule` to a native executable via clang.
///
/// `output_path` is the desired path for the final binary (e.g. `"./a.out"`).
///
/// Returns the `PathBuf` of the output binary on success, or a `CodegenError`
/// if clang cannot be found or any compilation/link step fails.
/// Requires at least one zero-argument function (preferably named `main`) as the entry point.
pub fn build_binary(module: &IrModule, output_path: &Path) -> Result<PathBuf, CodegenError> {
    use crate::codegen::llvm_ir::emit_llvm_ir_for_binary;

    let has_entry = module
        .functions()
        .iter()
        .any(|f| f.name == "main" || f.params.is_empty());
    if !has_entry {
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: "no entry point (define main() or a zero-argument function) for native binary".into(),
        });
    }

    // 1. Emit LLVM IR (with main wrapper for binary).
    let llvm_ir = emit_llvm_ir_for_binary(module)?;

    // 2. Set up a per-process temp directory so parallel builds don't collide.
    let tmp_dir = std::env::temp_dir()
        .join(format!("iris_build_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to create temp dir '{}': {}", tmp_dir.display(), e),
    })?;

    // 3. Write LLVM IR.
    let ll_path = tmp_dir.join("module.ll");
    std::fs::write(&ll_path, &llvm_ir).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write LLVM IR to '{}': {}", ll_path.display(), e),
    })?;

    // 4. Write embedded runtime sources.
    let h_path = tmp_dir.join("iris_runtime.h");
    let c_path = tmp_dir.join("iris_runtime.c");
    std::fs::write(&h_path, RUNTIME_H_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write runtime header: {}", e),
    })?;
    std::fs::write(&c_path, RUNTIME_C_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write runtime C source: {}", e),
    })?;

    // 5. Compile runtime C → object file.
    let rt_obj = tmp_dir.join("iris_runtime.o");
    let c_status = Command::new("clang")
        .args([
            "-O2",
            "-c",
            c_path.to_str().unwrap(),
            "-o",
            rt_obj.to_str().unwrap(),
            "-I",
            tmp_dir.to_str().unwrap(),
            "-lpthread",
        ])
        .status()
        .map_err(|e| CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("clang not found or could not start: {}", e),
        })?;
    if !c_status.success() {
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("clang failed to compile iris_runtime.c (exit: {:?})", c_status.code()),
        });
    }

    // 6. Link LLVM IR + runtime object → native binary.
    let link_status = Command::new("clang")
        .args([
            "-O2",
            ll_path.to_str().unwrap(),
            rt_obj.to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
            "-lm",
            "-lpthread",
        ])
        .status()
        .map_err(|e| CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("clang link step could not start: {}", e),
        })?;
    if !link_status.success() {
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("clang failed to link binary (exit: {:?})", link_status.code()),
        });
    }

    Ok(output_path.to_path_buf())
}

/// Emit LLVM IR text suitable for native binary compilation.
///
/// This is identical to `emit_llvm_ir` but provides a clear name for the
/// binary code-generation path.
pub fn emit_binary_ir(module: &IrModule) -> Result<String, CodegenError> {
    crate::codegen::llvm_ir::emit_llvm_ir(module)
}

/// Returns the embedded C runtime source as a static string.
///
/// Useful for writing the runtime to disk in build scripts or tests.
pub fn runtime_c_source() -> &'static str {
    RUNTIME_C_SRC
}

/// Returns the embedded C runtime header as a static string.
pub fn runtime_h_source() -> &'static str {
    RUNTIME_H_SRC
}
