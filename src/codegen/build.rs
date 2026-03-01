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
/// (updated: added time/OS, struct/tuple/closure fallback helpers)
pub const RUNTIME_H_SRC: &str = include_str!("../runtime/iris_runtime.h");

/// The C runtime implementation, embedded at compile time.
/// (updated: added iris_now_ms, iris_sleep_ms, iris_make_struct, iris_get_field,
///  iris_make_tuple, iris_get_element, iris_make_closure, etc.)
pub const RUNTIME_C_SRC: &str = include_str!("../runtime/iris_runtime.c");

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compile an `IrModule` to a native executable.
///
/// `output_path` is the desired path for the final binary (e.g. `"./a.out"`).
///
/// Returns the `PathBuf` of the output binary on success, or a `CodegenError`
/// if no compiler can be found or any compilation/link step fails.
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

    // Locate compiler tools.
    // clang — compiles LLVM IR (.ll) to object files.
    // cc    — compiles C and links; prefers MSYS2 ucrt64 gcc (has pthreads).
    let clang = find_clang();
    let cc    = find_c_compiler();
    let msys2_inc = msys2_ucrt64_include();
    let msys2_lib = msys2_ucrt64_lib();

    // On Windows, MSYS2 GCC sub-tools (cc1.exe, collect2.exe, ld.exe) depend
    // on DLLs located in the MSYS2 bin directory.  If that directory is not on
    // PATH the sub-tools silently exit with code 1.  We prepend it here so
    // every Command that invokes GCC inherits the correct PATH.
    let msys2_bin = msys2_ucrt64_bin();
    let extended_path: Option<std::ffi::OsString> = msys2_bin.as_ref().map(|bin| {
        let cur = std::env::var_os("PATH").unwrap_or_default();
        let mut new = std::ffi::OsString::from(bin);
        new.push(";");
        new.push(&cur);
        new
    });

    // 5a. Compile iris_runtime.c → iris_runtime.o using the C compiler.
    let rt_obj = tmp_dir.join("iris_runtime.o");
    let mut compile_cmd = Command::new(&cc);
    compile_cmd.args([
        "-O2", "-c",
        c_path.to_str().unwrap(),
        "-o", rt_obj.to_str().unwrap(),
        "-I", tmp_dir.to_str().unwrap(),
    ]);
    if let Some(ref inc) = msys2_inc {
        compile_cmd.arg("-I").arg(inc);
    }
    if let Some(ref p) = extended_path {
        compile_cmd.env("PATH", p);
    }
    let c_output = compile_cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("'{}' not found: {}", cc, e),
    })?;
    if !c_output.status.success() {
        let stderr = String::from_utf8_lossy(&c_output.stderr);
        let stdout = String::from_utf8_lossy(&c_output.stdout);
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!(
                "'{}' failed to compile iris_runtime.c (exit: {:?})\nstderr: {}\nstdout: {}",
                cc, c_output.status.code(), stderr, stdout
            ),
        });
    }

    // 5b. Compile LLVM IR → module.o using clang (only clang understands .ll).
    let mod_obj = tmp_dir.join("module.o");
    let mut ir_cmd = Command::new(&clang);
    ir_cmd.args([
        "-O2", "-c",
        ll_path.to_str().unwrap(),
        "-o", mod_obj.to_str().unwrap(),
    ]);
    // On Windows, target the same ABI as MSYS2 ucrt64 (MinGW).
    if cfg!(target_os = "windows") || std::path::Path::new("/c/msys64").exists() {
        ir_cmd.args(["-target", "x86_64-w64-windows-gnu"]);
    }
    let ir_status = ir_cmd.status().map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("'{}' not found: {}", clang, e),
    })?;
    if !ir_status.success() {
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("'{}' failed to compile LLVM IR (exit: {:?})", clang, ir_status.code()),
        });
    }

    // 6. Link module.o + iris_runtime.o → native binary using the C compiler.
    let mut link_cmd = Command::new(&cc);
    link_cmd.args([
        "-O2",
        mod_obj.to_str().unwrap(),
        rt_obj.to_str().unwrap(),
        "-o", output_path.to_str().unwrap(),
        "-lm", "-lpthread",
    ]);
    if let Some(ref lib) = msys2_lib {
        link_cmd.arg(format!("-L{}", lib));
    }
    if let Some(ref p) = extended_path {
        link_cmd.env("PATH", p);
    }
    let link_output = link_cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("'{}' link step could not start: {}", cc, e),
    })?;
    if !link_output.status.success() {
        let stderr = String::from_utf8_lossy(&link_output.stderr);
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("'{}' failed to link binary (exit: {:?})\n{}", cc, link_output.status.code(), stderr),
        });
    }

    Ok(output_path.to_path_buf())
}

/// Find clang — required for compiling LLVM IR (.ll files).
/// Checks Windows-native paths first, then MSYS2 paths, then PATH.
fn find_clang() -> String {
    let candidates = [
        // Windows native paths (work from CMD, PowerShell, VSCode terminal)
        r"C:\Program Files\LLVM\bin\clang.exe",
        r"C:\Program Files (x86)\LLVM\bin\clang.exe",
        // MSYS2-style paths (work from MSYS2/MINGW shells)
        "/c/Program Files/LLVM/bin/clang.exe",
        "/usr/bin/clang",
    ];
    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return p.to_string();
        }
    }
    // Fall back to PATH lookup.
    "clang".to_owned()
}

/// Find the best C compiler — prefers MSYS2 ucrt64 gcc (has MinGW pthreads).
fn find_c_compiler() -> String {
    let candidates = [
        // Windows native paths
        r"C:\msys64\ucrt64\bin\gcc.exe",
        r"C:\msys64\mingw64\bin\gcc.exe",
        // MSYS2-style paths
        "/c/msys64/ucrt64/bin/gcc.exe",
        "/c/msys64/mingw64/bin/gcc.exe",
    ];
    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return p.to_string();
        }
    }
    // Fall back to PATH lookup.
    for candidate in &["gcc", "clang", "cc"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return candidate.to_string();
        }
    }
    "clang".to_owned()
}

/// Return the MSYS2 ucrt64 include path if it exists (checks both Windows and MSYS2 forms).
fn msys2_ucrt64_include() -> Option<String> {
    let candidates = [
        r"C:\msys64\ucrt64\include",
        "/c/msys64/ucrt64/include",
    ];
    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    None
}

/// Return the MSYS2 ucrt64 bin path.  GCC sub-tools (cc1, collect2, ld) need
/// this on PATH to find their dependent DLLs.
fn msys2_ucrt64_bin() -> Option<String> {
    let candidates = [
        r"C:\msys64\ucrt64\bin",
        "/c/msys64/ucrt64/bin",
    ];
    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    None
}

/// Return the MSYS2 ucrt64 lib path if it exists (checks both Windows and MSYS2 forms).
fn msys2_ucrt64_lib() -> Option<String> {
    let candidates = [
        r"C:\msys64\ucrt64\lib",
        "/c/msys64/ucrt64/lib",
    ];
    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    None
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
