//! Native binary build pipeline for IRIS.
//!
//! Phase 54 — takes an `IrModule`, emits LLVM IR text, writes the embedded C
//! runtime to a temp dir, and invokes `clang` + `lld` to produce a native
//! executable.  **No GCC installation is required** — only LLVM/clang (with
//! the bundled `ld.lld`) and MinGW sysroot headers + libraries.
//!
//! Build steps
//! -----------
//! 1. Emit LLVM IR from the module via `emit_llvm_ir`.
//! 2. Write `module.ll` to `$TMPDIR/iris_build_<PID>/`.
//! 3. Write the embedded `iris_runtime.h` + `iris_runtime.c` to the same dir.
//! 4. `clang -target x86_64-w64-windows-gnu -O2 -c iris_runtime.c -o iris_runtime.o`
//! 5. `clang -target x86_64-w64-windows-gnu -O2 -c module.ll -o module.o`
//! 6. `clang -target x86_64-w64-windows-gnu -fuse-ld=lld module.o iris_runtime.o -o <output> -lm -lpthread`
//! 7. Return the path to the output binary.

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
    // clang — compiles LLVM IR (.ll) to object files AND compiles the C
    //         runtime AND links the final binary (with -fuse-ld=lld).
    //         No GCC installation is required.
    let clang = find_clang();
    let msys2_inc = msys2_ucrt64_include();
    let msys2_lib = msys2_ucrt64_lib();
    let gcc_lib = msys2_gcc_lib();

    // Common target triple for all clang invocations on Windows.
    let target_args: &[&str] = if cfg!(target_os = "windows") {
        &["-target", "x86_64-w64-windows-gnu"]
    } else {
        &[]
    };

    // 5a. Compile iris_runtime.c → iris_runtime.o using clang.
    let rt_obj = tmp_dir.join("iris_runtime.o");
    let mut compile_cmd = Command::new(&clang);
    compile_cmd.args(target_args);
    compile_cmd.args([
        "-O2", "-c",
        c_path.to_str().unwrap(),
        "-o", rt_obj.to_str().unwrap(),
        "-I", tmp_dir.to_str().unwrap(),
        "-Wno-pragma-pack",
    ]);
    if let Some(ref inc) = msys2_inc {
        compile_cmd.arg("-I").arg(inc);
    }
    let c_output = compile_cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("'{}' not found: {}", clang, e),
    })?;
    if !c_output.status.success() {
        let stderr = String::from_utf8_lossy(&c_output.stderr);
        let stdout = String::from_utf8_lossy(&c_output.stdout);
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!(
                "'{}' failed to compile iris_runtime.c (exit: {:?})\nstderr: {}\nstdout: {}",
                clang, c_output.status.code(), stderr, stdout
            ),
        });
    }

    // 5b. Compile LLVM IR → module.o using clang (only clang understands .ll).
    let mod_obj = tmp_dir.join("module.o");
    let mut ir_cmd = Command::new(&clang);
    ir_cmd.args(target_args);
    ir_cmd.args([
        "-O2", "-c",
        ll_path.to_str().unwrap(),
        "-o", mod_obj.to_str().unwrap(),
        "-Wno-override-module",
    ]);
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

    // 6. Link module.o + iris_runtime.o → native binary using clang + lld.
    let mut link_cmd = Command::new(&clang);
    link_cmd.args(target_args);
    link_cmd.args([
        "-fuse-ld=lld",
        "-O2",
        mod_obj.to_str().unwrap(),
        rt_obj.to_str().unwrap(),
        "-o", output_path.to_str().unwrap(),
        "-lm", "-lpthread",
    ]);
    if let Some(ref lib) = msys2_lib {
        link_cmd.arg(format!("-L{}", lib));
    }
    if let Some(ref lib) = gcc_lib {
        link_cmd.arg(format!("-L{}", lib));
    }
    let link_output = link_cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("'{}' link step could not start: {}", clang, e),
    })?;
    if !link_output.status.success() {
        let stderr = String::from_utf8_lossy(&link_output.stderr);
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("'{}' failed to link binary (exit: {:?})\n{}", clang, link_output.status.code(), stderr),
        });
    }

    Ok(output_path.to_path_buf())
}

/// Find clang — required for compiling LLVM IR, C code, and linking.
/// Search order: next to iris.exe, Inno Setup install dir, system LLVM, PATH.
pub(crate) fn find_clang() -> String {
    let mut candidates: Vec<String> = Vec::new();

    // 1. Relative to the running executable  (…/IRIS/toolchain/llvm/bin/clang.exe)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(format!(r"{}\toolchain\llvm\bin\clang.exe", dir.display()));
        }
    }

    // 2. Inno Setup default install dir  ({LOCALAPPDATA}\Programs\IRIS)
    if let Ok(lad) = std::env::var("LOCALAPPDATA") {
        candidates.push(format!(r"{}\Programs\IRIS\toolchain\llvm\bin\clang.exe", lad));
    }

    // 3. System-wide LLVM installs
    candidates.push(r"C:\Program Files\LLVM\bin\clang.exe".into());
    candidates.push(r"C:\Program Files (x86)\LLVM\bin\clang.exe".into());

    // 4. Legacy user-local fallback
    if let Ok(home) = std::env::var("USERPROFILE") {
        candidates.push(format!(r"{}\.iris\llvm\bin\clang.exe", home));
    }

    // 5. MSYS2-style paths (from MSYS2/MINGW shells)
    candidates.push("/c/Program Files/LLVM/bin/clang.exe".into());
    candidates.push("/usr/bin/clang".into());

    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return p.clone();
        }
    }
    // Fall back to PATH lookup.
    "clang".to_owned()
}

/// Return the MinGW ucrt64 include path if it exists.
/// Search order: next to iris.exe, Inno Setup dir, system MSYS2, legacy.
pub(crate) fn msys2_ucrt64_include() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(format!(r"{}\toolchain\ucrt64\include", dir.display()));
        }
    }
    if let Ok(lad) = std::env::var("LOCALAPPDATA") {
        candidates.push(format!(r"{}\Programs\IRIS\toolchain\ucrt64\include", lad));
    }
    candidates.push(r"C:\msys64\ucrt64\include".into());
    if let Ok(home) = std::env::var("USERPROFILE") {
        candidates.push(format!(r"{}\.iris\ucrt64\include", home));
    }
    candidates.push("/c/msys64/ucrt64/include".into());

    for p in &candidates {
        if std::path::Path::new(p.as_str()).exists() {
            return Some(p.clone());
        }
    }
    None
}

/// Return the MinGW ucrt64 lib path if it exists.
pub(crate) fn msys2_ucrt64_lib() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(format!(r"{}\toolchain\ucrt64\lib", dir.display()));
        }
    }
    if let Ok(lad) = std::env::var("LOCALAPPDATA") {
        candidates.push(format!(r"{}\Programs\IRIS\toolchain\ucrt64\lib", lad));
    }
    candidates.push(r"C:\msys64\ucrt64\lib".into());
    if let Ok(home) = std::env::var("USERPROFILE") {
        candidates.push(format!(r"{}\.iris\ucrt64\lib", home));
    }
    candidates.push("/c/msys64/ucrt64/lib".into());

    for p in &candidates {
        if std::path::Path::new(p.as_str()).exists() {
            return Some(p.clone());
        }
    }
    None
}

/// Return the GCC internal lib path (contains CRT start files like crtbegin.o,
/// libgcc.a) inside the MinGW ucrt64 tree.
pub(crate) fn msys2_gcc_lib() -> Option<String> {
    let triple = "x86_64-w64-mingw32";
    let versions = ["14.2.0", "14.1.0", "13.2.0", "13.1.0", "12.2.0"];

    let mut base_dirs: Vec<String> = Vec::new();

    // Next to the running executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            base_dirs.push(format!(r"{}\toolchain\ucrt64\lib\gcc", dir.display()));
        }
    }
    // Inno Setup default install location
    if let Ok(lad) = std::env::var("LOCALAPPDATA") {
        base_dirs.push(format!(r"{}\Programs\IRIS\toolchain\ucrt64\lib\gcc", lad));
    }
    // System MSYS2
    base_dirs.push(r"C:\msys64\ucrt64\lib\gcc".into());
    // Legacy user-local
    if let Ok(home) = std::env::var("USERPROFILE") {
        base_dirs.push(format!(r"{}\.iris\ucrt64\lib\gcc", home));
    }
    base_dirs.push("/c/msys64/ucrt64/lib/gcc".into());

    for base in &base_dirs {
        for ver in &versions {
            let p = format!("{}\\{}\\{}", base, triple, ver);
            if std::path::Path::new(&p).exists() {
                return Some(p);
            }
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
