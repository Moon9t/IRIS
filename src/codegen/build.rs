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
    build_binary_with_target(module, output_path, None)
}

/// Like `build_binary` but overrides the LLVM/clang target triple.
pub fn build_binary_with_target(
    module: &IrModule,
    output_path: &Path,
    target: Option<&str>,
) -> Result<PathBuf, CodegenError> {
    use crate::codegen::llvm_ir::emit_llvm_ir_for_binary;
    if target.is_some() {
        build_binary_impl(
            crate::codegen::llvm_ir::emit_llvm_ir_for_binary_with_target(module, target)?,
            output_path,
            target,
        )
    } else {
        build_binary_impl(emit_llvm_ir_for_binary(module)?, output_path, None)
    }
}

/// Like `build_binary` but uses the eval wrapper: the entry function's return
/// value is printed to stdout instead of being used as the process exit code.
/// Used by `EmitKind::Eval` so that tests get the same output as the interpreter.
pub fn build_binary_for_eval(
    module: &IrModule,
    output_path: &Path,
) -> Result<PathBuf, CodegenError> {
    build_binary_for_eval_with_target(module, output_path, None)
}

/// Like `build_binary_for_eval` but overrides the LLVM/clang target triple.
pub fn build_binary_for_eval_with_target(
    module: &IrModule,
    output_path: &Path,
    target: Option<&str>,
) -> Result<PathBuf, CodegenError> {
    use crate::codegen::llvm_ir::emit_llvm_ir_for_eval;
    if target.is_some() {
        build_binary_impl(
            crate::codegen::llvm_ir::emit_llvm_ir_for_eval_with_target(module, target)?,
            output_path,
            target,
        )
    } else {
        build_binary_impl(emit_llvm_ir_for_eval(module)?, output_path, None)
    }
}

/// Build and execute a temporary native binary using the eval wrapper.
///
/// The entry function's return value is printed to stdout, matching the
/// observable behavior of `EmitKind::Eval`.
pub fn execute_binary_for_eval(module: &IrModule) -> Result<String, CodegenError> {
    execute_binary_for_eval_with_target(module, None)
}

/// Like `execute_binary_for_eval` but overrides the LLVM/clang target triple.
pub fn execute_binary_for_eval_with_target(
    module: &IrModule,
    target: Option<&str>,
) -> Result<String, CodegenError> {
    let output = run_binary_for_eval_entry_capture(module, None, target)?;
    let stdout = String::from_utf8_lossy(&output.stdout)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CodegenError::Unsupported {
            backend: "native".into(),
            detail: format!("runtime error (exit {}): {}", output.status, stderr.trim()),
        });
    }
    Ok(stdout)
}

pub(crate) fn run_binary_for_eval_entry_capture(
    module: &IrModule,
    entry_name: Option<&str>,
    target: Option<&str>,
) -> Result<std::process::Output, CodegenError> {
    let bin_path = if let Some(name) = entry_name {
        build_binary_from_llvm_ir(
            crate::codegen::llvm_ir::emit_llvm_ir_for_named_eval_with_target(
                module,
                Some(name),
                target,
            )?,
            &temp_eval_binary_path(),
            target,
        )?
    } else {
        build_binary_for_eval_with_target(module, &temp_eval_binary_path(), target)?
    };
    let run_path = std::fs::canonicalize(&bin_path).unwrap_or(bin_path.clone());
    let output = Command::new(&run_path).output().map_err(CodegenError::Io)?;
    let _ = std::fs::remove_file(&run_path);
    Ok(output)
}

pub(crate) fn run_native_test_capture(
    module: &IrModule,
    entry_name: &str,
    target: Option<&str>,
) -> Result<std::process::Output, CodegenError> {
    let bin_path = build_binary_from_llvm_ir(
        crate::codegen::llvm_ir::emit_llvm_ir_for_test_entry_with_target(
            module, entry_name, target,
        )?,
        &temp_eval_binary_path(),
        target,
    )?;
    let run_path = std::fs::canonicalize(&bin_path).unwrap_or(bin_path.clone());
    let output = Command::new(&run_path).output().map_err(CodegenError::Io)?;
    let _ = std::fs::remove_file(&run_path);
    Ok(output)
}

fn temp_eval_binary_path() -> PathBuf {
    let pid = std::process::id();
    let tid = format!("{:?}", std::thread::current().id())
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    std::env::temp_dir().join(format!(
        "iris_eval_{}_{}_{}{}",
        pid,
        tid,
        nanos,
        std::env::consts::EXE_SUFFIX
    ))
}

fn build_binary_from_llvm_ir(
    llvm_ir: String,
    output_path: &Path,
    target: Option<&str>,
) -> Result<PathBuf, CodegenError> {
    build_binary_impl(llvm_ir, output_path, target)
}

fn build_binary_impl(
    llvm_ir: String,
    output_path: &Path,
    target: Option<&str>,
) -> Result<PathBuf, CodegenError> {
    if !llvm_ir.contains("define i32 @main(") {
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: "no entry point (define main() or a zero-argument function) for native binary"
                .into(),
        });
    }

    // 1. LLVM IR already emitted.

    // 2. Set up a per-call temp directory so parallel builds don't collide.
    // Derive from output_path's stem (which already contains pid+tid+nanos for eval builds).
    let build_id = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| format!("{}_bld", s))
        .unwrap_or_else(|| format!("iris_build_{}", std::process::id()));
    let tmp_dir = std::env::temp_dir().join(build_id);
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

    // Helper: convert a PathBuf to &str, returning a descriptive error on non-UTF8 paths.
    fn path_str(p: &std::path::Path) -> Result<&str, CodegenError> {
        p.to_str().ok_or_else(|| CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("path contains non-UTF8 characters: {}", p.display()),
        })
    }

    let resolved_target = resolve_target_triple(target);
    let target_args = ["-target".to_owned(), resolved_target.clone()];

    // 5a. Compile iris_runtime.c → iris_runtime.o using clang.
    let rt_obj = tmp_dir.join("iris_runtime.o");
    let mut compile_cmd = Command::new(&clang);
    compile_cmd.args(&target_args);
    compile_cmd.args([
        "-O2",
        "-c",
        path_str(&c_path)?,
        "-o",
        path_str(&rt_obj)?,
        "-I",
        path_str(&tmp_dir)?,
        "-Wno-pragma-pack",
    ]);
    if resolved_target.contains("windows") {
        if let Some(ref inc) = msys2_inc {
            compile_cmd.arg("-I").arg(inc);
        }
    }
    let c_output = compile_cmd
        .output()
        .map_err(|e| CodegenError::Unsupported {
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
                clang,
                c_output.status.code(),
                stderr,
                stdout
            ),
        });
    }

    // 5b. Compile LLVM IR → module.o using clang (only clang understands .ll).
    // Use -O1 for user IR to avoid clang 17 optimizer crashes with complex IR patterns.
    let mod_obj = tmp_dir.join("module.o");
    let mut ir_cmd = Command::new(&clang);
    ir_cmd.args(&target_args);
    ir_cmd.args([
        "-O1",
        "-c",
        path_str(&ll_path)?,
        "-o",
        path_str(&mod_obj)?,
        "-Wno-override-module",
    ]);
    let ir_status = ir_cmd.status().map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("'{}' not found: {}", clang, e),
    })?;
    if !ir_status.success() {
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!(
                "'{}' failed to compile LLVM IR (exit: {:?})",
                clang,
                ir_status.code()
            ),
        });
    }

    // 6. Link module.o + iris_runtime.o → native binary using clang + lld.
    let mut link_cmd = Command::new(&clang);
    link_cmd.args(&target_args);
    link_cmd.args([
        "-fuse-ld=lld",
        "-O2",
        path_str(&mod_obj)?,
        path_str(&rt_obj)?,
        "-o",
        path_str(output_path)?,
        "-lm",
        "-lpthread",
    ]);
    // Windows: link WinSock2 for TCP/HTTP builtins
    if resolved_target.contains("windows") {
        link_cmd.arg("-lws2_32");
    }
    if resolved_target.contains("windows") {
        if let Some(ref lib) = msys2_lib {
            link_cmd.arg(format!("-L{}", lib));
        }
        if let Some(ref lib) = gcc_lib {
            link_cmd.arg(format!("-L{}", lib));
        }
    }
    if !resolved_target.contains("windows") {
        // Non-Windows targets keep relying on the target toolchain's standard sysroot.
    }
    let link_output = link_cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("'{}' link step could not start: {}", clang, e),
    })?;
    if !link_output.status.success() {
        let stderr = String::from_utf8_lossy(&link_output.stderr);
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!(
                "'{}' failed to link binary (exit: {:?})\n{}",
                clang,
                link_output.status.code(),
                stderr
            ),
        });
    }

    stage_sqlite_dll_next_to(output_path);
    Ok(output_path.to_path_buf())
}

fn resolve_target_triple(target: Option<&str>) -> String {
    match target {
        Some(t) => crate::codegen::llvm_ir::target_preset_to_triple(t).unwrap_or(t),
        None => crate::codegen::llvm_ir::native_target_triple(),
    }
    .to_owned()
}

fn stage_sqlite_dll_next_to(output_path: &Path) {
    use std::path::{Path, PathBuf};

    let candidate_dirs = [
        std::env::var("SQLITE3_DIR").ok(),
        Some(r"C:\Program Files\Cheat Engine\win64".to_owned()),
        Some(r"C:\Program Files\Common Files\Apple\Mobile Device Support".to_owned()),
        Some(r"C:\Program Files (x86)\Common Files\Apple\Mobile Device Support".to_owned()),
    ];

    let mut source_path: Option<PathBuf> = None;
    for dir in candidate_dirs.into_iter().flatten() {
        for file_name in ["sqlite3.dll", "SQLite3.dll"] {
            let path = Path::new(&dir).join(file_name);
            if path.exists() {
                source_path = Some(path);
                break;
            }
        }
        if source_path.is_some() {
            break;
        }
    }

    let Some(source_path) = source_path else {
        return;
    };

    if let Some(parent) = output_path.parent() {
        let target = parent.join(source_path.file_name().unwrap_or_default());
        let _ = std::fs::copy(&source_path, target);
    }
}

/// Find clang — required for compiling LLVM IR, C code, and linking.
/// Search order: IRIS_CLANG env var, next to iris binary (bundled),
/// Inno Setup install dir, system LLVM, PATH.
pub(crate) fn find_clang() -> String {
    // Env-var override takes precedence over all automatic detection.
    if let Ok(v) = std::env::var("IRIS_CLANG") {
        if !v.is_empty() {
            return v;
        }
    }

    let mut candidates: Vec<String> = Vec::new();

    // 1. Relative to the running executable  (…/toolchain/llvm/bin/clang[.exe])
    //    Works for both bundled release archives and local dev installs.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            #[cfg(target_os = "windows")]
            {
                candidates.push(format!(r"{}\toolchain\llvm\bin\clang.exe", dir.display()));
            }
            #[cfg(not(target_os = "windows"))]
            {
                candidates.push(format!("{}/toolchain/llvm/bin/clang", dir.display()));
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // 2. Inno Setup default install dir  ({LOCALAPPDATA}\Programs\IRIS)
        if let Ok(lad) = std::env::var("LOCALAPPDATA") {
            candidates.push(format!(
                r"{}\Programs\IRIS\toolchain\llvm\bin\clang.exe",
                lad
            ));
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
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: package-installed toolchain, Homebrew LLVM, Xcode CLT
        candidates.push("/usr/local/share/iris/toolchain/llvm/bin/clang".into());
        candidates.push("/opt/homebrew/opt/llvm/bin/clang".into());
        candidates.push("/usr/local/opt/llvm/bin/clang".into());
        candidates.push("/usr/bin/clang".into());
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(format!("{}/.iris/toolchain/llvm/bin/clang", home));
            candidates.push(format!("{}/.iris/llvm/bin/clang", home));
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Linux: package-installed toolchain, then common distribution paths
        candidates.push("/usr/share/iris/toolchain/llvm/bin/clang".into());
        candidates.push("/usr/bin/clang".into());
        candidates.push("/usr/lib/llvm-18/bin/clang".into());
        candidates.push("/usr/lib/llvm-17/bin/clang".into());
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(format!("{}/.iris/toolchain/llvm/bin/clang", home));
            candidates.push(format!("{}/.iris/llvm/bin/clang", home));
        }
    }

    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return p.clone();
        }
    }
    // Fall back to PATH lookup.
    "clang".to_owned()
}

/// Return the MinGW ucrt64 include path if it exists.
/// Windows-only: needed for cross-compiling to the windows-gnu target.
/// On Linux/macOS, system headers are used via clang's built-in paths.
pub(crate) fn msys2_ucrt64_include() -> Option<String> {
    #[cfg(not(target_os = "windows"))]
    {
        None
    }

    #[cfg(target_os = "windows")]
    {
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
            candidates.push(format!(r"{}\.iris\toolchain\ucrt64\include", home));
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
}

/// Return the MinGW ucrt64 lib path if it exists (Windows-only).
pub(crate) fn msys2_ucrt64_lib() -> Option<String> {
    #[cfg(not(target_os = "windows"))]
    {
        None
    }

    #[cfg(target_os = "windows")]
    {
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
            candidates.push(format!(r"{}\.iris\toolchain\ucrt64\lib", home));
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
}

/// Return the GCC internal lib path (contains CRT start files like crtbegin.o,
/// libgcc.a) inside the MinGW ucrt64 tree (Windows-only).
pub(crate) fn msys2_gcc_lib() -> Option<String> {
    #[cfg(not(target_os = "windows"))]
    {
        None
    }

    #[cfg(target_os = "windows")]
    {
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
            base_dirs.push(format!(r"{}\.iris\toolchain\ucrt64\lib\gcc", home));
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
