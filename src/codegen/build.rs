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

/// Helper: convert a Path to &str, returning a descriptive error on non-UTF8 paths.
fn path_str(p: &Path) -> Result<&str, CodegenError> {
    p.to_str().ok_or_else(|| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("path contains non-UTF8 characters: {}", p.display()),
    })
}

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

/// Native ML backend shims used by generated IRIS binaries.
pub const ONNX_SHIM_H_SRC: &str = include_str!("../runtime/onnx_shim.h");
pub const ONNX_SHIM_C_SRC: &str = include_str!("../runtime/onnx_shim.c");
pub const TF_SHIM_C_SRC: &str = include_str!("../runtime/tf_shim.c");
pub const PYTORCH_SHIM_CPP_SRC: &str = include_str!("../runtime/pytorch_shim.cpp");

/// Lazy runtime loader for the ML backends. Lets the shims compile with no ML
/// SDK installed — the shared libraries are resolved via dlopen on first use.
pub const ML_DYNLOAD_H_SRC: &str = include_str!("../runtime/iris_ml_dynload.h");

// Prebuilt runtime objects embedded by the top-level build script. Defines
// `PrebuiltRuntime`, `RUNTIME_SOURCES_HASH` and `PREBUILT_RUNTIMES`. When a set
// is present for the target triple we skip the C compiler entirely.
include!(concat!(env!("OUT_DIR"), "/prebuilt_runtime.rs"));

/// Look up prebuilt runtime objects for `triple`, if this compiler embeds any.
///
/// The build script only emits entries whose recorded source hash matches the
/// current C sources, so anything returned here is known to be up to date.
fn find_prebuilt_runtime(triple: &str) -> Option<&'static PrebuiltRuntime> {
    PREBUILT_RUNTIMES.iter().find(|p| p.triple == triple)
}

/// Vendored ONNX Runtime C API header (MIT, Microsoft). Supplies the ABI layout
/// only; it creates no link-time dependency. See src/runtime/vendor/README.md.
pub const ONNXRUNTIME_C_API_H_SRC: &str =
    include_str!("../runtime/vendor/onnxruntime_c_api.h");

/// ML compute kernels header — convolution, pooling, losses, optimizers, etc.
pub const ML_KERNELS_H_SRC: &str = include_str!("../runtime/iris_ml_kernels.h");
/// ML compute kernels implementation.
pub const ML_KERNELS_C_SRC: &str = include_str!("../runtime/iris_ml_kernels.c");

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
    let link_libs: Vec<String> = module.extern_fns.iter()
        .filter_map(|e| e.link_lib.clone())
        .collect();
    if target.is_some() {
        build_binary_impl(
            crate::codegen::llvm_ir::emit_llvm_ir_for_binary_with_target(module, target)?,
            output_path,
            target,
            link_libs,
        )
    } else {
        build_binary_impl(emit_llvm_ir_for_binary(module)?, output_path, None, link_libs)
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
    let link_libs: Vec<String> = module.extern_fns.iter()
        .filter_map(|e| e.link_lib.clone())
        .collect();
    if target.is_some() {
        build_binary_impl(
            crate::codegen::llvm_ir::emit_llvm_ir_for_eval_with_target(module, target)?,
            output_path,
            target,
            link_libs,
        )
    } else {
        build_binary_impl(emit_llvm_ir_for_eval(module)?, output_path, None, link_libs)
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
    let link_libs: Vec<String> = module.extern_fns.iter()
        .filter_map(|e| e.link_lib.clone())
        .collect();
    let bin_path = if let Some(name) = entry_name {
        build_binary_impl(
            crate::codegen::llvm_ir::emit_llvm_ir_for_named_eval_with_target(
                module,
                Some(name),
                target,
            )?,
            &temp_eval_binary_path(),
            target,
            link_libs,
        )?
    } else {
        build_binary_for_eval_with_target(module, &temp_eval_binary_path(), target)?
    };
    let run_path = std::fs::canonicalize(&bin_path).unwrap_or(bin_path.clone());
    // Run the native binary with a timeout. Programs that use spawn/TCP may hang
    // indefinitely in the native runtime, so we fall back to the interpreter.
    //
    // The limit is configurable because it is wall-clock, not work: on a slow or
    // loaded machine a correct program can exceed a fixed budget and be reported
    // as a failure, which is a false negative rather than a real defect.
    let timeout_secs = native_timeout_secs();
    let output = run_with_timeout(&run_path, std::time::Duration::from_secs(timeout_secs))
        .map_err(|_| CodegenError::Unsupported {
            backend: "native".into(),
            detail: format!(
                "native binary timed out ({}s); raise IRIS_NATIVE_TIMEOUT or use the interpreter",
                timeout_secs
            ),
        })?;
    let _ = std::fs::remove_file(&run_path);
    Ok(output)
}

/// Wall-clock budget for a natively built program run under `--emit eval`.
///
/// Override with `IRIS_NATIVE_TIMEOUT` (seconds). Values that do not parse, or a
/// zero, fall back to the default rather than disabling the timeout — an
/// unbounded wait would hang the whole test suite on a single spawning program.
pub(crate) fn native_timeout_secs() -> u64 {
    const DEFAULT_NATIVE_TIMEOUT_SECS: u64 = 15;
    std::env::var("IRIS_NATIVE_TIMEOUT")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_NATIVE_TIMEOUT_SECS)
}

/// Whether to skip native compilation entirely and interpret instead.
///
/// Set `IRIS_FORCE_INTERP=1` to enable. `--emit eval` otherwise compiles and
/// links a native binary per program, which dominates test-suite runtime: the
/// suite pays a clang invocation per test even though the interpreter would
/// answer in milliseconds. This is the switch that makes the suite fast enough
/// to iterate on, and it also removes native-timeout flakiness from CI.
pub fn force_interpreter() -> bool {
    matches!(
        std::env::var("IRIS_FORCE_INTERP").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Run a child process and wait for it to finish, with a timeout.
fn run_with_timeout(
    bin: &std::path::Path,
    timeout: std::time::Duration,
) -> Result<std::process::Output, ()> {
    let mut child = std::process::Command::new(bin)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|_| ())?;
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Read remaining stdout/stderr in case the process buffered before exiting
                let output = child.wait_with_output().map_err(|_| ())?;
                return Ok(std::process::Output {
                    status,
                    stdout: output.stdout,
                    stderr: output.stderr,
                });
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(());
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(());
            }
        }
    }
}

pub(crate) fn run_native_test_capture(
    module: &IrModule,
    entry_name: &str,
    target: Option<&str>,
) -> Result<std::process::Output, CodegenError> {
    let link_libs: Vec<String> = module.extern_fns.iter()
        .filter_map(|e| e.link_lib.clone())
        .collect();
    let bin_path = build_binary_impl(
        crate::codegen::llvm_ir::emit_llvm_ir_for_test_entry_with_target(
            module, entry_name, target,
        )?,
        &temp_eval_binary_path(),
        target,
        link_libs,
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

fn _build_binary_from_llvm_ir(
    llvm_ir: String,
    output_path: &Path,
    target: Option<&str>,
) -> Result<PathBuf, CodegenError> {
    build_binary_impl(llvm_ir, output_path, target, vec![])
}

fn build_binary_impl(
    llvm_ir: String,
    output_path: &Path,
    target: Option<&str>,
    link_libs: Vec<String>,
) -> Result<PathBuf, CodegenError> {
    let resolved_target = resolve_target_triple(target);

    // WASM compilation path — uses WASI sysroot + wasi-libc + compiler-rt.
    if resolved_target.contains("wasm32") {
        return build_wasm_binary_impl(&llvm_ir, output_path, &resolved_target);
    }

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
    // Removes the directory on every exit path, success or error.
    let _build_dir_guard = BuildDirGuard(tmp_dir.clone());

    // 3. Write LLVM IR.
    let ll_path = tmp_dir.join("module.ll");
    std::fs::write(&ll_path, &llvm_ir).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write LLVM IR to '{}': {}", ll_path.display(), e),
    })?;

    // 4. Write embedded runtime sources.
    let h_path = tmp_dir.join("iris_runtime.h");
    let c_path = tmp_dir.join("iris_runtime.c");
    let onnx_h_path = tmp_dir.join("onnx_shim.h");
    let onnx_c_path = tmp_dir.join("onnx_shim.c");
    let tf_c_path = tmp_dir.join("tf_shim.c");
    let pytorch_cpp_path = tmp_dir.join("pytorch_shim.cpp");
    std::fs::write(&h_path, RUNTIME_H_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write runtime header: {}", e),
    })?;
    std::fs::write(&c_path, RUNTIME_C_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write runtime C source: {}", e),
    })?;
    // The ML dynamic loader and the vendored ONNX C API header must sit next to
    // the shims: they are included by onnx_shim.c / tf_shim.c and resolved via
    // the `-I <tmp_dir>` already passed to every compile step below.
    let ml_dynload_h_path = tmp_dir.join("iris_ml_dynload.h");
    let ort_api_h_path = tmp_dir.join("onnxruntime_c_api.h");
    std::fs::write(&ml_dynload_h_path, ML_DYNLOAD_H_SRC).map_err(|e| {
        CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("failed to write ML dynamic-loader header: {}", e),
        }
    })?;
    std::fs::write(&ort_api_h_path, ONNXRUNTIME_C_API_H_SRC).map_err(|e| {
        CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("failed to write vendored ONNX C API header: {}", e),
        }
    })?;
    std::fs::write(&onnx_h_path, ONNX_SHIM_H_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write ONNX shim header: {}", e),
    })?;
    std::fs::write(&onnx_c_path, ONNX_SHIM_C_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write ONNX shim source: {}", e),
    })?;
    std::fs::write(&tf_c_path, TF_SHIM_C_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write TensorFlow shim source: {}", e),
    })?;
    std::fs::write(&pytorch_cpp_path, PYTORCH_SHIM_CPP_SRC).map_err(|e| {
        CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("failed to write PyTorch shim source: {}", e),
        }
    })?;

    // ML compute kernels (conv2d, softmax, losses, optimizers, blocked matmul)
    let ml_h_path = tmp_dir.join("iris_ml_kernels.h");
    let ml_c_path = tmp_dir.join("iris_ml_kernels.c");
    std::fs::write(&ml_h_path, ML_KERNELS_H_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write ML kernels header: {}", e),
    })?;
    std::fs::write(&ml_c_path, ML_KERNELS_C_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("failed to write ML kernels source: {}", e),
    })?;

    // Locate compiler tools.
    // clang — compiles LLVM IR (.ll) to object files AND compiles the C
    //         runtime AND links the final binary (with -fuse-ld=lld).
    //         No GCC installation is required.
    let clang = find_clang();
    let msys2_inc = msys2_ucrt64_include();
    let msys2_lib = msys2_ucrt64_lib();
    let gcc_lib = msys2_gcc_lib();

    let target_args = ["-target".to_owned(), resolved_target.clone()];
    let onnx_sdk = if let Ok(dir) = std::env::var("ONNXRUNTIME_DIR") {
        Some(PathBuf::from(dir))
    } else if Path::new("C:\\onnxruntime").exists() {
        Some(PathBuf::from("C:\\onnxruntime"))
    } else {
        None
    };
    let tf_sdk = if let Ok(dir) = std::env::var("TENSORFLOW_DIR") {
        Some(PathBuf::from(dir))
    } else if Path::new("C:\\tensorflow").exists() {
        Some(PathBuf::from("C:\\tensorflow"))
    } else {
        None
    };
    let libtorch_sdk = if resolved_target.contains("msvc") {
        if let Ok(dir) = std::env::var("LIBTORCH_DIR") {
            Some(PathBuf::from(dir))
        } else if Path::new("C:\\libtorch").exists() {
            Some(PathBuf::from("C:\\libtorch"))
        } else {
            None
        }
    } else {
        None
    };
    let openblas_dir = if let Ok(dir) = std::env::var("OPENBLAS_DIR") {
        Some(PathBuf::from(dir))
    } else if Path::new("C:\\openblas").exists() {
        Some(PathBuf::from("C:\\openblas"))
    } else {
        None
    };
    let use_blas = openblas_dir.is_some() || std::env::var("IRIS_USE_BLAS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // NOTE: IRIS_NATIVE_ML_BACKENDS no longer affects codegen. ONNX and
    // TensorFlow are always compiled in and resolved at runtime via dlopen, and
    // LibTorch is a separately built plugin. The SDK paths below are retained
    // only so `iris build-plugin torch` can find LibTorch, and so the ONNX DLL
    // can be staged next to the output binary as a convenience.
    if std::env::var("IRIS_NATIVE_ML_BACKENDS").is_ok() {
        println!(
            "iris_codegen: note — IRIS_NATIVE_ML_BACKENDS is obsolete and ignored; \
             ML backends are now loaded at runtime"
        );
    }


    // 5a. Compile iris_runtime.c → iris_runtime.o using clang (cached).
    //     Cache lives next to the compiler binary or in ~/.iris/cache/.
    //     Invalidated when the compiler binary is newer than the cached .o.
    let rt_obj = tmp_dir.join("iris_runtime.o");
    let compile_rt = |c_path: &Path, out_obj: &Path| -> Result<(), CodegenError> {
        let mut cmd = Command::new(&clang);
        cmd.args(&target_args);
        cmd.args(["-O2", "-c", path_str(c_path)?, "-o", path_str(out_obj)?, "-I", path_str(&tmp_dir)?, "-Wno-pragma-pack"]);
        // No -D<BACKEND>_ENABLED flags: the ML backends are resolved at runtime
        // via dlopen (see iris_ml_dynload.h), so this object is byte-identical
        // whether or not any SDK is installed. That invariant is what allows a
        // single prebuilt iris_runtime.o to be shipped per target triple.
        if resolved_target.contains("windows") && !resolved_target.contains("msvc") {
            if let Some(ref inc) = msys2_inc { cmd.arg("-I").arg(inc); }
        }
        let output = cmd.output().map_err(|e| CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("'{}' not found: {}", clang, e),
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(CodegenError::Unsupported {
                backend: "binary".into(),
                detail: format!("'{}' failed to compile iris_runtime.c (exit: {:?})\nstderr: {}\nstdout: {}", clang, output.status.code(), stderr, stdout),
            });
        }
        Ok(())
    };
    // Preferred path: this compiler embeds runtime objects prebuilt for the
    // target, so no C compiler is needed at all. Anything present here was
    // verified by the build script to come from the current C sources.
    let prebuilt = find_prebuilt_runtime(&resolved_target);
    if let Some(pre) = prebuilt {
        for (name, bytes) in pre.objects {
            std::fs::write(tmp_dir.join(name), bytes).map_err(|e| CodegenError::Unsupported {
                backend: "binary".into(),
                detail: format!("failed to write prebuilt object '{}': {}", name, e),
            })?;
        }
        println!(
            "iris_codegen: using prebuilt runtime objects for {} (no C compiler required)",
            resolved_target
        );
    } else {
        let cache_dir = runtime_cache_dir();
        let cached_rt = cache_dir.join("iris_runtime.o");
        let use_cache = std::fs::metadata(&cached_rt).ok().and_then(|m| m.modified().ok()).and_then(|cache_mtime| {
            std::env::current_exe().ok().and_then(|exe| std::fs::metadata(exe).ok().and_then(|em| em.modified().ok()))
                .map(|exe_mtime| exe_mtime <= cache_mtime)
                .or(Some(false))
        }).unwrap_or(false);
        if use_cache {
            std::fs::copy(&cached_rt, &rt_obj).map_err(|e| CodegenError::Unsupported {
                backend: "binary".into(),
                detail: format!("failed to copy cached iris_runtime.o: {}", e),
            })?;
        } else {
            compile_rt(&c_path, &rt_obj)?;
            // Populate cache (best-effort)
            if std::fs::create_dir_all(&cache_dir).is_ok() {
                let _ = std::fs::copy(&rt_obj, &cached_rt);
            }
        }
    }

    // 5b. Compile LLVM IR → module.o using clang (only clang understands .ll).
    let mut support_objs = vec![rt_obj.clone()];
    // The ONNX and TensorFlow shims compile against the vendored/hand-declared
    // ABI in iris_ml_dynload.h and resolve their libraries via dlopen, so they
    // need no SDK include paths and no -D switches.
    //
    // pytorch_shim.cpp is deliberately absent: LibTorch is C++, so it is built
    // as a separate C-ABI plugin library rather than linked into every binary.
    // See build_torch_plugin() and iris_torch_available() in iris_runtime.c.
    for (src, obj_name, backend_name) in [
        (&onnx_c_path, "onnx_shim.o", "ONNX shim"),
        (&tf_c_path, "tf_shim.o", "TensorFlow shim"),
        (&ml_c_path, "iris_ml_kernels.o", "ML kernels"),
    ] {
        let obj = tmp_dir.join(obj_name);

        // Already written from the embedded prebuilt set — nothing to compile.
        if prebuilt.is_some() {
            support_objs.push(obj);
            continue;
        }

        let mut shim_cmd = Command::new(&clang);
        shim_cmd.args(&target_args);
        shim_cmd.args([
            "-O2",
            "-c",
            path_str(src)?,
            "-o",
            path_str(&obj)?,
            "-I",
            path_str(&tmp_dir)?,
            "-Wno-pragma-pack",
        ]);
        if resolved_target.contains("windows") && !resolved_target.contains("msvc") {
            if let Some(ref inc) = msys2_inc {
                shim_cmd.arg("-I").arg(inc);
            }
        }
        if backend_name == "ML kernels" && use_blas {
            shim_cmd.arg("-DIRIS_USE_BLAS");
            if let Some(ref dir) = openblas_dir {
                shim_cmd.arg("-I").arg(dir.join("include"));
            }
        }

        let shim_output = shim_cmd.output().map_err(|e| CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("'{}' could not compile {}: {}", clang, backend_name, e),
        })?;
        if !shim_output.status.success() {
            let stderr = String::from_utf8_lossy(&shim_output.stderr);
            let stdout = String::from_utf8_lossy(&shim_output.stdout);
            return Err(CodegenError::Unsupported {
                backend: "binary".into(),
                detail: format!(
                    "'{}' failed to compile {} (exit: {:?})\nstderr: {}\nstdout: {}",
                    clang,
                    backend_name,
                    shim_output.status.code(),
                    stderr,
                    stdout
                ),
            });
        }
        support_objs.push(obj);
    }

    // Compile .ll → .o: try clang first (safe, always works), fall back to
    // LLVM C API only when clang is unavailable. The LLVM C API conflicts with
    // TensorFlow in the same process (both embed LLVM), so we prefer clang.
    let mod_obj = tmp_dir.join("module.o");
    let mut ir_cmd = Command::new(&clang);
    ir_cmd.args(&target_args);
    ir_cmd.args([
        "-O2",
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
        // Clang failed; try LLVM C API as a last resort.
        if let Err(e) = crate::codegen::llvm_c_api::compile_llvm_ir_to_object(
            &llvm_ir,
            &mod_obj,
            Some(&resolved_target),
        ) {
            return Err(CodegenError::Unsupported {
                backend: "binary".into(),
                detail: format!(
                    "both clang and LLVM C API failed to compile LLVM IR: {}. clang exit: {:?}",
                    e,
                    ir_status.code()
                ),
            });
        }
    }

    // 6. Link module.o + iris_runtime.o → native binary.
    //    Try ld.lld directly first (avoids clang subprocess), fall back to clang.
    //    Note: ld.lld from LLVM 20+ doesn't support MinGW archive format,
    //    so skip it for MinGW targets and go straight to clang.
    let use_lld = !(resolved_target.contains("windows") && !resolved_target.contains("msvc"));
    let lld_path = if use_lld { crate::codegen::build::find_lld() } else { None };
    let link_result = if let Some(lld) = lld_path {
        link_with_lld(
            &lld,
            &resolved_target,
            &mod_obj,
            &support_objs,
            output_path,
            &target_args,
            &msys2_lib,
            &gcc_lib,
            &onnx_sdk,
            &tf_sdk,
            &libtorch_sdk,
            &openblas_dir,
            use_blas,
            &link_libs,
        )
    } else {
        Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("ld.lld not available (skipped for MinGW target {})", resolved_target),
        })
    };

    let link_output = match link_result {
        Ok(()) => {
            println!("iris_codegen: linked via ld.lld directly");
            stage_sqlite_dll_next_to(output_path);
            stage_onnxruntime_dll_next_to(output_path);
            return Ok(output_path.to_path_buf());
        }
        Err(e) => {
            println!("iris_codegen: ld.lld link failed ({}), falling back to clang", e);
            // Fallback: link via clang
            let mut link_cmd = Command::new(&clang);
            link_cmd.args(&target_args);
            if cfg!(target_os = "macos") {
                link_cmd.args(["-O2", path_str(&mod_obj)?]);
            } else {
                link_cmd.args(["-fuse-ld=lld", "-O2", path_str(&mod_obj)?]);
            }
            for obj in &support_objs {
                link_cmd.arg(path_str(obj)?);
            }
            link_cmd.args(["-o", path_str(output_path)?]);
            if !resolved_target.contains("msvc") {
                link_cmd.args(["-lm", "-lpthread"]);
            }
            if resolved_target.contains("windows") {
                link_cmd.arg("-lws2_32");
            }
            if resolved_target.contains("windows") && !resolved_target.contains("msvc") {
                if let Some(ref lib) = msys2_lib {
                    link_cmd.arg(format!("-L{}", lib));
                }
                if let Some(ref lib) = gcc_lib {
                    link_cmd.arg(format!("-L{}", lib));
                }
            }
            // ONNX Runtime, TensorFlow and LibTorch are intentionally NOT linked.
            // They are resolved at runtime (dlopen) so that a binary which never
            // calls into ML does not require the libraries to be present, and so
            // that the runtime object stays SDK-independent.
            if use_blas {
                if let Some(ref dir) = openblas_dir {
                    link_cmd.arg(format!("-L{}", dir.join("lib").display()));
                }
                link_cmd.arg("-lopenblas");
            }
            for lib in &link_libs {
                link_cmd.arg(format!("-l{}", lib));
            }
            link_cmd.output()
        }
    };

    let link_output = match link_output {
        Ok(link_output) => link_output,
        Err(e) => return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("'{}' link step could not start: {}", clang, e),
        }),
    };
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
    stage_onnxruntime_dll_next_to(output_path);
    Ok(output_path.to_path_buf())
}

/// Delete a per-build temp directory once its outputs have been linked.
///
/// These directories hold `module.ll`, the staged C runtime sources and the
/// object files — tens of megabytes for a single build. Nothing removed them, and
/// `--emit eval` creates one per evaluated program, so a test run leaked one
/// directory per test: 2129 of them had accumulated on this machine, occupying
/// roughly 23 GB and eventually filling the disk, which fails builds in ways that
/// look nothing like a disk problem.
///
/// Best-effort by design: a build that succeeded must not be reported as failed
/// because cleanup could not run. Set `IRIS_KEEP_BUILD_DIR=1` to retain the
/// directory when inspecting generated IR or objects.
fn cleanup_build_dir(tmp_dir: &std::path::Path) {
    if std::env::var("IRIS_KEEP_BUILD_DIR").is_ok() {
        println!(
            "iris_codegen: keeping build dir {} (IRIS_KEEP_BUILD_DIR set)",
            tmp_dir.display()
        );
        return;
    }
    let _ = std::fs::remove_dir_all(tmp_dir);
}

/// Removes a build directory when it goes out of scope.
///
/// `build_binary_impl` returns early through `?` at roughly a dozen points — a
/// missing compiler, a clang failure, a link error. Cleaning up only on the
/// success paths therefore still leaked one directory per *failed* build, which a
/// test run produces plenty of. Tying removal to scope exit covers every path,
/// including the error ones, without threading cleanup through each `?`.
struct BuildDirGuard(PathBuf);

impl Drop for BuildDirGuard {
    fn drop(&mut self) {
        cleanup_build_dir(&self.0);
    }
}

/// Build a WASM binary using the WASI sysroot (wasi-libc + compiler-rt).
fn build_wasm_binary_impl(
    llvm_ir: &str,
    output_path: &Path,
    target: &str,
) -> Result<PathBuf, CodegenError> {
    // Locate the WASI sysroot.
    let sysroot = find_wasi_sysroot()?;

    // Determine P1 vs P2 target
    let is_p2 = target.contains("wasip2");
    let wasm_target = if is_p2 { "wasm32-wasip2" } else { "wasm32-wasip1" };
    let wasm_lib_subdir = if is_p2 { "wasm32-wasip2" } else { "wasm32-wasip1" };

    // Set up temp directory for build artifacts.
    let build_id = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| format!("{}_wasm", s))
        .unwrap_or_else(|| format!("iris_wasm_build_{}", std::process::id()));
    let tmp_dir = std::env::temp_dir().join(build_id);
    std::fs::create_dir_all(&tmp_dir).map_err(|e| CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: format!("failed to create temp dir '{}': {}", tmp_dir.display(), e),
    })?;

    let clang = find_clang();

    // Write LLVM IR to module.ll
    let ll_path = tmp_dir.join("module.ll");
    std::fs::write(&ll_path, llvm_ir).map_err(|e| CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: format!("failed to write LLVM IR to '{}': {}", ll_path.display(), e),
    })?;

    // Write embedded runtime sources
    let h_path = tmp_dir.join("iris_runtime.h");
    let c_path = tmp_dir.join("iris_runtime.c");
    std::fs::write(&h_path, RUNTIME_H_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: format!("failed to write runtime header: {}", e),
    })?;
    std::fs::write(&c_path, RUNTIME_C_SRC).map_err(|e| CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: format!("failed to write runtime C source: {}", e),
    })?;

    // WASM compilation flags
    let wasi_defines = [
        "-D_POSIX_C_SOURCE=200809L",
        "-D_WASI_EMULATED_SIGNAL",
        "-D_WASI_EMULATED_PROCESS_CLOCKS",
        "-D_WASI_EMULATED_GETPID",
    ];
    let sysroot_opt = format!("--sysroot={}", sysroot.display());
    let inc_opt = format!("-I{}", tmp_dir.display());

    // Step 1: Compile iris_runtime.c → iris_runtime.o
    let rt_obj = tmp_dir.join("iris_runtime.o");
    let mut rt_cmd = std::process::Command::new(&clang);
    rt_cmd.args([
        "-target",
        wasm_target,
        "-O2",
        "-c",
        path_str(&c_path)?,
        "-o",
        path_str(&rt_obj)?,
        &inc_opt,
        &sysroot_opt,
    ]);
    if is_p2 {
        rt_cmd.arg("-D__wasip2__=1");
    }
    for def in &wasi_defines {
        rt_cmd.arg(def);
    }
    let rt_output = rt_cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: format!("'{}' not found: {}", clang, e),
    })?;
    if !rt_output.status.success() {
        let stderr = String::from_utf8_lossy(&rt_output.stderr);
        return Err(CodegenError::Unsupported {
            backend: "wasm".into(),
            detail: format!(
                "'{}' failed to compile iris_runtime.c for WASM (exit: {:?})\nstderr: {}",
                clang,
                rt_output.status.code(),
                stderr,
            ),
        });
    }

    // Step 3: Link → .wasm (pass LLVM IR directly instead of pre-compiling to .o)
    let sysroot_lib = sysroot.join("lib").join(wasm_lib_subdir);
    let p1_lib_dir = sysroot.join("lib").join("wasm32-wasip1");
    let mut link_cmd = std::process::Command::new(&clang);
    link_cmd.args([
        "-target",
        wasm_target,
        "-O2",
        &sysroot_opt,
        "-nodefaultlibs",
        path_str(&ll_path)?,       // LLVM IR directly (not pre-compiled)
        path_str(&rt_obj)?,        // C runtime object
    ]);
    // Add library paths
    link_cmd.arg(format!("-L{}", sysroot_lib.display()));
    if is_p2 {
        // P2 needs P1 lib dir for compiler-rt (not bundled in P2 sysroot)
        link_cmd.arg(format!("-L{}", p1_lib_dir.display()));
    }
    link_cmd.args([
        "-lc",
        "-lwasi-emulated-signal",
        "-lwasi-emulated-process-clocks",
        "-lclang_rt.builtins-wasm32",
        "-o",
        path_str(output_path)?,
    ]);
    let link_output = link_cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: format!("'{}' link step could not start: {}", clang, e),
    })?;
    if !link_output.status.success() {
        let stderr = String::from_utf8_lossy(&link_output.stderr);
        return Err(CodegenError::Unsupported {
            backend: "wasm".into(),
            detail: format!(
                "'{}' failed to link WASM binary (exit: {:?})\n{}",
                clang,
                link_output.status.code(),
                stderr,
            ),
        });
    }

    // Step 4 (P2 only): Post-process with wasm-tools → P2 component
    if is_p2 {
        let wasm_tools = find_wasm_tools()?;
        let adapter = find_wasi_p2_adapter()?;
        let p2_output = tmp_dir.join("output.p2.wasm");

        let mut component_cmd = std::process::Command::new(&wasm_tools);
        component_cmd.args([
            "component",
            "new",
            path_str(output_path)?,
            "--adapt",
            path_str(&adapter)?,
            "-o",
            path_str(&p2_output)?,
        ]);
        let comp_output = component_cmd.output().map_err(|e| CodegenError::Unsupported {
            backend: "wasm".into(),
            detail: format!("'{}' could not start: {}", wasm_tools.display(), e),
        })?;
        if !comp_output.status.success() {
            let stderr = String::from_utf8_lossy(&comp_output.stderr);
            return Err(CodegenError::Unsupported {
                backend: "wasm".into(),
                detail: format!(
                    "'{}' failed to convert WASM to P2 component (exit: {:?})\n{}",
                    wasm_tools.display(),
                    comp_output.status.code(),
                    stderr,
                ),
            });
        }
        // Replace the output with the P2 component
        std::fs::copy(&p2_output, output_path).map_err(|e| CodegenError::Unsupported {
            backend: "wasm".into(),
            detail: format!("failed to copy P2 component to output path: {}", e),
        })?;
    }

    Ok(output_path.to_path_buf())
}

/// Locate `wasm-tools` executable by searching PATH.
fn find_wasm_tools() -> Result<PathBuf, CodegenError> {
    // Check environment override
    if let Ok(path) = std::env::var("IRIS_WASM_TOOLS") {
        let p = PathBuf::from(&path);
        if p.is_file() { return Ok(p); }
    }
    // Search PATH using `where` (Windows) or `which` (Unix)
    #[cfg(windows)]
    let search_cmd = "where";
    #[cfg(not(windows))]
    let search_cmd = "which";
    if let Ok(output) = std::process::Command::new(search_cmd)
        .arg("wasm-tools")
        .output()
    {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !path_str.is_empty() {
                let p = PathBuf::from(&path_str);
                if p.is_file() { return Ok(p); }
            }
        }
    }
    Err(CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: "wasm-tools not found. Install with: 'cargo install wasm-tools' or set IRIS_WASM_TOOLS".into(),
    })
}

/// Locate the WASI P1→P2 adapter module.
fn find_wasi_p2_adapter() -> Result<PathBuf, CodegenError> {
    // Check environment override
    if let Ok(path) = std::env::var("IRIS_WASI_P2_ADAPTER") {
        let p = PathBuf::from(&path);
        if p.is_file() { return Ok(p); }
    }
    // Check alongside wasm-tools or in well-known locations
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    let candidates = [
        format!(r"{}\AppData\Local\Temp\wasi_snapshot_preview1.wasm", home),
        format!(r"{}\~\.iris\toolchain\wasi_snapshot_preview1.wasm", home),
        "wasi_snapshot_preview1.wasm".to_string(),
    ];
    for c in &candidates {
        let p = Path::new(c);
        if p.is_file() { return Ok(p.to_path_buf()); }
    }
    Err(CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: "WASI P1→P2 adapter module not found. Download from https://github.com/bytecodealliance/wasmtime/releases and set IRIS_WASI_P2_ADAPTER".into(),
    })
}

/// Locate the WASI sysroot directory (contains wasi-libc, compiler-rt, etc.).
fn find_wasi_sysroot() -> Result<PathBuf, CodegenError> {
    let candidates: Vec<PathBuf> = {
        let mut c = Vec::new();
        // IRIS_WASI_SYSROOT env var
        if let Ok(v) = std::env::var("IRIS_WASI_SYSROOT") {
            if !v.is_empty() {
                c.push(PathBuf::from(v));
            }
        }
        // Next to the running executable
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                c.push(dir.join("toolchain").join("wasi-sysroot"));
            }
        }
        // Inno Setup default install
        if let Ok(lad) = std::env::var("LOCALAPPDATA") {
            c.push(PathBuf::from(lad).join("Programs/IRIS/toolchain/wasi-sysroot"));
        }
        // User-local .iris install
        if let Ok(home) = std::env::var("USERPROFILE") {
            c.push(PathBuf::from(home).join(".iris/toolchain/wasi-sysroot"));
        }
        if let Ok(home) = std::env::var("HOME") {
            c.push(PathBuf::from(home).join(".iris/toolchain/wasi-sysroot"));
        }
        c
    };

    // Check each candidate for an actual sysroot version directory (e.g., wasi-sysroot-24.0)
    for base in &candidates {
        if !base.exists() {
            continue;
        }
        // Look for a versioned subdirectory or use the base itself
        if let Ok(entries) = std::fs::read_dir(base) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && path.file_name().and_then(|s| s.to_str()).map_or(false, |s| s.starts_with("wasi-sysroot")) {
                    // Verify it has lib/wasm32-wasip1/libc.a
                    let libc = path.join("lib/wasm32-wasip1/libc.a");
                    if libc.exists() {
                        return Ok(path);
                    }
                }
            }
            // Also check the base itself
            let libc = base.join("lib/wasm32-wasip1/libc.a");
            if libc.exists() {
                return Ok(base.to_path_buf());
            }
        }
    }

    Err(CodegenError::Unsupported {
        backend: "wasm".into(),
        detail: "WASI sysroot not found. Install the WASI SDK (wasi-sysroot) to ~/.iris/toolchain/wasi-sysroot/ or set IRIS_WASI_SYSROOT".into(),
    })
}

fn resolve_target_triple(target: Option<&str>) -> String {
    match target {
        Some(t) => crate::codegen::llvm_ir::target_preset_to_triple(t).unwrap_or(t),
        None => crate::codegen::llvm_ir::native_target_triple(),
    }
    .to_owned()
}

/// Locate `sqlite3.dll` to stage beside a built binary.
///
/// Explicit `SQLITE3_DIR` first, then any directory on `PATH`. No install
/// locations are hardcoded: this previously listed absolute paths to unrelated
/// third-party applications that happen to ship a `sqlite3.dll`, which resolve on
/// a single machine and disclose what was installed on it.
fn find_sqlite_dll() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::var_os("SQLITE3_DIR") {
        if !dir.is_empty() {
            dirs.push(PathBuf::from(dir));
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    for dir in dirs {
        for file_name in ["sqlite3.dll", "SQLite3.dll"] {
            let candidate = dir.join(file_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn stage_sqlite_dll_next_to(output_path: &Path) {
    let Some(source_path) = find_sqlite_dll() else {
        return;
    };

    if let Some(parent) = output_path.parent() {
        let target = parent.join(source_path.file_name().unwrap_or_default());
        let _ = std::fs::copy(&source_path, target);
    }
}

fn stage_onnxruntime_dll_next_to(output_path: &Path) {
    // `ONNXRUNTIME_DIR` overrides; the conventional install root is only the
    // fallback, so a non-standard location no longer requires editing the source.
    let root = std::env::var("ONNXRUNTIME_DIR").unwrap_or_else(|_| r"C:\onnxruntime".to_owned());
    let lib_dir = Path::new(&root).join("lib");

    let Some(parent) = output_path.parent() else {
        return;
    };
    for name in ["onnxruntime.dll", "onnxruntime_providers_shared.dll"] {
        let candidate = lib_dir.join(name);
        if candidate.is_file() {
            let _ = std::fs::copy(&candidate, parent.join(name));
        }
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

        // 3. Standalone LLVM version directories (LLVM 20, etc.)
        candidates.push(r"C:\llvm-20\bin\clang.exe".into());
        candidates.push(r"C:\llvm-19\bin\clang.exe".into());
        candidates.push(r"C:\llvm-18\bin\clang.exe".into());

        // 4. System-wide LLVM installs (standard Inno Setup path)
        candidates.push(r"C:\Program Files\LLVM\bin\clang.exe".into());
        candidates.push(r"C:\Program Files (x86)\LLVM\bin\clang.exe".into());

        // 5. Legacy user-local fallback
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

/// Find `ld.lld` (LLVM linker) — used as a clang-free alternative for linking.
/// Search order mirrors `find_clang`: env var, exe-relative, Inno Setup, system LLVM, PATH.
pub(crate) fn find_lld() -> Option<String> {
    if let Ok(v) = std::env::var("IRIS_LD") {
        if !v.is_empty() && std::path::Path::new(&v).exists() {
            return Some(v);
        }
    }

    let exe_name = if cfg!(target_os = "windows") {
        "ld.lld.exe"
    } else {
        "ld.lld"
    };

    let mut candidates: Vec<String> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(format!(r"{}\{}", dir.display(), exe_name));
            candidates.push(format!(r"{}\toolchain\llvm\bin\{}", dir.display(), exe_name));
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(lad) = std::env::var("LOCALAPPDATA") {
            candidates.push(format!(r"{}\Programs\IRIS\toolchain\llvm\bin\{}", lad, exe_name));
        }
        // Standalone LLVM version directories (LLVM 20, etc.)
        candidates.push(format!(r"C:\llvm-20\bin\{}", exe_name));
        candidates.push(format!(r"C:\llvm-19\bin\{}", exe_name));
        candidates.push(format!(r"C:\llvm-18\bin\{}", exe_name));
        // System-wide LLVM (standard Inno Setup path)
        candidates.push(format!(r"C:\Program Files\LLVM\bin\{}", exe_name));
        candidates.push(format!(r"C:\Program Files (x86)\LLVM\bin\{}", exe_name));
        if let Ok(home) = std::env::var("USERPROFILE") {
            candidates.push(format!(r"{}\.iris\toolchain\llvm\bin\{}", home, exe_name));
        }
        candidates.push(format!("/c/Program Files/LLVM/bin/{}", exe_name));
    }

    #[cfg(not(target_os = "windows"))]
    {
        candidates.push(format!("/usr/local/share/iris/toolchain/llvm/bin/{}", exe_name));
        candidates.push(format!("/opt/homebrew/opt/llvm/bin/{}", exe_name));
        candidates.push(format!("/usr/local/opt/llvm/bin/{}", exe_name));
        candidates.push(format!("/usr/bin/{}", exe_name));
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(format!("{}/.iris/toolchain/llvm/bin/{}", home, exe_name));
        }
    }

    for p in &candidates {
        if std::path::Path::new(p).exists() {
            return Some(p.clone());
        }
    }
    None
}

/// Link object files into a native binary using `ld.lld` directly (no clang).
/// This is a best-effort function: if linking fails, the caller falls back to clang.
#[allow(clippy::too_many_arguments)]
fn link_with_lld(
    lld_path: &str,
    target_triple: &str,
    main_obj: &Path,
    support_objs: &[PathBuf],
    output: &Path,
    _target_args: &[String],
    msys2_lib: &Option<String>,
    gcc_lib: &Option<String>,
    _onnx_sdk: &Option<PathBuf>,
    _tf_sdk: &Option<PathBuf>,
    _libtorch_sdk: &Option<PathBuf>,
    _openblas_dir: &Option<PathBuf>,
    _use_blas: bool,
    _link_libs: &[String],
) -> Result<(), CodegenError> {
    let mut cmd = std::process::Command::new(lld_path);

    // Input files
    cmd.arg(path_str(main_obj)?);
    for obj in support_objs {
        cmd.arg(path_str(obj)?);
    }

    // Output
    cmd.arg("-o");
    cmd.arg(path_str(output)?);

    // Target-specific linking
    if target_triple.contains("windows") && !target_triple.contains("msvc") {
        // MinGW target: need CRT, GCC runtime, and MinGW libraries
        if let Some(ref lib) = msys2_lib {
            cmd.arg(format!("-L{}", lib));
        }
        if let Some(ref lib) = gcc_lib {
            cmd.arg(format!("-L{}", lib));
        }

        // CRT startup objects (look in the GCC lib dir)
        if let Some(ref gcc) = gcc_lib {
            let crt_begin = Path::new(gcc).join("crtbegin.o");
            if crt_begin.exists() {
                cmd.arg(path_str(&crt_begin)?);
            }
            let crt_end = Path::new(gcc).join("crtend.o");
            if crt_end.exists() {
                cmd.arg(path_str(&crt_end)?);
            }
        }
        if let Some(ref lib) = msys2_lib {
            let crt1 = Path::new(lib).join("crt1.o");
            if crt1.exists() {
                cmd.arg(path_str(&crt1)?);
            }
            let crti = Path::new(lib).join("crti.o");
            if crti.exists() {
                cmd.arg(path_str(&crti)?);
            }
            let crtn = Path::new(lib).join("crtn.o");
            if crtn.exists() {
                cmd.arg(path_str(&crtn)?);
            }
        }

        // MinGW libraries
        cmd.arg("-lmingw32");
        cmd.arg("-lgcc");
        cmd.arg("-lgcc_eh");
        cmd.arg("-lmoldname");
        cmd.arg("-lmingwex");
        cmd.arg("-lmsvcrt");
        cmd.arg("-lpthread");
        cmd.arg("-lm");
        cmd.arg("-lws2_32");
    } else if target_triple.contains("msvc") {
        // MSVC target — use lld-link compatible flags
        // For MSVC, lld-link expects /NODEFAULTLIB etc.
        // This is a best-effort path; the clang fallback handles this better.
        cmd.arg("/defaultlib:libcmt");
        cmd.arg("/defaultlib:oldnames");
    } else {
        // Linux/macOS: standard system libraries
        cmd.arg("-lm");
        cmd.arg("-lpthread");
    }

    let output_result = cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "binary".into(),
        detail: format!("ld.lld could not start: {}", e),
    })?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("ld.lld failed: {}", stderr),
        });
    }

    Ok(())
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

/// Return a persistent cache directory for compiled C runtime objects.
/// Prefers a directory next to the compiler binary; falls back to ~/.iris/cache/.
fn runtime_cache_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("iris_cache");
            if candidate.exists() || std::fs::create_dir_all(&candidate).is_ok() {
                return candidate;
            }
        }
    }
    if let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        let candidate = PathBuf::from(home).join(".iris").join("cache");
        if candidate.exists() || std::fs::create_dir_all(&candidate).is_ok() {
            return candidate;
        }
    }
    std::env::temp_dir().join("iris_cache")
}

/// Build the optional LibTorch plugin (`iris_torch_plugin`).
///
/// LibTorch exposes a C++ API, so unlike ONNX/TensorFlow it cannot be `dlopen`ed
/// directly from the runtime. Instead `pytorch_shim.cpp` — whose entry points are
/// already `extern "C"` — is compiled into a standalone shared library that links
/// LibTorch, and `iris_torch_available()` in the runtime resolves its four
/// symbols at first use.
///
/// This keeps `iris_runtime.o` free of any LibTorch reference, which is what
/// allows a single prebuilt runtime object to be shipped per target triple.
///
/// `libtorch_dir` defaults to `$LIBTORCH_DIR`, then `C:\libtorch`.
/// Returns the path to the produced shared library.
pub fn build_torch_plugin(
    libtorch_dir: Option<&Path>,
    output_dir: &Path,
) -> Result<PathBuf, CodegenError> {
    let sdk: PathBuf = match libtorch_dir {
        Some(p) => p.to_path_buf(),
        None => std::env::var("LIBTORCH_DIR")
            .map(PathBuf::from)
            .ok()
            .or_else(|| {
                let d = PathBuf::from(r"C:\libtorch");
                d.exists().then_some(d)
            })
            .ok_or_else(|| CodegenError::Unsupported {
                backend: "torch-plugin".into(),
                detail: "LibTorch not found — set LIBTORCH_DIR to your install".into(),
            })?,
    };

    if !sdk.join("include").is_dir() {
        return Err(CodegenError::Unsupported {
            backend: "torch-plugin".into(),
            detail: format!("'{}' does not look like a LibTorch install (no include/)", sdk.display()),
        });
    }

    std::fs::create_dir_all(output_dir).map_err(CodegenError::Io)?;

    // Stage the sources the plugin needs: the shim itself plus the runtime
    // header it includes for IrisTensor.
    let src = output_dir.join("pytorch_shim.cpp");
    std::fs::write(&src, PYTORCH_SHIM_CPP_SRC).map_err(CodegenError::Io)?;
    std::fs::write(output_dir.join("iris_runtime.h"), RUNTIME_H_SRC).map_err(CodegenError::Io)?;

    let lib_name = if cfg!(target_os = "windows") {
        "iris_torch_plugin.dll"
    } else if cfg!(target_os = "macos") {
        "libiris_torch_plugin.dylib"
    } else {
        "libiris_torch_plugin.so"
    };
    let out = output_dir.join(lib_name);

    let clang = find_clang();
    let mut cmd = Command::new(&clang);
    cmd.args(["-shared", "-O2", "-x", "c++", "-std=c++17"]);
    cmd.arg("-DLIBTORCH_ENABLED");
    cmd.arg(path_str(&src)?);
    cmd.args(["-o", path_str(&out)?]);
    cmd.arg("-I").arg(output_dir);
    cmd.arg("-I").arg(sdk.join("include"));
    cmd.arg("-I").arg(sdk.join("include/torch/csrc/api/include"));
    cmd.arg(format!("-L{}", sdk.join("lib").display()));
    cmd.args(["-ltorch", "-ltorch_cpu", "-lc10"]);
    if !cfg!(target_os = "windows") {
        cmd.arg("-lstdc++");
    }

    let output = cmd.output().map_err(|e| CodegenError::Unsupported {
        backend: "torch-plugin".into(),
        detail: format!("'{}' could not start: {}", clang, e),
    })?;
    if !output.status.success() {
        return Err(CodegenError::Unsupported {
            backend: "torch-plugin".into(),
            detail: format!(
                "failed to build LibTorch plugin (exit {:?})\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ),
        });
    }

    Ok(out)
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
