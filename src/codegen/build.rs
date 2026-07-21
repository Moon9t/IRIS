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

/// Native ML backend shims used by generated IRIS binaries.
pub const ONNX_SHIM_H_SRC: &str = include_str!("../runtime/onnx_shim.h");
pub const ONNX_SHIM_C_SRC: &str = include_str!("../runtime/onnx_shim.c");
pub const TF_SHIM_C_SRC: &str = include_str!("../runtime/tf_shim.c");
pub const PYTORCH_SHIM_CPP_SRC: &str = include_str!("../runtime/pytorch_shim.cpp");

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
    // Run the native binary with a 15-second timeout. Programs that use spawn/TCP may
    // hang indefinitely in the native runtime, so we fall back to the interpreter.
    let output = run_with_timeout(&run_path, std::time::Duration::from_secs(15))
        .map_err(|_| CodegenError::Unsupported {
            backend: "native".into(),
            detail: "native binary timed out (15s); try the interpreter instead".into(),
        })?;
    let _ = std::fs::remove_file(&run_path);
    Ok(output)
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

    // Helper: convert a PathBuf to &str, returning a descriptive error on non-UTF8 paths.
    fn path_str(p: &std::path::Path) -> Result<&str, CodegenError> {
        p.to_str().ok_or_else(|| CodegenError::Unsupported {
            backend: "binary".into(),
            detail: format!("path contains non-UTF8 characters: {}", p.display()),
        })
    }

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
    let native_ml_backends = onnx_sdk.is_some() || tf_sdk.is_some() || libtorch_sdk.is_some() || std::env::var("IRIS_NATIVE_ML_BACKENDS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    println!("iris_codegen: native_ml_backends = {}", native_ml_backends);
    println!("iris_codegen: onnx_sdk = {:?}", onnx_sdk);
    println!("iris_codegen: libtorch_sdk = {:?}", libtorch_sdk);

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
    if onnx_sdk.is_some() {
        compile_cmd.arg("-DONNX_RUNTIME_ENABLED");
    }
    if tf_sdk.is_some() {
        compile_cmd.arg("-DTENSORFLOW_ENABLED");
    }
    if libtorch_sdk.is_some() {
        compile_cmd.arg("-DLIBTORCH_ENABLED");
    }
    if resolved_target.contains("windows") && !resolved_target.contains("msvc") {
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
    let mut support_objs = vec![rt_obj.clone()];
    for (src, obj_name, backend_name) in [
        (&onnx_c_path, "onnx_shim.o", "ONNX shim"),
        (&tf_c_path, "tf_shim.o", "TensorFlow shim"),
        (&pytorch_cpp_path, "pytorch_shim.o", "PyTorch shim"),
        (&ml_c_path, "iris_ml_kernels.o", "ML kernels"),
    ] {
        let obj = tmp_dir.join(obj_name);
        let is_msvc = resolved_target.contains("msvc");
        let use_cl = backend_name == "PyTorch shim" && is_msvc;

        let mut shim_cmd = if use_cl {
            Command::new("cl.exe")
        } else {
            let mut cmd = Command::new(&clang);
            cmd.args(&target_args);
            cmd
        };

        if use_cl {
            shim_cmd.args([
                "/O2",
                "/c",
                "/EHsc",
                path_str(src)?,
                &format!("/Fo:{}", path_str(&obj)?),
                "/I",
                path_str(&tmp_dir)?,
            ]);
            if backend_name == "PyTorch shim" {
                if let Some(ref sdk) = libtorch_sdk {
                    shim_cmd.arg("/DLIBTORCH_ENABLED");
                    shim_cmd.arg("/std:c++17");
                    shim_cmd.arg("/I").arg(sdk.join("include"));
                    shim_cmd
                        .arg("/I")
                        .arg(sdk.join("include/torch/csrc/api/include"));
                }
            }
        } else {
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
            if backend_name == "ONNX shim" {
                if let Some(ref sdk) = onnx_sdk {
                    shim_cmd.arg("-DONNX_RUNTIME_ENABLED");
                    shim_cmd.arg("-I").arg(sdk.join("include"));
                }
            }
            if backend_name == "TensorFlow shim" {
                if let Some(ref sdk) = tf_sdk {
                    shim_cmd.arg("-DTENSORFLOW_ENABLED");
                    shim_cmd.arg("-I").arg(sdk.join("include"));
                }
            }
            if backend_name == "PyTorch shim" {
                shim_cmd.arg("-x").arg("c++");
                if let Some(ref sdk) = libtorch_sdk {
                    shim_cmd.arg("-DLIBTORCH_ENABLED");
                    shim_cmd.arg("-std=c++17");
                    if resolved_target.contains("windows") {
                        shim_cmd.arg("-fms-extensions");
                        shim_cmd.arg("-DNDEBUG");
                        shim_cmd.arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
                        if !resolved_target.contains("msvc") {
                            shim_cmd.arg("-DC10_USING_CUSTOM_GENERATED_MACROS");
                        }
                    }
                    shim_cmd.arg("-I").arg(sdk.join("include"));
                    shim_cmd
                        .arg("-I")
                        .arg(sdk.join("include/torch/csrc/api/include"));
                }
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

    // Use -O2 for user IR — we bundle LLVM 18 which handles complex IR patterns.
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
    // Apple clang rejects -fuse-ld=lld; only request lld on non-macOS hosts.
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
    // Windows: link WinSock2 for TCP/HTTP builtins
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
    if let Some(ref sdk) = onnx_sdk {
        link_cmd.arg(format!("-L{}", sdk.join("lib").display()));
        link_cmd.arg("-lonnxruntime");
    }
    if let Some(ref sdk) = tf_sdk {
        link_cmd.arg(format!("-L{}", sdk.join("lib").display()));
        link_cmd.arg("-ltensorflow");
    }
    if let Some(ref sdk) = libtorch_sdk {
        link_cmd.arg(format!("-L{}", sdk.join("lib").display()));
        link_cmd.arg("-ltorch");
        link_cmd.arg("-ltorch_cpu");
        link_cmd.arg("-lc10");
        if !resolved_target.contains("msvc") {
            link_cmd.arg("-lstdc++");
        }
    }
    if use_blas {
        if let Some(ref dir) = openblas_dir {
            link_cmd.arg(format!("-L{}", dir.join("lib").display()));
        }
        link_cmd.arg("-lopenblas");
    }
    // Link libraries specified by extern "C" declarations with @link(name = "lib")
    for lib in &link_libs {
        link_cmd.arg(format!("-l{}", lib));
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
    stage_onnxruntime_dll_next_to(output_path);
    Ok(output_path.to_path_buf())
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

    // Helper: PathBuf → &str with error.
    fn path_str(p: &Path) -> Result<&str, CodegenError> {
        p.to_str().ok_or_else(|| CodegenError::Unsupported {
            backend: "wasm".into(),
            detail: format!("path contains non-UTF8 characters: {}", p.display()),
        })
    }

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

fn stage_onnxruntime_dll_next_to(output_path: &Path) {
    let candidate = Path::new("C:\\onnxruntime\\lib\\onnxruntime.dll");
    if !candidate.exists() {
        return;
    }
    if let Some(parent) = output_path.parent() {
        let target = parent.join("onnxruntime.dll");
        let _ = std::fs::copy(candidate, target);
    }
    // Also stage the providers shared DLL
    let providers = Path::new("C:\\onnxruntime\\lib\\onnxruntime_providers_shared.dll");
    if providers.exists() {
        if let Some(parent) = output_path.parent() {
            let target = parent.join("onnxruntime_providers_shared.dll");
            let _ = std::fs::copy(providers, target);
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
