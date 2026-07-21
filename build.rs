// Cargo build script — ensures the embedded C runtime files
// trigger a rebuild when they change (include_str! alone is not
// tracked by Cargo's dependency fingerprinting).

fn main() {
    println!("cargo:rustc-check-cfg=cfg(onnx_runtime_enabled)");
    println!("cargo:rustc-check-cfg=cfg(libtorch_enabled)");
    println!("cargo:rustc-check-cfg=cfg(tensorflow_enabled)");
    println!("cargo:rerun-if-changed=src/runtime/iris_runtime.c");
    println!("cargo:rerun-if-changed=src/runtime/iris_runtime.h");
    println!("cargo:rerun-if-changed=src/runtime/onnx_shim.c");
    println!("cargo:rerun-if-changed=src/runtime/onnx_shim.h");
    println!("cargo:rerun-if-env-changed=SQLITE3_DIR");
    println!("cargo:rerun-if-env-changed=IRIS_LINK_ML_SHIMS");
    println!("cargo:rerun-if-env-changed=IRIS_USE_DEFAULT_SDK_PATHS");
    // Re-run when HEAD changes (new commit / checkout).
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");

    // Capture build-time metadata for --version output.
    let build_date = chrono_lite_date();
    println!("cargo:rustc-env=IRIS_BUILD_DATE={}", build_date);

    // Git commit hash (short + long).
    let (git_hash_short, git_hash_long) = git_commit_hashes();
    println!("cargo:rustc-env=IRIS_GIT_HASH_SHORT={}", git_hash_short);
    println!("cargo:rustc-env=IRIS_GIT_HASH={}", git_hash_long);

    // Git branch name.
    let git_branch = git_branch();
    println!("cargo:rustc-env=IRIS_GIT_BRANCH={}", git_branch);

    // Git dirty flag.
    let git_dirty = git_is_dirty();
    println!(
        "cargo:rustc-env=IRIS_GIT_DIRTY={}",
        if git_dirty { "true" } else { "false" }
    );

    // Rustc version.
    let rustc_ver = rustc_version();
    println!("cargo:rustc-env=IRIS_RUSTC_VERSION={}", rustc_ver);

    // Target triple (set by Cargo).
    if let Ok(target) = std::env::var("TARGET") {
        println!("cargo:rustc-env=IRIS_TARGET={}", target);
    }
    // Host triple.
    if let Ok(host) = std::env::var("HOST") {
        println!("cargo:rustc-env=IRIS_HOST={}", host);
    }
    // Profile (debug / release).
    if let Ok(profile) = std::env::var("PROFILE") {
        println!("cargo:rustc-env=IRIS_PROFILE={}", profile);
    }
    // OPT_LEVEL.
    if let Ok(opt) = std::env::var("OPT_LEVEL") {
        println!("cargo:rustc-env=IRIS_OPT_LEVEL={}", opt);
    }

    // Attempt to compile the C runtime and optional shims.
    compile_c_runtime();
}

// Compile C runtime + optional shims. Detect backend dirs via env vars:
// ONNXRUNTIME_DIR, LIBTORCH_DIR, TENSORFLOW_DIR. When present, enable
// corresponding -D flags and add link search paths + libs.
fn compile_c_runtime() {
    use std::env;

    let mut build = cc::Build::new();
    build.file("src/runtime/iris_runtime.c");
    build.file("src/runtime/onnx_shim.c");
    build.file("src/runtime/tf_shim.c");
    build.include("src/runtime");

    // ONNX Runtime
    if let Some(onnx_dir) = sdk_dir("ONNXRUNTIME_DIR", r"C:\onnxruntime") {
        println!("cargo:rustc-cfg=onnx_runtime_enabled");
        build.define("ONNX_RUNTIME_ENABLED", None);
        let include_dir = format!("{}/include", onnx_dir);
        let lib_dir = format!("{}/lib", onnx_dir);
        build.include(&include_dir);
        println!("cargo:rustc-link-search=native={}", lib_dir);
        link_lib_if_present(&lib_dir, "onnxruntime");
        stage_runtime_dlls(&lib_dir);
    }

    // LibTorch (optional C++ shim later)
    if let Some(lt_dir) = sdk_dir("LIBTORCH_DIR", r"C:\libtorch") {
        println!("cargo:rustc-cfg=libtorch_enabled");
        build.define("LIBTORCH_ENABLED", None);
        let include_dir = format!("{}/include", lt_dir);
        let lib_dir = format!("{}/lib", lt_dir);
        build.include(&include_dir);
        build.file("src/runtime/pytorch_shim.cpp");
        build.cpp(true);
        if cfg!(windows) {
            build.flag("/std:c++20");
        } else {
            build.flag("-std=c++20");
        }
        if let Ok(cxxflags) = env::var("LIBTORCH_CXXFLAGS") {
            for flag in cxxflags.split_whitespace() {
                build.flag(flag);
            }
        }
        println!("cargo:rustc-link-search=native={}", lib_dir);
        for lib in ["torch", "torch_cpu", "c10", "torch_global_deps"] {
            link_lib_if_present(&lib_dir, lib);
        }
        stage_runtime_dlls(&lib_dir);
    }

    // TensorFlow
    if let Some(tf_dir) = sdk_dir("TENSORFLOW_DIR", r"C:\tensorflow") {
        println!("cargo:rustc-cfg=tensorflow_enabled");
        build.define("TENSORFLOW_ENABLED", None);
        let include_dir = format!("{}/include", tf_dir);
        let lib_dir = format!("{}/lib", tf_dir);
        build.include(&include_dir);
        println!("cargo:rustc-link-search=native={}", lib_dir);
        link_lib_if_present(&lib_dir, "tensorflow");
        stage_runtime_dlls(&lib_dir);
    }
    build.flag_if_supported("-std=gnu11");
    build.flag_if_supported("-fPIC");
    build.compile("iris_runtime_c");
    stage_sqlite_runtime_dll();
}

fn sdk_dir(var_name: &str, default_path: &str) -> Option<String> {
    if let Ok(val) = std::env::var(var_name) {
        if !val.is_empty() {
            return Some(val);
        }
    }
    let candidate = std::path::Path::new(default_path);
    if candidate.exists() {
        return Some(default_path.to_owned());
    }
    None
}

fn link_lib_if_present(lib_dir: &str, lib_name: &str) {
    let candidates = if cfg!(windows) {
        vec![format!("{}.lib", lib_name)]
    } else if cfg!(target_os = "macos") {
        vec![
            format!("lib{}.dylib", lib_name),
            format!("lib{}.a", lib_name),
        ]
    } else {
        vec![format!("lib{}.so", lib_name), format!("lib{}.a", lib_name)]
    };

    let lib_dir = std::path::Path::new(lib_dir);
    for candidate in candidates {
        if std::fs::metadata(lib_dir.join(&candidate)).is_ok() {
            println!("cargo:rustc-link-lib={}", lib_name);
            return;
        }
    }
}

fn stage_runtime_dlls(lib_dir: &str) {
    let out_dir = match std::env::var("OUT_DIR") {
        Ok(v) => v,
        Err(_) => return,
    };

    let mut profile_dir = std::path::PathBuf::from(out_dir);
    for _ in 0..3 {
        if !profile_dir.pop() {
            return;
        }
    }

    let deps_dir = profile_dir.join("deps");
    let examples_dir = profile_dir.join("examples");
    let _ = std::fs::create_dir_all(&deps_dir);
    let _ = std::fs::create_dir_all(&examples_dir);

    let entries = match std::fs::read_dir(lib_dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let is_dll = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("dll"))
            .unwrap_or(false);
        if !is_dll {
            continue;
        }

        if let Some(file_name) = path.file_name() {
            let profile_target = profile_dir.join(file_name);
            let deps_target = deps_dir.join(file_name);
            let examples_target = examples_dir.join(file_name);
            let _ = std::fs::copy(&path, profile_target);
            let _ = std::fs::copy(&path, deps_target);
            let _ = std::fs::copy(&path, examples_target);
        }
    }
}

fn stage_sqlite_runtime_dll() {
    use std::path::{Path, PathBuf};

    let candidate_dirs = [
        std::env::var("SQLITE3_DIR").ok(),
        Some(r"C:\Program Files\Common Files\Apple\Mobile Device Support".to_owned()),
        Some(r"C:\Program Files\Cheat Engine\win64".to_owned()),
        Some(r"C:\Program Files (x86)\Common Files\Apple\Mobile Device Support".to_owned()),
        Some(r"C:\Program Files (x86)\Passixer\Passixer iPhone Unlocker".to_owned()),
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

    let out_dir = match std::env::var("OUT_DIR") {
        Ok(v) => v,
        Err(_) => return,
    };

    let mut profile_dir = PathBuf::from(out_dir);
    for _ in 0..3 {
        if !profile_dir.pop() {
            return;
        }
    }

    let deps_dir = profile_dir.join("deps");
    let examples_dir = profile_dir.join("examples");
    let _ = std::fs::create_dir_all(&deps_dir);
    let _ = std::fs::create_dir_all(&examples_dir);
    if let Some(file_name) = source_path.file_name() {
        let profile_target = profile_dir.join(file_name);
        let deps_target = deps_dir.join(file_name);
        let examples_target = examples_dir.join(file_name);
        let _ = std::fs::copy(&source_path, profile_target);
        let _ = std::fs::copy(&source_path, deps_target);
        let _ = std::fs::copy(&source_path, examples_target);
    }
}

// Run compile step after the metadata printed above.
#[allow(dead_code)]
fn _maybe_compile() {
    compile_c_runtime();
}

/// Minimal date helper that doesn't depend on the `chrono` crate.
/// Returns YYYY-MM-DD in UTC.
fn chrono_lite_date() -> String {
    use std::process::Command;
    #[cfg(unix)]
    {
        if let Ok(out) = Command::new("date").arg("+%Y-%m-%d").output() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_owned();
            if !s.is_empty() {
                return s;
            }
        }
    }
    #[cfg(windows)]
    {
        if let Ok(out) = Command::new("powershell")
            .args(["-NoProfile", "-Command", "Get-Date -Format 'yyyy-MM-dd'"])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_owned();
            if !s.is_empty() {
                return s;
            }
        }
    }
    "unknown".to_owned()
}

/// Returns (short_hash, long_hash) from `git rev-parse`.
fn git_commit_hashes() -> (String, String) {
    use std::process::Command;
    let long = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_owned();
            if s.is_empty() || !o.status.success() {
                None
            } else {
                Some(s)
            }
        })
        .unwrap_or_else(|| "unknown".to_owned());
    let short = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_owned();
            if s.is_empty() || !o.status.success() {
                None
            } else {
                Some(s)
            }
        })
        .unwrap_or_else(|| "unknown".to_owned());
    (short, long)
}

/// Returns the current git branch name.
fn git_branch() -> String {
    use std::process::Command;
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_owned();
            if s.is_empty() || !o.status.success() {
                None
            } else {
                Some(s)
            }
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Returns true if the working tree has uncommitted changes.
fn git_is_dirty() -> bool {
    use std::process::Command;
    Command::new("git")
        .args(["diff", "--quiet", "HEAD"])
        .status()
        .map(|s| !s.success())
        .unwrap_or(false)
}

/// Returns the rustc version string (e.g. "rustc 1.78.0 (9b00956e5 2024-04-29)").
fn rustc_version() -> String {
    use std::process::Command;
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_owned();
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

// Touched to trigger stdlib rebuild with legacy ROS2 backward compatibility.
