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
    if let Ok(onnx_dir) = env::var("ONNXRUNTIME_DIR") {
        println!("cargo:rustc-cfg=onnx_runtime_enabled");
        build.define("ONNX_RUNTIME_ENABLED", None);
        println!("cargo:rustc-link-search=native={}", format!("{}/lib", onnx_dir));
        println!("cargo:rustc-link-lib=onnxruntime");
    }

    // LibTorch (optional C++ shim later)
    if let Ok(lt_dir) = env::var("LIBTORCH_DIR") {
        println!("cargo:rustc-cfg=libtorch_enabled");
        build.define("LIBTORCH_ENABLED", None);
        build.file("src/runtime/pytorch_shim.cpp");
        println!("cargo:rustc-link-search=native={}", format!("{}/lib", lt_dir));
        // Ensure C++ compilation and link flags
        build.cpp(true);
        if let Ok(cxxflags) = env::var("LIBTORCH_CXXFLAGS") {
            for flag in cxxflags.split_whitespace() { build.flag(flag); }
        }
    }

    // TensorFlow
    if let Ok(tf_dir) = env::var("TENSORFLOW_DIR") {
        println!("cargo:rustc-cfg=tensorflow_enabled");
        build.define("TENSORFLOW_ENABLED", None);
        println!("cargo:rustc-link-search=native={}", format!("{}/lib", tf_dir));
        println!("cargo:rustc-link-lib=tensorflow");
    }
    build.flag_if_supported("-std=c11");
    build.flag_if_supported("-fPIC");
    build.compile("iris_runtime_c");
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
