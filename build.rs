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
    // Deliberately NOT watching .git/HEAD or .git/refs.
    //
    // Watching them keeps the embedded commit hash perfectly fresh, but it also
    // means any git operation — commit, checkout, even a branch read that touches
    // a ref mtime — invalidates this build script and therefore the whole crate,
    // forcing a full rebuild of the library and every test binary. On a small
    // machine that is minutes of rebuild per git command, which is a far higher
    // price than a possibly-stale hash in `--version`.
    //
    // Set IRIS_WATCH_GIT=1 when producing a release build, where an exact commit
    // stamp matters more than incremental build time.
    if std::env::var("IRIS_WATCH_GIT").is_ok() {
        println!("cargo:rerun-if-changed=.git/HEAD");
        println!("cargo:rerun-if-changed=.git/refs");
    }
    println!("cargo:rerun-if-env-changed=IRIS_WATCH_GIT");

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

    // Opt-in: regenerate the prebuilt runtime objects for this host, then embed
    // them. Two-stage by design — CI runs it once with a C compiler available so
    // that released binaries need none.
    //
    // Without this rerun trigger, setting the variable does nothing: cargo
    // considers the build script fresh and skips it, so the generator only ran
    // when some *other* input happened to change in the same invocation. That
    // silently shipped objects built from older C sources -- caught only because
    // the hash check below refused them. See known-issues #50.
    println!("cargo:rerun-if-env-changed=IRIS_GENERATE_PREBUILT");
    if std::env::var("IRIS_GENERATE_PREBUILT").is_ok() {
        generate_prebuilt_runtime();
    }

    // Embed any prebuilt runtime objects so end users need no C compiler.
    emit_prebuilt_runtime_table();

    // Attempt to compile the C runtime and optional shims.
    compile_c_runtime();
}

// ---------------------------------------------------------------------------
// Prebuilt runtime objects
// ---------------------------------------------------------------------------
//
// Compiling `iris_runtime.c` (6.6k lines) is the only reason a user of a
// released IRIS binary would need a C compiler. To remove that requirement we
// prebuild the runtime objects per target triple in CI and embed them here;
// `codegen::build` writes them straight out instead of invoking clang.
//
// Layout:
//   src/runtime/prebuilt/<target-triple>/
//       iris_runtime.o  onnx_shim.o  tf_shim.o  iris_ml_kernels.o
//       sources.hash          <- hash of the C sources they were built from
//
// STALENESS: an object built from older C sources would silently link the wrong
// runtime, which is a nasty class of bug. So each directory records the hash of
// the sources it came from, and any directory whose hash does not match the
// current sources is simply omitted from the table — the build then falls back
// to compiling with clang, exactly as before. Editing the C runtime therefore
// cannot produce a stale binary; it only costs you the prebuilt fast path until
// CI regenerates the objects.
fn emit_prebuilt_runtime_table() {
    use std::fmt::Write as _;

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest = std::path::Path::new(&out_dir).join("prebuilt_runtime.rs");

    println!("cargo:rerun-if-changed=src/runtime/prebuilt");

    let current_hash = runtime_sources_hash();

    // Object files that together make up the support objects for a link.
    const OBJECTS: [&str; 4] = [
        "iris_runtime.o",
        "onnx_shim.o",
        "tf_shim.o",
        "iris_ml_kernels.o",
    ];

    let mut entries = String::new();
    let mut count = 0usize;

    let root = std::path::Path::new("src/runtime/prebuilt");
    if let Ok(dirs) = std::fs::read_dir(root) {
        for dir in dirs.flatten() {
            if !dir.path().is_dir() {
                continue;
            }
            let triple = dir.file_name().to_string_lossy().into_owned();

            // Drop anything built from different sources.
            let recorded = std::fs::read_to_string(dir.path().join("sources.hash"))
                .unwrap_or_default()
                .trim()
                .to_owned();
            if recorded != current_hash {
                if !recorded.is_empty() {
                    println!(
                        "cargo:warning=prebuilt runtime for {} is stale (built from \
                         different C sources) — falling back to compiling it",
                        triple
                    );
                }
                continue;
            }

            // Require the full set; a partial set cannot satisfy a link.
            if !OBJECTS.iter().all(|o| dir.path().join(o).is_file()) {
                continue;
            }

            let mut objs = String::new();
            for obj in OBJECTS {
                let p = dir.path().join(obj);
                let _ = write!(
                    objs,
                    "        ({:?}, include_bytes!({:?})),\n",
                    obj,
                    p.canonicalize().unwrap_or(p)
                );
            }
            let _ = write!(
                entries,
                "    PrebuiltRuntime {{\n        triple: {:?},\n        objects: &[\n{}        ],\n    }},\n",
                triple, objs
            );
            count += 1;
        }
    }

    let src = format!(
        "// @generated by build.rs — do not edit.\n\
         /// A set of runtime objects prebuilt for one target triple.\n\
         pub struct PrebuiltRuntime {{\n\
         \x20   pub triple: &'static str,\n\
         \x20   /// (file name, contents) pairs, written verbatim at link time.\n\
         \x20   pub objects: &'static [(&'static str, &'static [u8])],\n\
         }}\n\n\
         /// Hash of the C sources this compiler was built from. Prebuilt objects\n\
         /// only appear below when they were produced from these same sources.\n\
         pub const RUNTIME_SOURCES_HASH: &str = {:?};\n\n\
         /// Prebuilt runtime objects embedded in this compiler ({} target(s)).\n\
         pub static PREBUILT_RUNTIMES: &[PrebuiltRuntime] = &[\n{}];\n",
        current_hash, count, entries
    );

    std::fs::write(&dest, src).expect("failed to write prebuilt_runtime.rs");
}

/// Compile the runtime objects for the host triple into `src/runtime/prebuilt/`.
///
/// Run via `IRIS_GENERATE_PREBUILT=1 cargo build`. This deliberately lives in the
/// build script rather than a standalone tool so it shares `runtime_sources_hash`
/// with the embedding step — two implementations of that hash could disagree and
/// would silently ship a stale runtime.
///
/// Honest caveat: this produces objects for the *host* only. Cross-target
/// prebuilts require running this on (or cross-compiling from) each target, which
/// is what the CI matrix is for.
fn generate_prebuilt_runtime() {
    // Key the directory by the *LLVM* triple codegen will ask for, not Cargo's
    // TARGET.
    //
    // These differ on Windows — Cargo says `x86_64-pc-windows-msvc` while
    // `codegen::native_target_triple()` returns `x86_64-pc-windows-gnu` — so
    // objects generated under the Cargo name were looked up under a name that
    // never occurs and the prebuilt path silently never fired. Worse, compiling
    // without `-target` produced MSVC-ABI objects that the MinGW link step could
    // not have used anyway.
    let target = llvm_host_triple();
    if target.is_empty() {
        println!("cargo:warning=IRIS_GENERATE_PREBUILT set but the target is unknown; skipping");
        return;
    }

    let out_dir = std::path::PathBuf::from("src/runtime/prebuilt").join(&target);
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        println!("cargo:warning=cannot create {}: {}", out_dir.display(), e);
        return;
    }

    // (source, object) pairs — must match OBJECTS in emit_prebuilt_runtime_table.
    let units = [
        ("src/runtime/iris_runtime.c", "iris_runtime.o"),
        ("src/runtime/onnx_shim.c", "onnx_shim.o"),
        ("src/runtime/tf_shim.c", "tf_shim.o"),
        ("src/runtime/iris_ml_kernels.c", "iris_ml_kernels.o"),
    ];

    let cc = std::env::var("IRIS_CLANG")
        .or_else(|_| std::env::var("CC"))
        .unwrap_or_else(|_| "clang".to_owned());

    // MinGW targets need the ucrt64 headers (pthread.h in particular), exactly as
    // the link-time path does. Overridable for non-standard installs.
    let msys2_inc = std::env::var("IRIS_MSYS2_INCLUDE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "C:/msys64/ucrt64/include".to_owned());

    for (src, obj) in units {
        let dest = out_dir.join(obj);
        let mut cmd = std::process::Command::new(&cc);
        // `-target` matters: without it clang uses its host default, which on
        // Windows is the MSVC ABI, and the resulting objects cannot link against
        // a MinGW-targeted module.
        cmd.args(["-target", &target]);
        cmd.args(["-O2", "-c", src, "-o"])
            .arg(&dest)
            .args(["-I", "src/runtime", "-I", "src/runtime/vendor", "-Wno-pragma-pack"]);
        if target.contains("windows") && !target.contains("msvc") {
            if std::path::Path::new(&msys2_inc).is_dir() {
                cmd.args(["-I", &msys2_inc]);
            }
        }
        let status = cmd.status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                println!("cargo:warning=prebuilt: '{}' failed on {} (exit {:?})", cc, src, s.code());
                return;
            }
            Err(e) => {
                println!("cargo:warning=prebuilt: cannot run '{}': {}", cc, e);
                return;
            }
        }
    }

    // Written last: the hash is what makes the set usable, so a partial failure
    // above leaves the directory without one and it is simply ignored.
    let hash = runtime_sources_hash();
    if let Err(e) = std::fs::write(out_dir.join("sources.hash"), &hash) {
        println!("cargo:warning=prebuilt: failed to write sources.hash: {}", e);
        return;
    }
    println!(
        "cargo:warning=generated prebuilt runtime for {} ({})",
        target, hash
    );
}

/// The LLVM target triple this compiler will hand to clang at *run* time.
///
/// Must agree with `codegen::native_target_triple()` in `src/codegen/llvm_ir.rs`.
/// The prebuilt objects are looked up by that name, so a disagreement here does
/// not fail loudly — it just means the lookup never matches and every user keeps
/// paying for a C compile. Derived from `CARGO_CFG_TARGET_*` rather than the
/// build host's own `cfg!`, so it stays correct under cross-compilation.
fn llvm_host_triple() -> String {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    match (os.as_str(), arch.as_str()) {
        ("windows", "x86_64") => "x86_64-pc-windows-gnu",
        ("windows", "aarch64") => "aarch64-pc-windows-gnu",
        ("macos", "x86_64") => "x86_64-apple-macosx14.0",
        ("macos", "aarch64") => "aarch64-apple-macosx14.0",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("linux", "riscv64") => "riscv64gc-unknown-linux-gnu",
        _ => "",
    }
    .to_owned()
}

/// FNV-1a 64 over every C source that contributes to the prebuilt objects.
///
/// A cryptographic hash is unnecessary here — this only needs to detect that
/// somebody edited the runtime, not resist an adversary — and FNV keeps the
/// build script dependency-free.
fn runtime_sources_hash() -> String {
    const SOURCES: [&str; 8] = [
        "src/runtime/iris_runtime.c",
        "src/runtime/iris_runtime.h",
        "src/runtime/onnx_shim.c",
        "src/runtime/onnx_shim.h",
        "src/runtime/tf_shim.c",
        "src/runtime/iris_ml_kernels.c",
        "src/runtime/iris_ml_kernels.h",
        "src/runtime/iris_ml_dynload.h",
    ];

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for path in SOURCES {
        println!("cargo:rerun-if-changed={}", path);
        // Normalise line endings before hashing. These are text files, and git
        // rewrites them on checkout (`core.autocrlf`), so raw bytes describe the
        // *checkout*, not the content. Hashing them raw made a committed
        // prebuilt set read as "built from different C sources" on the very
        // machine that generated it -- and would never match across two
        // machines with different autocrlf settings, which defeats the point of
        // shipping the objects at all. See known-issues #50.
        let raw = std::fs::read(path).unwrap_or_default();
        // CR (13) and LF (10) by value, so this file contains no escape that a
        // rewrite could turn into a literal newline.
        let bytes: Vec<u8> = {
            let mut out = Vec::with_capacity(raw.len());
            let mut i = 0;
            while i < raw.len() {
                let is_crlf = raw[i] == 13u8 && raw.get(i + 1) == Some(&10u8);
                if is_crlf {
                    i += 1; // drop the CR, keep the LF
                    continue;
                }
                out.push(raw[i]);
                i += 1;
            }
            out
        };
        // Mix the name too, so moving content between files changes the hash.
        for b in path.as_bytes().iter().chain(bytes.iter()) {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{:016x}", hash)
}

// Compile the C runtime + ML shims into the compiler binary.
//
// The ML backends are NOT selected here any more. ONNX Runtime and TensorFlow
// are resolved at runtime via dlopen (see src/runtime/iris_ml_dynload.h), and
// LibTorch is built separately as a C-ABI plugin. So this build:
//
//   * defines no *_ENABLED macros,
//   * links no ML libraries, and
//   * produces the same object regardless of which SDKs are installed.
//
// That invariant is what allows a single prebuilt iris_runtime.o to be shipped
// per target triple, which in turn removes the C-compiler requirement for users.
fn compile_c_runtime() {
    let mut build = cc::Build::new();
    build.file("src/runtime/iris_runtime.c");
    build.file("src/runtime/onnx_shim.c");
    build.file("src/runtime/tf_shim.c");
    build.include("src/runtime");
    // Supplies the vendored onnxruntime_c_api.h that iris_ml_dynload.h includes
    // for ABI layout. Header-only: it creates no link-time dependency.
    build.include("src/runtime/vendor");

    // pytorch_shim.cpp is deliberately NOT compiled here. LibTorch is C++ and
    // would drag a C++ toolchain plus the Torch libraries into every build of
    // the compiler. It is built on demand by codegen::build::build_torch_plugin.

    // The SDK directories are still probed purely to stage runtime DLLs next to
    // the compiler for convenience — never to link against.
    if let Some(onnx_dir) = sdk_dir("ONNXRUNTIME_DIR", r"C:\onnxruntime") {
        stage_runtime_dlls(&format!("{}/lib", onnx_dir));
    }
    if let Some(tf_dir) = sdk_dir("TENSORFLOW_DIR", r"C:\tensorflow") {
        stage_runtime_dlls(&format!("{}/lib", tf_dir));
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

// Retained for future non-ML SDKs. The ML backends no longer link against
// anything — they are resolved at runtime — so this currently has no callers.
#[allow(dead_code)]
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

/// Locate `sqlite3.dll` for staging next to build outputs.
///
/// Explicit `SQLITE3_DIR` first, then any directory on `PATH`.
///
/// This deliberately hardcodes no install locations. It previously listed the
/// absolute paths of unrelated third-party applications that happen to ship a
/// `sqlite3.dll`, which resolve on a single machine and disclose what was
/// installed on it. Discovery is environment-driven instead.
fn find_sqlite_dll() -> Option<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Some(dir) = std::env::var_os("SQLITE3_DIR") {
        if !dir.is_empty() {
            dirs.push(std::path::PathBuf::from(dir));
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

fn stage_sqlite_runtime_dll() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=SQLITE3_DIR");

    let Some(source_path) = find_sqlite_dll() else {
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
