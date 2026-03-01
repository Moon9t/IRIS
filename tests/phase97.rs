//! Phase 97 integration tests: stdlib I/O & OS — time, path, fs.

use iris::{compile_multi, EmitKind};

// ── 1. now_ms() returns a positive i64 ──────────────────────────────────────
#[test]
fn test_now_ms_positive() {
    let src = r#"
bring std.time
def f() -> i64 {
    val t = now_ms()
    if t > 0 { 1 } else { 0 }
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 2. sleep(10) returns 0 without error ────────────────────────────────────
#[test]
fn test_sleep_returns_zero() {
    let src = r#"
bring std.time
def f() -> i64 {
    sleep(10)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ── 3. join_path("a", "b") → "a/b" ─────────────────────────────────────────
#[test]
fn test_join_path() {
    let src = r#"
bring std.path
def f() -> str {
    join_path("a", "b")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "a/b");
}

// ── 4. basename("/foo/bar.iris") → "bar.iris" ───────────────────────────────
#[test]
fn test_basename() {
    let src = r#"
bring std.path
def f() -> str {
    basename("/foo/bar.iris")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "bar.iris");
}

// ── 5. extension("file.iris") → "iris" ─────────────────────────────────────
#[test]
fn test_extension() {
    let src = r#"
bring std.path
def f() -> str {
    extension("file.iris")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "iris");
}

// ── 6. write_text then read_text round-trips ────────────────────────────────
#[test]
fn test_write_read_roundtrip() {
    let src = r#"
bring std.fs
def f() -> str {
    val path = "/tmp/iris_phase97_test.txt"
    val ok = write_text(path, "hello iris")
    read_text(path)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello iris");
}

// ── 7. dirname("/foo/bar.iris") → "/foo" ───────────────────────────────────
#[test]
fn test_dirname() {
    let src = r#"
bring std.path
def f() -> str {
    dirname("/foo/bar.iris")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "/foo");
}

// ── 8. stem("hello.world.iris") → "hello.world" ────────────────────────────
#[test]
fn test_stem() {
    let src = r#"
bring std.path
def f() -> str {
    stem("hello.world.iris")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello.world");
}
