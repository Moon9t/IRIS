//! Phase 39 integration tests: `to_str(v)` and `format("...", args...)` builtins.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. to_str(i64) converts integer to string
// ---------------------------------------------------------------------------
#[test]
fn test_to_str_int() {
    let src = r#"
def f() -> str {
    to_str(42)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("42"), "expected '42' in output, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 2. to_str(f64) converts float to string
// ---------------------------------------------------------------------------
#[test]
fn test_to_str_float() {
    let src = r#"
def f() -> str {
    to_str(3.14)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("3.14"), "expected '3.14' in output, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. to_str(bool) converts bool to string
// ---------------------------------------------------------------------------
#[test]
fn test_to_str_bool() {
    let src = r#"
def f() -> str {
    to_str(true)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("true"), "expected 'true' in output, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. format with one placeholder
// ---------------------------------------------------------------------------
#[test]
fn test_format_one_arg() {
    let src = r#"
def f() -> str {
    format("Hello, {}!", "world")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(
        out.contains("Hello, world!") || out.contains("Hello,") && out.contains("world!"),
        "expected 'Hello, world!', got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 5. format with integer argument
// ---------------------------------------------------------------------------
#[test]
fn test_format_int_arg() {
    let src = r#"
def f() -> str {
    format("count: {}", 42)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("count:") && out.contains("42"), "expected 'count: 42', got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. format with multiple placeholders
// ---------------------------------------------------------------------------
#[test]
fn test_format_multi_arg() {
    let src = r#"
def f() -> str {
    format("{} + {} = {}", 1, 2, 3)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(
        out.contains("1") && out.contains("2") && out.contains("3"),
        "expected '1 + 2 = 3', got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 7. format appears in IR as str_concat chain
// ---------------------------------------------------------------------------
#[test]
fn test_format_ir() {
    let src = r#"
def f() -> str {
    format("x={}", 10)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should emit IR");
    assert!(
        out.contains("str_concat") || out.contains("to_str"),
        "IR should contain str_concat or to_str, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 8. to_str used in concat to build a message
// ---------------------------------------------------------------------------
#[test]
fn test_to_str_in_concat() {
    let src = r#"
def f() -> str {
    val n = 7
    concat("n=", to_str(n))
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("n=") && out.contains("7"), "expected 'n=7', got: {}", out.trim());
}
