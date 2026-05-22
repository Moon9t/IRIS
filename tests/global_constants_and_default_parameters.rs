//! Phase 126 integration tests: global constants and default parameters.

use iris::{compile, EmitKind};

// ── 1. Global constant integer ──────────────────────────────────────────────
#[test]
fn test_global_const_int() {
    let src = r#"
const MAX_SIZE: i64 = 100
def f() -> i64 {
    MAX_SIZE
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "100");
}

// ── 2. Global constant in expression ────────────────────────────────────────
#[test]
fn test_global_const_expr() {
    let src = r#"
const OFFSET: i64 = 10
def f() -> i64 {
    val x = 5
    x + OFFSET
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15");
}

// ── 3. Multiple global constants ────────────────────────────────────────────
#[test]
fn test_multiple_globals() {
    let src = r#"
const A: i64 = 10
const B: i64 = 20
def f() -> i64 {
    A + B
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "30");
}

// ── 4. Global constant string ───────────────────────────────────────────────
#[test]
fn test_global_const_string() {
    let src = r#"
const GREETING: str = "Hello"
def f() -> str {
    GREETING
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Hello");
}

// ── 5. Default parameter value ──────────────────────────────────────────────
#[test]
fn test_default_param() {
    let src = r#"
def greet(name: str = "World") -> str {
    concat("Hello ", name)
}
def f() -> str {
    greet()
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Hello World");
}

// ── 6. Default parameter overridden ─────────────────────────────────────────
#[test]
fn test_default_param_override() {
    let src = r#"
def greet(name: str = "World") -> str {
    concat("Hello ", name)
}
def f() -> str {
    greet("IRIS")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Hello IRIS");
}

// ── 7. Default parameter integer ────────────────────────────────────────────
#[test]
fn test_default_param_int() {
    let src = r#"
def add(a: i64, b: i64 = 10) -> i64 {
    a + b
}
def f() -> i64 {
    add(5) + add(5, 20)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "40");
}

// ── 8. Global constant in loop bound ────────────────────────────────────────
#[test]
fn test_global_const_in_loop() {
    let src = r#"
const LIMIT: i64 = 5
def f() -> i64 {
    var sum = 0
    for i in 1..LIMIT {
        sum = sum + i
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 1+2+3+4 = 10
    assert_eq!(result.trim(), "10");
}
