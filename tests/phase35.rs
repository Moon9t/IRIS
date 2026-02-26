//! Phase 35 integration tests: global constants (`const` declarations).
//!
//! `const NAME = expr` defines a module-level constant that is inlined
//! (as a compile-time value) at every use within functions.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Simple integer constant is accessible inside a function
// ---------------------------------------------------------------------------
#[test]
fn test_const_int_eval() {
    let src = r#"
const ANSWER = 42

def f() -> i64 {
    ANSWER
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "ANSWER should be 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 2. Float constant is accessible
// ---------------------------------------------------------------------------
#[test]
fn test_const_float_eval() {
    let src = r#"
const PI = 3.14159

def f() -> f64 {
    PI
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 3.14159).abs() < 1e-5, "PI should be ~3.14159, got: {}", v);
}

// ---------------------------------------------------------------------------
// 3. Constant used in arithmetic expression
// ---------------------------------------------------------------------------
#[test]
fn test_const_in_expr() {
    let src = r#"
const BASE = 10

def f() -> i64 {
    BASE * 2 + 5
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "25", "BASE*2+5 = 25, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. Multiple constants
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_consts() {
    let src = r#"
const A = 3
const B = 4

def f() -> i64 {
    A + B
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "7", "3 + 4 = 7, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. Constant used as function argument
// ---------------------------------------------------------------------------
#[test]
fn test_const_as_arg() {
    let src = r#"
const SIZE = 100

def double(x: i64) -> i64 { x * 2 }

def f() -> i64 {
    double(SIZE)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "200", "double(SIZE) = 200, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. Constant with type annotation
// ---------------------------------------------------------------------------
#[test]
fn test_const_with_type_annotation() {
    let src = r#"
const LIMIT: i64 = 99

def f() -> i64 {
    LIMIT
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "99", "LIMIT should be 99, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. String constant
// ---------------------------------------------------------------------------
#[test]
fn test_const_string() {
    let src = r#"
const GREETING = "hello"

def f() -> str {
    GREETING
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("hello"), "GREETING should contain 'hello', got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. Constant appears in IR (not folded away by name)
// ---------------------------------------------------------------------------
#[test]
fn test_const_compiles_to_ir() {
    let src = r#"
const MAGIC = 7

def f() -> i64 {
    MAGIC
}
"#;
    // Just verify it compiles without error.
    compile(src, "test", EmitKind::Ir).expect("const should compile");
}
