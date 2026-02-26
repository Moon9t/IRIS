//! Phase 32 integration tests: math builtins — sqrt, abs, floor, ceil, pow, min, max.
//!
//! These are all lowered to `UnaryOp`/`BinOp` IR instructions and folded by
//! `ConstFoldPass` when the operands are compile-time constants.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. sqrt(4.0) == 2.0
// ---------------------------------------------------------------------------
#[test]
fn test_sqrt_eval() {
    let src = r#"
def f() -> f64 {
    sqrt(4.0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 2.0).abs() < 1e-9, "sqrt(4.0) should be 2.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 2. abs(-7.0) == 7.0
// ---------------------------------------------------------------------------
#[test]
fn test_abs_float_eval() {
    let src = r#"
def f() -> f64 {
    abs(-7.0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 7.0).abs() < 1e-9, "abs(-7.0) should be 7.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 3. abs(-5) == 5  (integer)
// ---------------------------------------------------------------------------
#[test]
fn test_abs_int_eval() {
    let src = r#"
def f() -> i64 {
    abs(-5)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "5", "abs(-5) should be 5, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. floor(3.7) == 3.0
// ---------------------------------------------------------------------------
#[test]
fn test_floor_eval() {
    let src = r#"
def f() -> f64 {
    floor(3.7)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 3.0).abs() < 1e-9, "floor(3.7) should be 3.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 5. ceil(3.2) == 4.0
// ---------------------------------------------------------------------------
#[test]
fn test_ceil_eval() {
    let src = r#"
def f() -> f64 {
    ceil(3.2)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 4.0).abs() < 1e-9, "ceil(3.2) should be 4.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 6. pow(2.0, 10.0) == 1024.0
// ---------------------------------------------------------------------------
#[test]
fn test_pow_eval() {
    let src = r#"
def f() -> f64 {
    pow(2.0, 10.0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 1024.0).abs() < 1e-6, "pow(2.0,10.0) should be 1024.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 7. min(3, 7) == 3  (integer)
// ---------------------------------------------------------------------------
#[test]
fn test_min_int_eval() {
    let src = r#"
def f() -> i64 {
    min(3, 7)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "3", "min(3,7) should be 3, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. max(3.5, 9.1) == 9.1
// ---------------------------------------------------------------------------
#[test]
fn test_max_float_eval() {
    let src = r#"
def f() -> f64 {
    max(3.5, 9.1)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 9.1).abs() < 1e-9, "max(3.5,9.1) should be 9.1, got: {}", v);
}
