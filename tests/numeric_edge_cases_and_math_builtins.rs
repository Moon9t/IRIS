//! Phase 113 integration tests: numeric edge cases and math builtins.

use iris::{compile, EmitKind};

// ── 1. Integer overflow wrapping behavior ───────────────────────────────────
#[test]
fn test_large_integer() {
    let src = r#"
def f() -> i64 {
    val a = 1000000
    val b = 1000000
    a * b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1000000000000");
}

// ── 2. Negative integer arithmetic ──────────────────────────────────────────
#[test]
fn test_negative_arithmetic() {
    let src = r#"
def f() -> i64 {
    val a = 0 - 10
    val b = 0 - 20
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "-30");
}

// ── 3. Float precision ─────────────────────────────────────────────────────
#[test]
fn test_float_precision() {
    let src = r#"
def f() -> f64 {
    val a = 0.1
    val b = 0.2
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!((v - 0.3).abs() < 0.001);
}

// ── 4. Integer modulo ───────────────────────────────────────────────────────
#[test]
fn test_integer_modulo() {
    let src = r#"
def f() -> i64 {
    17 % 5
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 5. Math sqrt ────────────────────────────────────────────────────────────
#[test]
fn test_math_sqrt() {
    let src = r#"
def f() -> f64 {
    sqrt(16.0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!((v - 4.0).abs() < 0.001);
}

// ── 6. Math abs ─────────────────────────────────────────────────────────────
#[test]
fn test_math_abs() {
    let src = r#"
def f() -> f64 {
    abs(0.0 - 42.5)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!((v - 42.5).abs() < 0.001);
}

// ── 7. Math min and max ─────────────────────────────────────────────────────
#[test]
fn test_math_min_max() {
    let src = r#"
def f() -> f64 {
    min(3.0, 7.0) + max(3.0, 7.0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!((v - 10.0).abs() < 0.001);
}

// ── 8. Math pow ─────────────────────────────────────────────────────────────
#[test]
fn test_math_pow() {
    let src = r#"
def f() -> f64 {
    pow(2.0, 10.0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!((v - 1024.0).abs() < 0.001);
}
