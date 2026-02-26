//! Phase 36 integration tests: extended math builtins —
//! sin, cos, tan, exp, log, log2, round, sign, clamp.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. sin(0.0) == 0.0
// ---------------------------------------------------------------------------
#[test]
fn test_sin_zero() {
    let src = r#"
def f() -> f64 {
    sin(0.0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 0.0).abs() < 1e-9, "sin(0.0) should be 0.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 2. cos(0.0) == 1.0
// ---------------------------------------------------------------------------
#[test]
fn test_cos_zero() {
    let src = r#"
def f() -> f64 {
    cos(0.0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 1.0).abs() < 1e-9, "cos(0.0) should be 1.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 3. exp(0.0) == 1.0
// ---------------------------------------------------------------------------
#[test]
fn test_exp_zero() {
    let src = r#"
def f() -> f64 {
    exp(0.0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 1.0).abs() < 1e-9, "exp(0.0) should be 1.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 4. log(1.0) == 0.0
// ---------------------------------------------------------------------------
#[test]
fn test_log_one() {
    let src = r#"
def f() -> f64 {
    log(1.0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 0.0).abs() < 1e-9, "log(1.0) should be 0.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 5. log2(8.0) == 3.0
// ---------------------------------------------------------------------------
#[test]
fn test_log2() {
    let src = r#"
def f() -> f64 {
    log2(8.0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 3.0).abs() < 1e-9, "log2(8.0) should be 3.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 6. round(2.7) == 3.0
// ---------------------------------------------------------------------------
#[test]
fn test_round() {
    let src = r#"
def f() -> f64 {
    round(2.7)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 3.0).abs() < 1e-9, "round(2.7) should be 3.0, got: {}", v);
}

// ---------------------------------------------------------------------------
// 7. sign(-5) == -1
// ---------------------------------------------------------------------------
#[test]
fn test_sign_negative() {
    let src = r#"
def f() -> i64 {
    sign(-5)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "-1", "sign(-5) should be -1, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. clamp(15, 0, 10) == 10
// ---------------------------------------------------------------------------
#[test]
fn test_clamp() {
    let src = r#"
def f() -> i64 {
    clamp(15, 0, 10)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "10", "clamp(15,0,10) should be 10, got: {}", out.trim());
}
