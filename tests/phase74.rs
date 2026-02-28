//! Phase 74 integration tests: strength reduction optimizer.
//!
//! Verifies that:
//! - `x * 2^n` computes the correct result (optimized to shift)
//! - `x / 2^n` computes the correct result (optimized to shift)
//! - `x - x` computes 0
//! - Non-power-of-2 mul/div still works correctly

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Multiply by 2 (2^1)
// ---------------------------------------------------------------------------
#[test]
fn test_mul_by_2() {
    let src = r#"
def f() -> i64 {
    val x = 21
    x * 2
}
"#;
    let out = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(out.trim(), "42");
}

// ---------------------------------------------------------------------------
// 2. Multiply by 4 (2^2)
// ---------------------------------------------------------------------------
#[test]
fn test_mul_by_4() {
    let src = r#"
def f() -> i64 {
    val x = 10
    x * 4
}
"#;
    let out = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(out.trim(), "40");
}

// ---------------------------------------------------------------------------
// 3. Multiply by 8 (2^3)
// ---------------------------------------------------------------------------
#[test]
fn test_mul_by_8() {
    let src = r#"
def f() -> i64 {
    val x = 5
    x * 8
}
"#;
    let out = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(out.trim(), "40");
}

// ---------------------------------------------------------------------------
// 4. Multiply by 16 (2^4)
// ---------------------------------------------------------------------------
#[test]
fn test_mul_by_16() {
    let src = r#"
def f() -> i64 {
    val x = 3
    x * 16
}
"#;
    let out = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(out.trim(), "48");
}

// ---------------------------------------------------------------------------
// 5. Divide by 2 (2^1)
// ---------------------------------------------------------------------------
#[test]
fn test_div_by_2() {
    let src = r#"
def f() -> i64 {
    val x = 84
    x / 2
}
"#;
    let out = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(out.trim(), "42");
}

// ---------------------------------------------------------------------------
// 6. Divide by 4 (2^2)
// ---------------------------------------------------------------------------
#[test]
fn test_div_by_4() {
    let src = r#"
def f() -> i64 {
    val x = 168
    x / 4
}
"#;
    let out = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(out.trim(), "42");
}

// ---------------------------------------------------------------------------
// 7. Subtraction of identical values → 0
// ---------------------------------------------------------------------------
#[test]
fn test_sub_self_is_zero() {
    let src = r#"
def f() -> i64 {
    val x = 999
    x - x
}
"#;
    let out = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(out.trim(), "0");
}

// ---------------------------------------------------------------------------
// 8. Non-power-of-2 multiply still correct
// ---------------------------------------------------------------------------
#[test]
fn test_mul_non_power_of_two() {
    let src = r#"
def f() -> i64 {
    val x = 6
    x * 7
}
"#;
    let out = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(out.trim(), "42");
}
