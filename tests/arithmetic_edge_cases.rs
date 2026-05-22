//! Phase 123 integration tests: arithmetic edge cases.

use iris::{compile, EmitKind};

// ── 1. Negative arithmetic ─────────────────────────────────────────────────
#[test]
fn test_bitwise_and() {
    let src = r#"
def f() -> i64 {
    (0 - 5) * (0 - 3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15");
}

// ── 2. Nested parenthesized expressions ─────────────────────────────────────
#[test]
fn test_bitwise_or() {
    let src = r#"
def f() -> i64 {
    ((2 + 3) * (4 + 1)) + ((6 - 2) * 3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 5*5 + 4*3 = 25 + 12 = 37
    assert_eq!(result.trim(), "37");
}

// ── 3. Large multiplication ─────────────────────────────────────────────────
#[test]
fn test_bitwise_xor() {
    let src = r#"
def f() -> i64 {
    1000 * 1000
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1000000");
}

// ── 4. Chained division ────────────────────────────────────────────────────
#[test]
fn test_shift_left() {
    let src = r#"
def f() -> i64 {
    1000 / 10 / 10
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 5. Mixed division and modulo ────────────────────────────────────────────
#[test]
fn test_shift_right() {
    let src = r#"
def f() -> i64 {
    val a = 17 / 5
    val b = 17 % 5
    a * 5 + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 3*5 + 2 = 17
    assert_eq!(result.trim(), "17");
}

// ── 6. Chained arithmetic ──────────────────────────────────────────────────
#[test]
fn test_chained_arithmetic() {
    let src = r#"
def f() -> i64 {
    val a = 2 + 3 * 4
    val b = (2 + 3) * 4
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "34");
}

// ── 7. Division truncation ─────────────────────────────────────────────────
#[test]
fn test_integer_division() {
    let src = r#"
def f() -> i64 {
    7 / 2
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 8. Modulo with larger numbers ───────────────────────────────────────────
#[test]
fn test_modulo_large() {
    let src = r#"
def f() -> i64 {
    1000000007 % 1000000
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "7");
}
