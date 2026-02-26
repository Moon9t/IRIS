//! Phase 34 integration tests: bitwise operations.
//!
//! New builtins: band, bor, bxor, shl, shr, bitnot.
//! All operations work on i64 integers.
//! Constant expressions are folded by ConstFoldPass.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. band(0b1010, 0b1100) == 0b1000 == 8
// ---------------------------------------------------------------------------
#[test]
fn test_band_eval() {
    let src = r#"
def f() -> i64 {
    band(10, 12)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "8", "10 & 12 = 8, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 2. bor(0b1010, 0b0101) == 0b1111 == 15
// ---------------------------------------------------------------------------
#[test]
fn test_bor_eval() {
    let src = r#"
def f() -> i64 {
    bor(10, 5)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "15", "10 | 5 = 15, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. bxor(0b1111, 0b0101) == 0b1010 == 10
// ---------------------------------------------------------------------------
#[test]
fn test_bxor_eval() {
    let src = r#"
def f() -> i64 {
    bxor(15, 5)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "10", "15 ^ 5 = 10, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. shl(1, 4) == 16  (left shift by 4)
// ---------------------------------------------------------------------------
#[test]
fn test_shl_eval() {
    let src = r#"
def f() -> i64 {
    shl(1, 4)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "16", "1 << 4 = 16, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. shr(32, 2) == 8  (right shift by 2)
// ---------------------------------------------------------------------------
#[test]
fn test_shr_eval() {
    let src = r#"
def f() -> i64 {
    shr(32, 2)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "8", "32 >> 2 = 8, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. bitnot(0) == -1  (all bits set in two's complement)
// ---------------------------------------------------------------------------
#[test]
fn test_bitnot_zero() {
    let src = r#"
def f() -> i64 {
    bitnot(0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "-1", "bitnot(0) = -1, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. bitnot(-1) == 0  (flip all bits)
// ---------------------------------------------------------------------------
#[test]
fn test_bitnot_minus_one() {
    let src = r#"
def f() -> i64 {
    bitnot(-1)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "0", "bitnot(-1) = 0, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. Bitwise ops produce correct IR opcodes
// ---------------------------------------------------------------------------
#[test]
fn test_bitwise_ir_opcodes() {
    let src = r#"
def f(a: i64, b: i64) -> i64 {
    band(a, b)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(ir.contains("band") || ir.contains("and"), "IR should contain band/and, got:\n{}", ir);
}
