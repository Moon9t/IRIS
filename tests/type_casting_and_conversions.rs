//! Phase 107 integration tests: type casting and conversions.

use iris::{compile, EmitKind};

// ── 1. to_str converts integer to string ────────────────────────────────────
#[test]
fn test_to_str_integer() {
    let src = r#"
def f() -> str {
    to_str(42)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 2. to_str converts float to string ──────────────────────────────────────
#[test]
fn test_to_str_float() {
    let src = r#"
def f() -> i64 {
    val s = to_str(3.14)
    len(s)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    let n: i64 = result.trim().parse().unwrap();
    assert!(
        n >= 3,
        "to_str(3.14) should have at least 3 chars, got {}",
        n
    );
}

// ── 3. to_str converts bool to string ───────────────────────────────────────
#[test]
fn test_to_str_bool() {
    let src = r#"
def f() -> str {
    to_str(true)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "true");
}

// ── 4. parse_i64 parses valid integer string ────────────────────────────────
#[test]
fn test_parse_i64_valid() {
    let src = r#"
def f() -> i64 {
    unwrap(parse_i64("123"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "123");
}

// ── 5. parse_f64 parses valid float string ──────────────────────────────────
#[test]
fn test_parse_f64_valid() {
    let src = r#"
def f() -> f64 {
    unwrap(parse_f64("2.718"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!((v - 2.718).abs() < 0.001);
}

// ── 6. parse_i64 with negative number ───────────────────────────────────────
#[test]
fn test_parse_i64_negative() {
    let src = r#"
def f() -> i64 {
    unwrap(parse_i64("-50"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "-50");
}

// ── 7. to_str then parse_i64 round-trip ─────────────────────────────────────
#[test]
fn test_to_str_parse_roundtrip() {
    let src = r#"
def f() -> i64 {
    val s = to_str(999)
    unwrap(parse_i64(s))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "999");
}

// ── 8. to_str with zero ─────────────────────────────────────────────────────
#[test]
fn test_to_str_zero() {
    let src = r#"
def f() -> str {
    to_str(0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}
