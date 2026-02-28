//! Phase 72 integration tests: inclusive range patterns in `when` arms.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Basic inclusive range match
// ---------------------------------------------------------------------------
#[test]
fn test_range_basic() {
    let src = r#"
def grade(n: i64) -> i64 {
    when n {
        90..=100 => 4,
        80..=89  => 3,
        70..=79  => 2,
        _        => 1,
    }
}
def f() -> i64 { grade(85) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ---------------------------------------------------------------------------
// 2. Low range boundary (exactly lo)
// ---------------------------------------------------------------------------
#[test]
fn test_range_boundary_lo() {
    let src = r#"
def classify(n: i64) -> i64 {
    when n {
        1..=5 => 10,
        _ => 0,
    }
}
def f() -> i64 { classify(1) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ---------------------------------------------------------------------------
// 3. High range boundary (exactly hi)
// ---------------------------------------------------------------------------
#[test]
fn test_range_boundary_hi() {
    let src = r#"
def classify(n: i64) -> i64 {
    when n {
        1..=5 => 10,
        _ => 0,
    }
}
def f() -> i64 { classify(5) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ---------------------------------------------------------------------------
// 4. Out of range falls to wildcard
// ---------------------------------------------------------------------------
#[test]
fn test_range_miss() {
    let src = r#"
def classify(n: i64) -> i64 {
    when n {
        1..=5 => 10,
        _ => 99,
    }
}
def f() -> i64 { classify(6) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "99");
}

// ---------------------------------------------------------------------------
// 5. Multiple non-overlapping ranges
// ---------------------------------------------------------------------------
#[test]
fn test_range_multiple() {
    let src = r#"
def bucket(n: i64) -> i64 {
    when n {
        0..=9   => 1,
        10..=99  => 2,
        100..=999 => 3,
        _ => 0,
    }
}
def f() -> i64 { bucket(5) + bucket(50) + bucket(500) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ---------------------------------------------------------------------------
// 6. Range mixed with literal pattern
// ---------------------------------------------------------------------------
#[test]
fn test_range_mixed_with_literal() {
    let src = r#"
def describe(n: i64) -> i64 {
    when n {
        0     => 0,
        1..=9 => 1,
        _     => 2,
    }
}
def f() -> i64 { describe(0) + describe(5) + describe(10) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ---------------------------------------------------------------------------
// 7. Single-element range (lo == hi)
// ---------------------------------------------------------------------------
#[test]
fn test_range_single() {
    let src = r#"
def is_seven(n: i64) -> i64 {
    when n {
        7..=7 => 1,
        _ => 0,
    }
}
def f() -> i64 { is_seven(7) + is_seven(8) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 8. Range with negative bounds
// ---------------------------------------------------------------------------
#[test]
fn test_range_negative() {
    let src = r#"
def sign_bucket(n: i64) -> i64 {
    when n {
        0..=100  => 1,
        _ => 0 - 1,
    }
}
def f() -> i64 { sign_bucket(50) + sign_bucket(101) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}
