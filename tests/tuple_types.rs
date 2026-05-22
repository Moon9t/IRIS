//! Phase 19 integration tests: Tuple types.
//!
//! Tuples are first-class ordered collections of heterogeneous values.
//! - Tuple literal:  `(a, b, c)`
//! - Index access:   `t.0`, `t.1`
//! - Destructuring:  `val (a, b) = t`
//! - Trailing comma: `(a,)` is a 1-element tuple.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Tuple literal is accepted by the parser → produces valid IR
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_literal_ir() {
    let src = r#"
def f() -> i64 {
    val t = (1, 2, 3)
    t.0
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(
        result.is_ok(),
        "tuple literal should compile: {:?}",
        result.err()
    );
    let out = result.unwrap();
    assert!(
        out.contains("make_tuple"),
        "IR should contain make_tuple: {}",
        out
    );
}

// ---------------------------------------------------------------------------
// 2. Tuple element access by index (t.0)
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_index_access_eval() {
    let src = r#"
def first() -> i64 {
    val t = (10, 20, 30)
    t.0
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "10", "t.0 should be 10, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. Access second and third elements
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_second_third_eval() {
    let src = r#"
def add_23() -> i64 {
    val t = (10, 20, 30)
    t.1 + t.2
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "50",
        "t.1+t.2 should be 50, got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 4. Tuple destructuring with `val (a, b) = expr`
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_destructuring_eval() {
    let src = r#"
def destruct() -> i64 {
    val pair = (3, 7)
    val (a, b) = pair
    a + b
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "10", "3+7 should be 10, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. Mixed-type tuple (i64 and bool)
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_mixed_types_eval() {
    let src = r#"
def mixed() -> i64 {
    val t = (42, true)
    t.0
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "42",
        "mixed tuple first elem should be 42, got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 6. Nested tuple: tuple of tuples
// ---------------------------------------------------------------------------
#[test]
fn test_nested_tuple_eval() {
    let src = r#"
def nested() -> i64 {
    val inner = (5, 6)
    val outer = (inner, 100)
    outer.1
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "100",
        "outer.1 should be 100, got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 7. Tuple returned from function
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_from_function_eval() {
    let src = r#"
def use_pair() -> i64 {
    val p = make_pair()
    val (a, b) = p
    a + b
}

def make_pair() -> (i64, i64) {
    (11, 22)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "33", "11+22 should be 33, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. Tuple with trailing comma (single-element)
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_trailing_comma_ir() {
    let src = r#"
def single() -> i64 {
    val t = (99,)
    t.0
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "99",
        "single-element tuple should work, got: {}",
        out.trim()
    );
}
