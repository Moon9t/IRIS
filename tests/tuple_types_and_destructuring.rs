//! Phase 125 integration tests: tuple types and destructuring.

use iris::{compile, EmitKind};

// ── 1. Basic tuple creation and access ──────────────────────────────────────
#[test]
fn test_tuple_basic() {
    let src = r#"
def f() -> i64 {
    val t = (10, 20, 30)
    t.0 + t.1 + t.2
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ── 2. Tuple returned from function ─────────────────────────────────────────
#[test]
fn test_tuple_return() {
    let src = r#"
def swap(a: i64, b: i64) -> (i64, i64) {
    (b, a)
}
def f() -> i64 {
    val t = swap(3, 7)
    t.0 * 10 + t.1
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "73");
}

// ── 3. Tuple with mixed types ───────────────────────────────────────────────
#[test]
fn test_tuple_mixed() {
    let src = r#"
def f() -> i64 {
    val t = (42, true)
    t.0
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 4. Tuple two-element ───────────────────────────────────────────────────
#[test]
fn test_tuple_pair() {
    let src = r#"
def f() -> i64 {
    val p = (100, 200)
    p.0 + p.1
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "300");
}

// ── 5. Tuple used in conditional ────────────────────────────────────────────
#[test]
fn test_tuple_conditional() {
    let src = r#"
def minmax(a: i64, b: i64) -> (i64, i64) {
    if a < b { (a, b) } else { (b, a) }
}
def f() -> i64 {
    val t = minmax(7, 3)
    t.0 * 10 + t.1
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "37");
}

// ── 6. Multiple tuples ─────────────────────────────────────────────────────
#[test]
fn test_multiple_tuples() {
    let src = r#"
def f() -> i64 {
    val a = (1, 2)
    val b = (3, 4)
    a.0 + a.1 + b.0 + b.1
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 7. Tuple as function parameter ──────────────────────────────────────────
#[test]
fn test_tuple_param() {
    let src = r#"
def sum_pair(p: (i64, i64)) -> i64 {
    p.0 + p.1
}
def f() -> i64 {
    sum_pair((15, 25))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "40");
}

// ── 8. Single element used from tuple ───────────────────────────────────────
#[test]
fn test_tuple_single_access() {
    let src = r#"
def f() -> i64 {
    val t = (99, 0)
    t.0
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "99");
}
