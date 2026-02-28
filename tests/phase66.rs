//! Phase 66 integration tests: tuple destructuring + tuple when-patterns.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Basic val (a, b) = tuple destructuring
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_destructure_pair() {
    let src = r#"
def f() -> i64 {
    val (a, b) = (10, 20)
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "30");
}

// ---------------------------------------------------------------------------
// 2. Three-element destructuring
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_destructure_triple() {
    let src = r#"
def f() -> i64 {
    val (x, y, z) = (100, 200, 300)
    x + y + z
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "600");
}

// ---------------------------------------------------------------------------
// 3. Destructure from a function return
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_destructure_from_fn() {
    let src = r#"
def make_pair(dummy: i64) -> (i64, i64) { (7, 8) }
def f() -> i64 {
    val (a, b) = make_pair(0)
    a * b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "56");
}

// ---------------------------------------------------------------------------
// 4. Tuple when-pattern: bind both elements
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_when_bind_both() {
    let src = r#"
def f() -> i64 {
    val t = (3, 4)
    when t {
        (a, b) => a + b,
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "7");
}

// ---------------------------------------------------------------------------
// 5. Tuple when-pattern with guard
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_when_with_guard() {
    let src = r#"
def f() -> i64 {
    val t = (5, 3)
    when t {
        (a, b) if a > b => 1,
        (a, b) => 0,
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 6. Tuple when-pattern with wildcard sub-pattern
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_when_wildcard_sub() {
    let src = r#"
def f() -> i64 {
    val t = (42, 0)
    when t {
        (x, _) => x,
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// 7. Multiple tuple when arms
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_when_multiple_arms() {
    let src = r#"
def classify(t: (i64, i64)) -> i64 {
    when t {
        (a, b) if a > b => 1,
        (a, b) if a < b => -1,
        (a, b) => 0,
    }
}
def f() -> i64 {
    classify((5, 3)) + classify((1, 9)) + classify((4, 4))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ---------------------------------------------------------------------------
// 8. Swap values using tuple destructuring
// ---------------------------------------------------------------------------
#[test]
fn test_tuple_swap() {
    let src = r#"
def f() -> i64 {
    val x = 10
    val y = 20
    val (a, b) = (y, x)
    a - b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}
