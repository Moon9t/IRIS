//! Phase 23 integration tests: Closures (`|params| body`, `MakeClosure`, `CallClosure`).

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Closure IR emission: should contain make_closure
// ---------------------------------------------------------------------------
#[test]
fn test_closure_ir() {
    let src = r#"
def f() -> i64 {
    val double = |x: i64| x * 2
    double(5)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("make_closure"),
        "IR should contain make_closure, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 2. Closure eval: double(21) == 42
// ---------------------------------------------------------------------------
#[test]
fn test_closure_eval() {
    let src = r#"
def f() -> i64 {
    val double = |x: i64| x * 2
    double(21)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "double(21) should be 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. Closure capture: captures outer val, add_n(5) == 15
// ---------------------------------------------------------------------------
#[test]
fn test_closure_capture() {
    let src = r#"
def f() -> i64 {
    val n = 10
    val add_n = |x: i64| x + n
    add_n(5)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "15", "add_n(5) should be 15, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. Two-parameter closure: (3, 4) -> 7
// ---------------------------------------------------------------------------
#[test]
fn test_closure_two_params() {
    let src = r#"
def f() -> i64 {
    val add = |a: i64, b: i64| a + b
    add(3, 4)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "7", "add(3,4) should be 7, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. Closure with no captures (pure lambda)
// ---------------------------------------------------------------------------
#[test]
fn test_closure_no_capture() {
    let src = r#"
def f() -> i64 {
    val square = |x: i64| x * x
    square(9)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "81", "square(9) should be 81, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. Closure with if-else body
// ---------------------------------------------------------------------------
#[test]
fn test_closure_cond_body() {
    let src = r#"
def f() -> i64 {
    val abs_val = |x: i64| if x < 0 { 0 - x } else { x }
    abs_val(0 - 7)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "7", "abs(-7) should be 7, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. Closure stored as val, called later
// ---------------------------------------------------------------------------
#[test]
fn test_closure_as_val_type() {
    let src = r#"
def f() -> i64 {
    val sq = |x: i64| x * x
    sq(6)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "36", "sq(6) should be 36, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. Chained closures: f(g(3)) == 7
// ---------------------------------------------------------------------------
#[test]
fn test_closure_chained() {
    let src = r#"
def f() -> i64 {
    val g = |x: i64| x * 2
    val h = |x: i64| x + 1
    h(g(3))
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "7", "h(g(3)) should be 7, got: {}", out.trim());
}
