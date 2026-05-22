//! Phase 105 integration tests: multi-variable closure captures and nested closures.

use iris::{compile, EmitKind};

// ── 1. Closure capturing two variables ──────────────────────────────────────
#[test]
fn test_closure_two_captures() {
    let src = r#"
def f() -> i64 {
    val a = 10
    val b = 20
    val sum_ab = |x: i64| x + a + b
    sum_ab(5)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "35");
}

// ── 2. Closure capturing three variables ────────────────────────────────────
#[test]
fn test_closure_three_captures() {
    let src = r#"
def f() -> i64 {
    val a = 1
    val b = 2
    val c = 3
    val total = |x: i64| x + a + b + c
    total(4)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 3. Two closures capturing overlapping variables ─────────────────────────
#[test]
fn test_two_closures_overlapping_captures() {
    let src = r#"
def f() -> i64 {
    val a = 10
    val b = 20
    val add_a = |x: i64| x + a
    val add_b = |x: i64| x + b
    add_a(5) + add_b(5)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "40");
}

// ── 4. Closure capturing a string variable ──────────────────────────────────
#[test]
fn test_closure_capture_string() {
    let src = r#"
def f() -> str {
    val prefix = "Hello"
    val greet = |name: str| concat(concat(prefix, " "), name)
    greet("World")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Hello World");
}

// ── 5. Closure capturing a boolean variable ─────────────────────────────────
#[test]
fn test_closure_capture_bool() {
    let src = r#"
def f() -> i64 {
    val flag = true
    val choose = |a: i64, b: i64| if flag { a } else { b }
    choose(42, 0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 6. Closure used as argument to user-defined HOF ─────────────────────────
#[test]
fn test_closure_as_hof_arg() {
    let src = r#"
def apply(func: (i64) -> i64, x: i64) -> i64 {
    func(x)
}
def f() -> i64 {
    val triple = |x: i64| x * 3
    apply(triple, 7)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "21");
}

// ── 7. Closure capturing function parameter ────────────────────────────────
#[test]
fn test_closure_captures_param() {
    let src = r#"
def apply_twice(func: (i64) -> i64, x: i64) -> i64 {
    func(func(x))
}
def f() -> i64 {
    val inc = |x: i64| x + 1
    apply_twice(inc, 10)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "12");
}

// ── 8. Closure as HOF with capture ──────────────────────────────────────────
#[test]
fn test_closure_hof_with_capture() {
    let src = r#"
def apply(func: (i64) -> i64, x: i64) -> i64 { func(x) }
def f() -> i64 {
    val base = 100
    val add_base = |x: i64| x + base
    apply(add_base, 50)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "150");
}
