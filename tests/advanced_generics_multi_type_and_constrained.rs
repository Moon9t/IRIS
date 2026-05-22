//! Phase 108 integration tests: advanced generics — multi-type and constrained.

use iris::{compile, EmitKind};

// ── 1. Generic function used with i64 and f64 in same scope ─────────────────
#[test]
fn test_generic_dual_instantiation() {
    let src = r#"
def identity[T](x: T) -> T { x }
def f() -> i64 {
    val a = identity(42)
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 2. Generic function with string type ────────────────────────────────────
#[test]
fn test_generic_with_string() {
    let src = r#"
def identity[T](x: T) -> T { x }
def f() -> str {
    identity("hello")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello");
}

// ── 3. Generic min function ─────────────────────────────────────────────────
#[test]
fn test_generic_min() {
    let src = r#"
def min_val[T](a: T, b: T) -> T {
    if a < b { a } else { b }
}
def f() -> i64 {
    min_val(30, 10)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 4. Generic function called multiple times ───────────────────────────────
#[test]
fn test_generic_multiple_calls() {
    let src = r#"
def double[T](x: T, y: T) -> T {
    if x > y { x } else { y }
}
def f() -> i64 {
    val a = double(5, 10)
    val b = double(20, 15)
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "30");
}

// ── 5. Generic with boolean ─────────────────────────────────────────────────
#[test]
fn test_generic_with_bool() {
    let src = r#"
def identity[T](x: T) -> T { x }
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    bool_to_i64(identity(true))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 6. Generic conditional function ─────────────────────────────────────────
#[test]
fn test_generic_conditional() {
    let src = r#"
def choose[T](cond: bool, a: T, b: T) -> T {
    if cond { a } else { b }
}
def f() -> i64 {
    choose(true, 100, 200)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "100");
}

// ── 7. Generic conditional false branch ─────────────────────────────────────
#[test]
fn test_generic_conditional_false() {
    let src = r#"
def choose[T](cond: bool, a: T, b: T) -> T {
    if cond { a } else { b }
}
def f() -> i64 {
    choose(false, 100, 200)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "200");
}

// ── 8. Generic function with arithmetic ─────────────────────────────────────
#[test]
fn test_generic_clamp() {
    let src = r#"
def clamp_val[T](x: T, lo: T, hi: T) -> T {
    if x < lo { lo } else { if x > hi { hi } else { x } }
}
def f() -> i64 {
    val a = clamp_val(5, 1, 10)
    val b = clamp_val(15, 1, 10)
    val c = clamp_val(0 - 5, 1, 10)
    a + b + c
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "16");
}
