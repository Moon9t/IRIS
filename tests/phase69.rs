//! Phase 69 integration tests: default parameter values.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Simple default parameter
// ---------------------------------------------------------------------------
#[test]
fn test_default_param_simple() {
    let src = r#"
def greet(x: i64, y: i64 = 10) -> i64 { x + y }
def f() -> i64 { greet(5) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15");
}

// ---------------------------------------------------------------------------
// 2. Override the default
// ---------------------------------------------------------------------------
#[test]
fn test_default_param_override() {
    let src = r#"
def add(x: i64, y: i64 = 100) -> i64 { x + y }
def f() -> i64 { add(3, 7) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ---------------------------------------------------------------------------
// 3. Multiple default params
// ---------------------------------------------------------------------------
#[test]
fn test_default_param_multiple() {
    let src = r#"
def make(a: i64, b: i64 = 2, c: i64 = 3) -> i64 { a * b * c }
def f() -> i64 { make(4) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "24");
}

// ---------------------------------------------------------------------------
// 4. Partially override defaults
// ---------------------------------------------------------------------------
#[test]
fn test_default_param_partial_override() {
    let src = r#"
def make(a: i64, b: i64 = 2, c: i64 = 3) -> i64 { a * b * c }
def f() -> i64 { make(4, 5) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ---------------------------------------------------------------------------
// 5. Default bool parameter
// ---------------------------------------------------------------------------
#[test]
fn test_default_param_bool() {
    let src = r#"
def toggle(x: i64, flip: bool = false) -> i64 {
    when flip {
        true  => 0 - x,
        false => x,
    }
}
def f() -> i64 { toggle(42) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// 6. Default param in recursive function
// ---------------------------------------------------------------------------
#[test]
fn test_default_param_recursive() {
    let src = r#"
def countdown(n: i64, acc: i64 = 0) -> i64 {
    when n <= 0 {
        true  => acc,
        false => countdown(n - 1, acc + n),
    }
}
def f() -> i64 { countdown(5) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15");
}

// ---------------------------------------------------------------------------
// 7. Default as expression (arithmetic)
// ---------------------------------------------------------------------------
#[test]
fn test_default_param_expr() {
    let src = r#"
def scale(x: i64, factor: i64 = 2 * 3) -> i64 { x * factor }
def f() -> i64 { scale(7) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// 8. All defaults supplied explicitly
// ---------------------------------------------------------------------------
#[test]
fn test_default_param_all_supplied() {
    let src = r#"
def clamp_val(x: i64, lo: i64 = 0, hi: i64 = 100) -> i64 {
    when x < lo {
        true => lo,
        false => when x > hi { true => hi, false => x },
    }
}
def f() -> i64 { clamp_val(150, 0, 100) + clamp_val(50, 0, 100) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "150");
}
