//! Phase 64 integration tests: @differentiable attribute + grad_of numerical differentiation.
//! Float literals default to f32 in IRIS, so closures use f32 params.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. @differentiable attribute is parsed without error
// ---------------------------------------------------------------------------
#[test]
fn test_differentiable_attr_parsed() {
    let src = r#"
@differentiable
def square(x: f32) -> f32 { x * x }
def f() -> i64 { square(3.0) to i64 }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "9");
}

// ---------------------------------------------------------------------------
// 2. grad_of x^2 at x=3.0 => derivative = 6.0
// ---------------------------------------------------------------------------
#[test]
fn test_grad_of_square_at_3() {
    let src = r#"
def f() -> i64 {
    val g = grad_of(|x: f32| x * x, 3.0)
    round(g) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ---------------------------------------------------------------------------
// 3. grad_of 2*x at x=1.0 => derivative = 2.0
// ---------------------------------------------------------------------------
#[test]
fn test_grad_of_linear() {
    let src = r#"
def f() -> i64 {
    val g = grad_of(|x: f32| 2.0 * x, 1.0)
    round(g) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// 4. grad_of constant => derivative = 0.0
// ---------------------------------------------------------------------------
#[test]
fn test_grad_of_constant() {
    let src = r#"
def f() -> i64 {
    val g = grad_of(|x: f32| 42.0, 5.0)
    round(g) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ---------------------------------------------------------------------------
// 5. grad_of x^3 at x=2.0 => derivative = 3*x^2 = 12.0
// ---------------------------------------------------------------------------
#[test]
fn test_grad_of_cube_at_2() {
    let src = r#"
def f() -> i64 {
    val g = grad_of(|x: f32| x * x * x, 2.0)
    round(g) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "12");
}

// ---------------------------------------------------------------------------
// 6. grad_of x + 5.0 at x=0.0 => derivative = 1.0
// ---------------------------------------------------------------------------
#[test]
fn test_grad_of_affine() {
    let src = r#"
def f() -> i64 {
    val g = grad_of(|x: f32| x + 5.0, 0.0)
    round(g) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 7. Multiple grad_of calls in same function
// ---------------------------------------------------------------------------
#[test]
fn test_grad_of_multiple() {
    let src = r#"
def f() -> i64 {
    val g1 = grad_of(|x: f32| x * x, 2.0)
    val g2 = grad_of(|x: f32| x * x, 5.0)
    round(g1) to i64 + round(g2) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // g1 = 4.0, g2 = 10.0 => 4 + 10 = 14
    assert_eq!(result.trim(), "14");
}

// ---------------------------------------------------------------------------
// 8. @differentiable on a named function, then grad_of via wrapper closure
// ---------------------------------------------------------------------------
#[test]
fn test_differentiable_named_fn_grad() {
    let src = r#"
@differentiable
def cubic(x: f32) -> f32 { x * x * x }
def f() -> i64 {
    val g = grad_of(|x: f32| cubic(x), 3.0)
    round(g) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // d/dx x^3 at x=3 = 3*9 = 27
    assert_eq!(result.trim(), "27");
}
