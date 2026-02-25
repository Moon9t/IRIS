//! Phase 30 integration tests: `grad<T>` — forward-mode automatic differentiation.
//!
//! grad(v) creates a dual number (value=v, tangent=1.0).
//! Arithmetic follows the chain rule automatically.
//! .value extracts the primal, .grad extracts the derivative.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. grad() produces a MakeGrad in IR
// ---------------------------------------------------------------------------
#[test]
fn test_grad_ir() {
    let src = r#"
def f() -> f64 {
    val x: grad<f64> = grad(3.0)
    x.value
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(out.contains("make_grad") || out.contains("grad"),
        "IR should contain grad instruction, got:\n{}", out);
}

// ---------------------------------------------------------------------------
// 2. grad(x).value == x
// ---------------------------------------------------------------------------
#[test]
fn test_grad_value_extraction() {
    let src = r#"
def f() -> f64 {
    val x: grad<f64> = grad(3.0)
    x.value
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "3", "grad(3.0).value should be 3, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. grad(x).grad == 1.0 (seed tangent)
// ---------------------------------------------------------------------------
#[test]
fn test_grad_tangent_seed() {
    let src = r#"
def f() -> f64 {
    val x: grad<f64> = grad(5.0)
    x.grad
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "1", "grad(5.0).grad should be 1 (seed), got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. grad<f64> addition: (x+y).value = x_val + y_val
// ---------------------------------------------------------------------------
#[test]
fn test_grad_add_value() {
    let src = r#"
def f() -> f64 {
    val x: grad<f64> = grad(3.0)
    val y: grad<f64> = grad(4.0)
    val z = x + y
    z.value
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "7", "grad(3)+grad(4) value = 7, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. grad<f64> addition: (x+y).grad = x.grad + y.grad = 2
// ---------------------------------------------------------------------------
#[test]
fn test_grad_add_tangent() {
    let src = r#"
def f() -> f64 {
    val x: grad<f64> = grad(3.0)
    val y: grad<f64> = grad(4.0)
    val z = x + y
    z.grad
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "2", "grad+grad tangent = 2, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. grad<f64> multiplication: (x*x).grad = 2*x (product rule)
// ---------------------------------------------------------------------------
#[test]
fn test_grad_mul_chain_rule() {
    let src = r#"
def f() -> f64 {
    val x: grad<f64> = grad(3.0)
    val y = x * x
    y.grad
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // product rule: d/dx(x*x) = x*1 + 1*x = 6 at x=3
    assert_eq!(out.trim(), "6", "d/dx(x^2) at x=3 = 6, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. grad<f64> multiplication: (x*x).value = x^2
// ---------------------------------------------------------------------------
#[test]
fn test_grad_mul_value() {
    let src = r#"
def f() -> f64 {
    val x: grad<f64> = grad(4.0)
    val y = x * x
    y.value
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "16", "grad(4)*grad(4) value = 16, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. grad subtraction: (x-y).grad = 1 - 1 = 0
// ---------------------------------------------------------------------------
#[test]
fn test_grad_sub_tangent() {
    let src = r#"
def f() -> f64 {
    val x: grad<f64> = grad(10.0)
    val y: grad<f64> = grad(3.0)
    val z = x - y
    z.grad
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "0", "d/dx(x-y)=1-1=0, got: {}", out.trim());
}
