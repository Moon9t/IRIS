//! Phase 45: Generic functions (parametric polymorphism)
//!
//! Tests for: `def f[T](...)` syntax — monomorphization at call sites.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// Generic identity function with i64
// ---------------------------------------------------------------------------

#[test]
fn test_generic_identity_i64() {
    let src = r#"
def identity[T](x: T) -> T { x }

def f() -> i64 {
    identity(42)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// Generic identity function with f64
// ---------------------------------------------------------------------------

#[test]
fn test_generic_identity_f64() {
    let src = r#"
def identity[T](x: T) -> T { x }

def f() -> f64 {
    identity(3.14)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert!(result.trim().starts_with("3.14"), "got: {}", result.trim());
}

// ---------------------------------------------------------------------------
// Generic max function with i64
// ---------------------------------------------------------------------------

#[test]
fn test_generic_max_i64() {
    let src = r#"
def max_val[T](a: T, b: T) -> T {
    if a > b { a } else { b }
}

def f() -> i64 {
    max_val(10, 20)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "20");
}

// ---------------------------------------------------------------------------
// Generic max function with f64
// ---------------------------------------------------------------------------

#[test]
fn test_generic_max_f64() {
    let src = r#"
def max_val[T](a: T, b: T) -> T {
    if a > b { a } else { b }
}

def f() -> f64 {
    max_val(1.5, 2.5)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert!(result.trim().starts_with("2.5"), "got: {}", result.trim());
}

// ---------------------------------------------------------------------------
// Same generic called twice with different types → two specializations
// ---------------------------------------------------------------------------

#[test]
fn test_generic_two_specializations() {
    let src = r#"
def identity[T](x: T) -> T { x }

def f() -> i64 {
    val a = identity(100)
    val b = identity(200)
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "300");
}

// ---------------------------------------------------------------------------
// Generic function with bool
// ---------------------------------------------------------------------------

#[test]
fn test_generic_identity_bool() {
    let src = r#"
def identity[T](x: T) -> T { x }

def f() -> bool {
    identity(true)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "true");
}

// ---------------------------------------------------------------------------
// IR text contains mangled name
// ---------------------------------------------------------------------------

#[test]
fn test_generic_ir_text() {
    let src = r#"
def identity[T](x: T) -> T { x }

def f() -> i64 {
    identity(7)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("identity__"), "expected mangled name in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// LLVM stub contains mangled name
// ---------------------------------------------------------------------------

#[test]
fn test_generic_llvm() {
    let src = r#"
def identity[T](x: T) -> T { x }

def f() -> i64 {
    identity(7)
}
"#;
    let ll = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(ll.contains("identity__"), "expected mangled name in LLVM:\n{}", ll);
}
