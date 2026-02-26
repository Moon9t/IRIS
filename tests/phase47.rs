//! Phase 47: Module System
//!
//! Tests for: `bring module_name`, `pub def`, multi-source compilation via `compile_multi`.

use iris::{compile_multi, EmitKind};

// ---------------------------------------------------------------------------
// Basic bring: import a pub function
// ---------------------------------------------------------------------------

#[test]
fn test_bring_basic() {
    let math_src = r#"
pub def add(a: i64, b: i64) -> i64 { a + b }
"#;
    let main_src = r#"
bring math
def f() -> i64 { add(3, 4) }
"#;
    let result = compile_multi(
        &[("math", math_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "7");
}

// ---------------------------------------------------------------------------
// Non-pub function from imported module is not visible (only pub ones are)
// ---------------------------------------------------------------------------

#[test]
fn test_bring_pub_only() {
    let util_src = r#"
pub def double(x: i64) -> i64 { x * 2 }
def private_fn(x: i64) -> i64 { x + 100 }
"#;
    let main_src = r#"
bring util
def f() -> i64 { double(21) }
"#;
    let result = compile_multi(
        &[("util", util_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// Imported function can call other functions in the same module
// ---------------------------------------------------------------------------

#[test]
fn test_bring_cross_call() {
    let math_src = r#"
pub def square(x: i64) -> i64 { x * x }
pub def sum_of_squares(a: i64, b: i64) -> i64 { square(a) + square(b) }
"#;
    let main_src = r#"
bring math
def f() -> i64 { sum_of_squares(3, 4) }
"#;
    let result = compile_multi(
        &[("math", math_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "25");
}

// ---------------------------------------------------------------------------
// Importing structs from another module
// ---------------------------------------------------------------------------

#[test]
fn test_bring_struct() {
    let types_src = r#"
record Vec2 { x: i64, y: i64 }
pub def dot(a: Vec2, b: Vec2) -> i64 { a.x * b.x + a.y * b.y }
"#;
    let main_src = r#"
bring types
def f() -> i64 {
    val a = Vec2 { x: 3, y: 4 }
    val b = Vec2 { x: 1, y: 2 }
    dot(a, b)
}
"#;
    let result = compile_multi(
        &[("types", types_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "11");
}

// ---------------------------------------------------------------------------
// Multiple brings
// ---------------------------------------------------------------------------

#[test]
fn test_bring_multiple() {
    let math_src = r#"
pub def add(a: i64, b: i64) -> i64 { a + b }
"#;
    let utils_src = r#"
pub def mul(a: i64, b: i64) -> i64 { a * b }
"#;
    let main_src = r#"
bring math
bring utils
def f() -> i64 { add(2, mul(3, 4)) }
"#;
    let result = compile_multi(
        &[("math", math_src), ("utils", utils_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "14");
}

// ---------------------------------------------------------------------------
// pub def visible in IR
// ---------------------------------------------------------------------------

#[test]
fn test_bring_ir_text() {
    let math_src = r#"
pub def helper(x: i64) -> i64 { x + 1 }
"#;
    let main_src = r#"
bring math
def f() -> i64 { helper(9) }
"#;
    let ir = compile_multi(
        &[("math", math_src), ("main", main_src)],
        "main",
        EmitKind::Ir,
    ).unwrap();
    assert!(ir.contains("helper"), "expected 'helper' in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// pub def visible in LLVM stub
// ---------------------------------------------------------------------------

#[test]
fn test_bring_llvm() {
    let math_src = r#"
pub def helper(x: i64) -> i64 { x + 1 }
"#;
    let main_src = r#"
bring math
def f() -> i64 { helper(9) }
"#;
    let ll = compile_multi(
        &[("math", math_src), ("main", main_src)],
        "main",
        EmitKind::Llvm,
    ).unwrap();
    assert!(ll.contains("helper"), "expected 'helper' in LLVM:\n{}", ll);
}

// ---------------------------------------------------------------------------
// Importing consts from another module
// ---------------------------------------------------------------------------

#[test]
fn test_bring_const() {
    let config_src = r#"
const MAX_VAL: i64 = 100
pub def get_max() -> i64 { MAX_VAL }
"#;
    let main_src = r#"
bring config
def f() -> i64 { get_max() }
"#;
    let result = compile_multi(
        &[("config", config_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "100");
}
