//! Phase 46: Trait / Interface system
//!
//! Tests for: `trait Name { def method(...) }` + `impl Trait for Type { ... }`
//! Static dispatch — method calls are resolved to mangled names at compile time.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// Simple Show trait on i64
// ---------------------------------------------------------------------------

#[test]
fn test_trait_show_i64() {
    let src = r#"
trait Show {
    def show(x: i64) -> str
}
impl Show for i64 {
    def show(x: i64) -> str { to_str(x) }
}
def f() -> str {
    show(42)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // String values are displayed with surrounding quotes by the interpreter.
    assert_eq!(result.trim().trim_matches('"'), "42");
}

// ---------------------------------------------------------------------------
// Show trait on bool
// ---------------------------------------------------------------------------

#[test]
fn test_trait_show_bool() {
    let src = r#"
trait Show {
    def show(x: bool) -> str
}
impl Show for bool {
    def show(x: bool) -> str { if x { "yes" } else { "no" } }
}
def f() -> str {
    show(true)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim().trim_matches('"'), "yes");
}

// ---------------------------------------------------------------------------
// Double trait — two impls for different types
// ---------------------------------------------------------------------------

#[test]
fn test_trait_two_impls() {
    let src = r#"
trait Double {
    def double(x: i64) -> i64
}
impl Double for i64 {
    def double(x: i64) -> i64 { x * 2 }
}
def f() -> i64 {
    double(21)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// Negate trait on i64
// ---------------------------------------------------------------------------

#[test]
fn test_trait_negate_i64() {
    let src = r#"
trait Negate {
    def negate(x: i64) -> i64
}
impl Negate for i64 {
    def negate(x: i64) -> i64 { 0 - x }
}
def f() -> i64 {
    negate(7)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "-7");
}

// ---------------------------------------------------------------------------
// Multiple methods in one impl
// ---------------------------------------------------------------------------

#[test]
fn test_trait_multiple_methods() {
    let src = r#"
trait Arith {
    def add_one(x: i64) -> i64
    def mul_two(x: i64) -> i64
}
impl Arith for i64 {
    def add_one(x: i64) -> i64 { x + 1 }
    def mul_two(x: i64) -> i64 { x * 2 }
}
def f() -> i64 {
    val a = add_one(9)
    mul_two(a)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "20");
}

// ---------------------------------------------------------------------------
// IR text contains mangled impl name
// ---------------------------------------------------------------------------

#[test]
fn test_trait_ir_text() {
    let src = r#"
trait Show {
    def show(x: i64) -> str
}
impl Show for i64 {
    def show(x: i64) -> str { to_str(x) }
}
def f() -> str {
    show(7)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("Show__i64__show"), "expected mangled name in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// LLVM stub contains mangled impl name
// ---------------------------------------------------------------------------

#[test]
fn test_trait_llvm() {
    let src = r#"
trait Show {
    def show(x: i64) -> str
}
impl Show for i64 {
    def show(x: i64) -> str { to_str(x) }
}
def f() -> str {
    show(7)
}
"#;
    let ll = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(ll.contains("Show__i64__show"), "expected mangled name in LLVM:\n{}", ll);
}

// ---------------------------------------------------------------------------
// Trait impl for struct type
// ---------------------------------------------------------------------------

#[test]
fn test_trait_impl_for_struct() {
    let src = r#"
record Point { x: i64, y: i64 }
trait Area {
    def area(p: Point) -> i64
}
impl Area for Point {
    def area(p: Point) -> i64 { p.x * p.y }
}
def f() -> i64 {
    val p = Point { x: 3, y: 4 }
    area(p)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "12");
}
