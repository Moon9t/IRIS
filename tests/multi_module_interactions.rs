//! Phase 116 integration tests: multi-module interactions.

use iris::{compile_multi, EmitKind};

// ── 1. Basic multi-module import ────────────────────────────────────────────
#[test]
fn test_multi_module_basic() {
    let lib = r#"pub def add(a: i64, b: i64) -> i64 { a + b }"#;
    let main = r#"
bring "lib.iris"
def f() -> i64 {
    add(10, 20)
}
"#;
    let result = compile_multi(&[("lib", lib), ("main", main)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "30");
}

// ── 2. Multi-module with record ─────────────────────────────────────────────
#[test]
fn test_multi_module_record() {
    let lib = r#"
pub record Point { x: i64, y: i64 }
pub def make_point(x: i64, y: i64) -> Point {
    Point { x: x, y: y }
}
"#;
    let main = r#"
bring "lib.iris"
def f() -> i64 {
    val p = make_point(3, 4)
    p.x + p.y
}
"#;
    let result = compile_multi(&[("lib", lib), ("main", main)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "7");
}

// ── 3. Multi-module with helper function chain ──────────────────────────────
#[test]
fn test_multi_module_chain() {
    let math = r#"
pub def double(x: i64) -> i64 { x * 2 }
pub def triple(x: i64) -> i64 { x * 3 }
"#;
    let main = r#"
bring "math.iris"
def f() -> i64 {
    double(5) + triple(5)
}
"#;
    let result = compile_multi(&[("math", math), ("main", main)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "25");
}

// ── 4. Multi-module with pub const ──────────────────────────────────────────
#[test]
fn test_multi_module_constants() {
    let config = r#"pub const ANSWER: i64 = 42"#;
    let main = r#"
bring "config.iris"
def f() -> i64 {
    ANSWER
}
"#;
    let result = compile_multi(
        &[("config", config), ("main", main)],
        "main",
        EmitKind::Eval,
    )
    .unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 5. Multi-module with string functions ───────────────────────────────────
#[test]
fn test_multi_module_strings() {
    let strutil = r#"
pub def greet(name: str) -> str {
    concat("Hello, ", name)
}
"#;
    let main = r#"
bring "strutil.iris"
def f() -> str {
    greet("IRIS")
}
"#;
    let result = compile_multi(
        &[("strutil", strutil), ("main", main)],
        "main",
        EmitKind::Eval,
    )
    .unwrap();
    assert_eq!(result.trim(), "Hello, IRIS");
}

// ── 6. Multi-module with boolean logic ──────────────────────────────────────
#[test]
fn test_multi_module_bool() {
    let logic = r#"pub def is_positive(x: i64) -> bool { x > 0 }"#;
    let main = r#"
bring "logic.iris"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    bool_to_i64(is_positive(42))
}
"#;
    let result =
        compile_multi(&[("logic", logic), ("main", main)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 7. Multi-module with enum via factory function ──────────────────────────
#[test]
fn test_multi_module_enum() {
    let shapes = r#"
pub choice Shape {
    Circle(i64),
    Square(i64),
}
pub def make_circle(r: i64) -> Shape {
    Shape.Circle(r)
}
pub def area_approx(s: Shape) -> i64 {
    when s {
        Shape.Circle(r) => r * r * 3,
        Shape.Square(side) => side * side,
    }
}
"#;
    let main = r#"
bring "shapes.iris"
def f() -> i64 {
    val c = make_circle(5)
    area_approx(c)
}
"#;
    let result = compile_multi(
        &[("shapes", shapes), ("main", main)],
        "main",
        EmitKind::Eval,
    )
    .unwrap();
    assert_eq!(result.trim(), "75");
}

// ── 8. Two dependencies ────────────────────────────────────────────────────
#[test]
fn test_multi_module_two_deps() {
    let math_src = r#"pub def square(x: i64) -> i64 { x * x }"#;
    let util_src = r#"pub def add_one(x: i64) -> i64 { x + 1 }"#;
    let main = r#"
bring "math.iris"
bring "util.iris"
def f() -> i64 {
    add_one(square(5))
}
"#;
    let result = compile_multi(
        &[("math", math_src), ("util", util_src), ("main", main)],
        "main",
        EmitKind::Eval,
    )
    .unwrap();
    assert_eq!(result.trim(), "26");
}
