//! Integration tests for multi-crossfile project namespace mangling and qualified references.

use iris::{compile_multi, EmitKind};

#[test]
fn test_qualified_function_and_struct() {
    let utils_src = r#"
pub record Circle {
    radius: f64
}
pub def get_pi() -> f64 {
    3.14
}
pub def area(c: Circle) -> f64 {
    c.radius * c.radius * get_pi()
}
"#;
    let main_src = r#"
bring utils
def f() -> f64 {
    val c = utils.Circle { radius: 2.0 }
    utils.area(c)
}
"#;
    let result = compile_multi(
        &[("utils", utils_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    )
    .unwrap();
    // 2.0 * 2.0 * 3.14 = 12.56
    assert_eq!(result.trim(), "12.56");
}

#[test]
fn test_qualified_constants() {
    let utils_src = r#"
pub const VAL: i64 = 42
"#;
    let main_src = r#"
bring utils
def f() -> i64 {
    utils.VAL
}
"#;
    let result = compile_multi(
        &[("utils", utils_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    )
    .unwrap();
    assert_eq!(result.trim(), "42");
}

#[test]
fn test_qualified_enum_variants() {
    let utils_src = r#"
pub choice Shape {
    Circle(i64),
    Square(i64),
    Unknown
}
"#;
    let main_src = r#"
bring utils
def f() -> i64 {
    val s1 = utils.Shape.Circle(10)
    val s2 = utils.Shape.Unknown
    when s1 {
        utils.Shape.Circle(x) => x,
        utils.Shape.Square(s) => s * s,
        _ => 0
    }
}
"#;
    let result = compile_multi(
        &[("utils", utils_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    )
    .unwrap();
    assert_eq!(result.trim(), "10");
}
