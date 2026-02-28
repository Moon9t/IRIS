//! Phase 71 integration tests: nested pattern matching (sequential when composition).

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Nested option patterns: some(some(v))
// ---------------------------------------------------------------------------
#[test]
fn test_nested_option_double_some() {
    let src = r#"
def wrap2(dummy: i64) -> option<option<i64>> { some(some(42)) }
def f() -> i64 {
    val outer = wrap2(0)
    when outer {
        some(inner) => when inner {
            some(v) => v,
            none => 0 - 1,
        },
        none => 0 - 2,
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// 2. Nested option: outer some, inner none
// ---------------------------------------------------------------------------
#[test]
fn test_nested_option_some_of_none() {
    let src = r#"
def make_some_none(dummy: i64) -> option<option<i64>> { some(none) }
def f() -> i64 {
    val outer = make_some_none(0)
    when outer {
        some(inner) => when inner {
            some(v) => v,
            none => 99,
        },
        none => 0 - 1,
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "99");
}

// ---------------------------------------------------------------------------
// 3. Nested enum: choice type matched in body
// ---------------------------------------------------------------------------
#[test]
fn test_nested_choice_matching() {
    let src = r#"
choice Color { Red, Green, Blue }
choice Shape { Circle(i64), Square(i64) }
def describe(c: Color, s: Shape) -> i64 {
    when c {
        Color.Red => when s {
            Shape.Circle(r) => r,
            Shape.Square(side) => side * 2,
        },
        Color.Green => 0,
        Color.Blue => 0 - 1,
    }
}
def f() -> i64 { describe(Color.Red, Shape.Circle(7)) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "7");
}

// ---------------------------------------------------------------------------
// 4. Nested result inside option (no type annotations)
// ---------------------------------------------------------------------------
#[test]
fn test_nested_result_in_option() {
    let src = r#"
def make_some_ok(dummy: i64) -> option<result<i64, i64>> { some(ok(55)) }
def f() -> i64 {
    val x = make_some_ok(0)
    when x {
        some(inner) => when inner {
            ok(v) => v,
            err(e) => 0 - e,
        },
        none => 0,
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "55");
}

// ---------------------------------------------------------------------------
// 5. Nested enum: two independent enums composed in when
// ---------------------------------------------------------------------------
#[test]
fn test_nested_enum_composition() {
    let src = r#"
choice Dir { North, South, East, West }
choice Sign { Pos, Neg }
def value(d: Dir, s: Sign) -> i64 {
    val base = when d {
        Dir.North => 10,
        Dir.South => 20,
        Dir.East  => 30,
        Dir.West  => 40,
    }
    when s {
        Sign.Pos => base,
        Sign.Neg => 0 - base,
    }
}
def f() -> i64 { value(Dir.East, Sign.Neg) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "-30");
}

// ---------------------------------------------------------------------------
// 6. Triple nested option
// ---------------------------------------------------------------------------
#[test]
fn test_triple_nested_option() {
    let src = r#"
def make3(dummy: i64) -> option<option<option<i64>>> { some(some(some(7))) }
def unwrap3(x: option<option<option<i64>>>, default: i64) -> i64 {
    when x {
        some(a) => when a {
            some(b) => when b {
                some(v) => v,
                none => default,
            },
            none => default,
        },
        none => default,
    }
}
def f() -> i64 { unwrap3(make3(0), 0) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "7");
}

// ---------------------------------------------------------------------------
// 7. Nested pattern with guard on inner match
// ---------------------------------------------------------------------------
#[test]
fn test_nested_with_inner_guard() {
    let src = r#"
def f() -> i64 {
    val x = some(5)
    val result = when x {
        some(v) => when v > 3 {
            true  => v * 2,
            false => v,
        },
        none => 0,
    }
    result
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ---------------------------------------------------------------------------
// 8. Nested enum + option combination
// ---------------------------------------------------------------------------
#[test]
fn test_nested_enum_with_option() {
    let src = r#"
choice Shape { Circle(i64), Square(i64) }
def area_or_default(s: option<Shape>, default: i64) -> i64 {
    when s {
        some(shape) => when shape {
            Shape.Circle(r) => r * r,
            Shape.Square(side) => side * side,
        },
        none => default,
    }
}
def f() -> i64 {
    area_or_default(some(Shape.Circle(5)), 0) + area_or_default(none, 99)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "124");
}
