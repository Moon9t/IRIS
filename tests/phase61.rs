//! Phase 61 integration tests: enhanced pattern matching.
//! Covers guards, wildcard `_`, integer/bool/string literal patterns.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Wildcard `_` matches anything
// ---------------------------------------------------------------------------
#[test]
fn test_wildcard_pattern() {
    let src = r#"
def classify(n: i64) -> i64 {
    when n {
        0 => 0,
        _ => 1,
    }
}
def f() -> i64 {
    classify(0) + classify(5) + classify(-3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// 2. Integer literal patterns
// ---------------------------------------------------------------------------
#[test]
fn test_int_literal_pattern() {
    let src = r#"
def describe(n: i64) -> i64 {
    when n {
        1 => 10,
        2 => 20,
        3 => 30,
        _ => 0,
    }
}
def f() -> i64 {
    describe(1) + describe(2) + describe(3) + describe(99)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ---------------------------------------------------------------------------
// 3. Bool literal patterns
// ---------------------------------------------------------------------------
#[test]
fn test_bool_literal_pattern() {
    let src = r#"
def to_int(b: bool) -> i64 {
    when b {
        true  => 1,
        false => 0,
    }
}
def f() -> i64 {
    to_int(true) + to_int(false) + to_int(true)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// 4. Guard on option: `some(v) if v > 0`
// ---------------------------------------------------------------------------
#[test]
fn test_guard_on_option_some() {
    let src = r#"
def safe_value(x: option<i64>) -> i64 {
    when x {
        some(v) if v > 0 => v,
        some(v) => 0,
        none => -1,
    }
}
def f() -> i64 {
    safe_value(some(42)) + safe_value(some(-5)) + safe_value(none)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "41");
}

// ---------------------------------------------------------------------------
// 5. Guard on integer literal
// ---------------------------------------------------------------------------
#[test]
fn test_guard_on_int_with_wildcard() {
    let src = r#"
def categorize(n: i64) -> i64 {
    when n {
        0 => 100,
        _ if n > 0 => 1,
        _ => -1,
    }
}
def f() -> i64 {
    categorize(0) + categorize(5) + categorize(-3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "100");
}

// ---------------------------------------------------------------------------
// 6. Mixed enum variants + wildcard
// ---------------------------------------------------------------------------
#[test]
fn test_enum_with_wildcard() {
    let src = r#"
choice Color { Red, Green, Blue }

def is_red(c: Color) -> i64 {
    when c {
        Color.Red => 1,
        _ => 0,
    }
}
def f() -> i64 {
    is_red(Color.Red) + is_red(Color.Green) + is_red(Color.Blue)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 7. Wildcard must be last: catches all remaining cases
// ---------------------------------------------------------------------------
#[test]
fn test_wildcard_catches_remaining() {
    let src = r#"
def f() -> i64 {
    val a = when 10 { 1 => 1, 2 => 2, _ => 99 }
    val b = when 2  { 1 => 1, 2 => 2, _ => 99 }
    val c = when 1  { 1 => 1, 2 => 2, _ => 99 }
    a + b + c
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "102");
}

// ---------------------------------------------------------------------------
// 8. Guard on ADT variant
// ---------------------------------------------------------------------------
#[test]
fn test_guard_on_adt_variant() {
    let src = r#"
choice Msg {
    Value(i64),
    Empty,
}
def process(m: Msg) -> i64 {
    when m {
        Msg.Value(v) if v > 100 => 2,
        Msg.Value(v) => 1,
        Msg.Empty => 0,
    }
}
def f() -> i64 {
    process(Msg.Value(200)) + process(Msg.Value(50)) + process(Msg.Empty)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}
