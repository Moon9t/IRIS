//! Phase 70 integration tests: string when-patterns.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Match exact string literal
// ---------------------------------------------------------------------------
#[test]
fn test_string_when_exact_match() {
    let src = r#"
def classify(s: str) -> i64 {
    when s {
        "hello" => 1,
        _ => 0,
    }
}
def f() -> i64 { classify("hello") }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 2. No match falls through to wildcard
// ---------------------------------------------------------------------------
#[test]
fn test_string_when_no_match() {
    let src = r#"
def classify(s: str) -> i64 {
    when s {
        "hello" => 1,
        _ => 0,
    }
}
def f() -> i64 { classify("world") }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ---------------------------------------------------------------------------
// 3. Multiple string arms
// ---------------------------------------------------------------------------
#[test]
fn test_string_when_multiple_arms() {
    let src = r#"
def day_num(s: str) -> i64 {
    when s {
        "Mon" => 1,
        "Tue" => 2,
        "Wed" => 3,
        _ => 0,
    }
}
def f() -> i64 { day_num("Tue") }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// 4. String match with accumulation
// ---------------------------------------------------------------------------
#[test]
fn test_string_when_sum_scores() {
    let src = r#"
def score(s: str) -> i64 {
    when s {
        "A" => 10,
        "B" => 7,
        "C" => 4,
        _ => 0,
    }
}
def f() -> i64 { score("A") + score("B") + score("C") + score("D") }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "21");
}

// ---------------------------------------------------------------------------
// 5. Empty string match
// ---------------------------------------------------------------------------
#[test]
fn test_string_when_empty() {
    let src = r#"
def check(s: str) -> i64 {
    when s {
        "" => 1,
        _ => 0,
    }
}
def f() -> i64 { check("") + check("x") }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 6. String match in nested expression
// ---------------------------------------------------------------------------
#[test]
fn test_string_when_nested() {
    let src = r#"
def code(s: str) -> i64 {
    when s {
        "ok"  => 200,
        "err" => 500,
        _ => 404,
    }
}
def f() -> i64 {
    val a = code("ok")
    val b = code("missing")
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "604");
}

// ---------------------------------------------------------------------------
// 7. String match with guard
// ---------------------------------------------------------------------------
#[test]
fn test_string_when_with_guard() {
    let src = r#"
def check(s: str, n: i64) -> i64 {
    when s {
        "pos" if n > 0 => n,
        "pos" => 0 - n,
        _ => 0,
    }
}
def f() -> i64 { check("pos", 5) + check("pos", -3) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "8");
}

// ---------------------------------------------------------------------------
// 8. String match selecting operation
// ---------------------------------------------------------------------------
#[test]
fn test_string_when_op_select() {
    let src = r#"
def calc(op: str, a: i64, b: i64) -> i64 {
    when op {
        "add" => a + b,
        "sub" => a - b,
        "mul" => a * b,
        _ => 0,
    }
}
def f() -> i64 { calc("add", 3, 4) + calc("mul", 2, 5) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "17");
}
