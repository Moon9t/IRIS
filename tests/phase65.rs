//! Phase 65 integration tests: f-string interpolation (`f"Hello {name}!"`).
//! Note: IrValue::Str displays with surrounding quotes, so expected strings include them.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Basic single-variable interpolation
// ---------------------------------------------------------------------------
#[test]
fn test_fstring_single_var() {
    let src = r#"
def f() -> str {
    val name = "World"
    f"Hello {name}!"
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "\"Hello World!\"");
}

// ---------------------------------------------------------------------------
// 2. Multiple variables
// ---------------------------------------------------------------------------
#[test]
fn test_fstring_multiple_vars() {
    let src = r#"
def f() -> str {
    val a = "foo"
    val b = "bar"
    f"{a} and {b}"
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "\"foo and bar\"");
}

// ---------------------------------------------------------------------------
// 3. No placeholders — identical to a plain string
// ---------------------------------------------------------------------------
#[test]
fn test_fstring_no_placeholders() {
    let src = r#"
def f() -> str {
    f"just a plain string"
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "\"just a plain string\"");
}

// ---------------------------------------------------------------------------
// 4. Numeric variable — use len() to get integer for clean comparison
// ---------------------------------------------------------------------------
#[test]
fn test_fstring_integer_var() {
    let src = r#"
def f() -> i64 {
    val n = 42
    val s = f"The answer is {n}"
    len(s)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // "The answer is 42" has 16 chars
    assert_eq!(result.trim(), "16");
}

// ---------------------------------------------------------------------------
// 5. Use contains() to verify interpolated content
// ---------------------------------------------------------------------------
#[test]
fn test_fstring_contains_check() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val name = "IRIS"
    val s = f"Language: {name}"
    bool_to_i64(contains(s, "IRIS"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 6. Placeholder at the start
// ---------------------------------------------------------------------------
#[test]
fn test_fstring_leading_placeholder() {
    let src = r#"
def f() -> str {
    val x = "hi"
    f"{x}, there!"
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "\"hi, there!\"");
}

// ---------------------------------------------------------------------------
// 7. f-string in a function that returns i64
// ---------------------------------------------------------------------------
#[test]
fn test_fstring_used_in_len() {
    let src = r#"
def f() -> i64 {
    val a = "hello"
    val b = "world"
    val s = f"{a} {b}"
    len(s)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // "hello world" has 11 chars
    assert_eq!(result.trim(), "11");
}

// ---------------------------------------------------------------------------
// 8. Three variables with surrounding text
// ---------------------------------------------------------------------------
#[test]
fn test_fstring_three_vars() {
    let src = r#"
def f() -> i64 {
    val x = 1
    val y = 2
    val z = 3
    val s = f"{x} + {y} = {z}"
    len(s)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // "1 + 2 = 3" has 9 chars
    assert_eq!(result.trim(), "9");
}
