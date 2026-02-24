//! Phase 21 integration tests: String type (`str`), `len()`, `concat()`, `print()`.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. String literal lowering and IR emission
// ---------------------------------------------------------------------------
#[test]
fn test_str_literal_ir() {
    let src = r#"
def f() -> str {
    val s = "hello"
    s
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(out.contains("const.str"), "IR should contain const.str, got:\n{}", out);
}

// ---------------------------------------------------------------------------
// 2. String literal eval
// ---------------------------------------------------------------------------
#[test]
fn test_str_literal_eval() {
    let src = r#"
def f() -> str {
    "hello world"
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("hello world"), "should get string value, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. len() builtin
// ---------------------------------------------------------------------------
#[test]
fn test_str_len_eval() {
    let src = r#"
def f() -> i64 {
    val s = "hello"
    len(s)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "5", "len(\"hello\") should be 5, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. len() on empty string
// ---------------------------------------------------------------------------
#[test]
fn test_str_len_empty() {
    let src = r#"
def f() -> i64 {
    len("")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "0", "len(\"\") should be 0, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. concat() builtin
// ---------------------------------------------------------------------------
#[test]
fn test_str_concat_eval() {
    let src = r#"
def f() -> str {
    val a = "hello"
    val b = " world"
    concat(a, b)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("hello world"), "concat should produce 'hello world', got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. str type annotation in function parameter
// ---------------------------------------------------------------------------
#[test]
fn test_str_type_in_param() {
    let src = r#"
def f() -> str {
    greet("Alice")
}
def greet(name: str) -> str {
    concat("Hello, ", name)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("Hello, Alice"), "should greet Alice, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. len() of concatenated string
// ---------------------------------------------------------------------------
#[test]
fn test_len_of_concat() {
    let src = r#"
def f() -> i64 {
    val s = concat("abc", "de")
    len(s)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "5", "len of 'abcde' should be 5, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. String IR contains const.str and str_len
// ---------------------------------------------------------------------------
#[test]
fn test_str_ir_contains_str_len() {
    let src = r#"
def f() -> i64 {
    val s = "test"
    len(s)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(ir.contains("const.str"), "IR should contain const.str, got:\n{}", ir);
    assert!(ir.contains("str_len"), "IR should contain str_len, got:\n{}", ir);
}
