//! Phase 121 integration tests: advanced f-string and formatting.

use iris::{compile, EmitKind};

// ── 1. F-string with multiple types ─────────────────────────────────────────
#[test]
fn test_fstring_multi_type() {
    let src = r#"
def f() -> str {
    val name = "IRIS"
    val ver = "2"
    f"{name} v{ver}"
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "IRIS v2");
}

// ── 2. F-string with computed value ─────────────────────────────────────────
#[test]
fn test_fstring_computed() {
    let src = r#"
def f() -> str {
    val x = 21
    val doubled = to_str(x * 2)
    f"Answer: {doubled}"
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Answer: 42");
}

// ── 3. F-string nested in concat ────────────────────────────────────────────
#[test]
fn test_fstring_in_concat() {
    let src = r#"
def f() -> str {
    val greeting = "Hello"
    val name = "World"
    concat(f"{greeting} ", f"{name}!")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Hello World!");
}

// ── 4. F-string empty placeholder list ──────────────────────────────────────
#[test]
fn test_fstring_plain_text() {
    let src = r#"
def f() -> str {
    f"no placeholders here"
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "no placeholders here");
}

// ── 5. to_str with various values ───────────────────────────────────────────
#[test]
fn test_to_str_variety() {
    let src = r#"
def f() -> str {
    val a = to_str(42)
    val b = to_str(true)
    concat(concat(a, " "), b)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42 true");
}

// ── 6. String replace ──────────────────────────────────────────────────────
#[test]
fn test_string_replace() {
    let src = r#"
def f() -> str {
    str_replace("hello world", "world", "iris")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello iris");
}

// ── 7. String find ─────────────────────────────────────────────────────────
#[test]
fn test_string_find() {
    let src = r#"
def f() -> i64 {
    unwrap(find("hello world", "world"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ── 8. String slice ────────────────────────────────────────────────────────
#[test]
fn test_string_slice() {
    let src = r#"
def f() -> str {
    slice("hello world", 0, 5)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello");
}
