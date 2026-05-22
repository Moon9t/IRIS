//! Phase 112 integration tests: string combinations and edge cases.

use iris::{compile, EmitKind};

// ── 1. String concatenation with empty ──────────────────────────────────────
#[test]
fn test_string_concat_empty() {
    let src = r#"
def f() -> str {
    concat("hello", "")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello");
}

// ── 2. String repeat ────────────────────────────────────────────────────────
#[test]
fn test_string_repeat() {
    let src = r#"
def f() -> str {
    repeat("ab", 3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "ababab");
}

// ── 3. String to_upper ─────────────────────────────────────────────────────
#[test]
fn test_string_to_upper() {
    let src = r#"
def f() -> str {
    to_upper("hello")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "HELLO");
}

// ── 4. String to_lower ─────────────────────────────────────────────────────
#[test]
fn test_string_to_lower() {
    let src = r#"
def f() -> str {
    to_lower("WORLD")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "world");
}

// ── 5. String trim ─────────────────────────────────────────────────────────
#[test]
fn test_string_trim() {
    let src = r#"
def f() -> str {
    trim("  hello  ")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello");
}

// ── 6. String starts_with / ends_with ───────────────────────────────────────
#[test]
fn test_string_starts_ends_with() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val s = "hello world"
    bool_to_i64(starts_with(s, "hello")) + bool_to_i64(ends_with(s, "world"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 7. String contains check ───────────────────────────────────────────────
#[test]
fn test_string_contains() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    bool_to_i64(contains("abcdef", "cde"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 8. String length of concatenated result ─────────────────────────────────
#[test]
fn test_string_len_concat() {
    let src = r#"
def f() -> i64 {
    val s = concat("abc", "def")
    len(s)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}
