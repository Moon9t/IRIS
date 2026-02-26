//! Phase 33 integration tests: extended string operations.
//!
//! New builtins: contains, starts_with, ends_with, to_upper, to_lower, trim, repeat.
//!
//! Note: String values are displayed with surrounding quotes in eval output,
//! so we use `output.contains("expected")` for string results.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. contains("hello world", "world") == true
// ---------------------------------------------------------------------------
#[test]
fn test_str_contains_true() {
    let src = r#"
def f() -> bool {
    contains("hello world", "world")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "true", "contains should be true, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 2. contains("hello", "xyz") == false
// ---------------------------------------------------------------------------
#[test]
fn test_str_contains_false() {
    let src = r#"
def f() -> bool {
    contains("hello", "xyz")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "false", "contains should be false, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. starts_with("hello", "hel") == true
// ---------------------------------------------------------------------------
#[test]
fn test_str_starts_with() {
    let src = r#"
def f() -> bool {
    starts_with("hello", "hel")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "true", "starts_with should be true, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. ends_with("hello", "llo") == true
// ---------------------------------------------------------------------------
#[test]
fn test_str_ends_with() {
    let src = r#"
def f() -> bool {
    ends_with("hello", "llo")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "true", "ends_with should be true, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. to_upper("hello") output contains "HELLO"
// ---------------------------------------------------------------------------
#[test]
fn test_str_to_upper() {
    let src = r#"
def f() -> str {
    to_upper("hello")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("HELLO"), "to_upper should produce HELLO, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. to_lower("WORLD") output contains "world"
// ---------------------------------------------------------------------------
#[test]
fn test_str_to_lower() {
    let src = r#"
def f() -> str {
    to_lower("WORLD")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("world"), "to_lower should produce world, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. trim("  hello  ") output contains "hello"
// ---------------------------------------------------------------------------
#[test]
fn test_str_trim() {
    let src = r#"
def f() -> str {
    trim("  hello  ")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("hello"), "trim should produce hello, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. repeat("ab", 3) output contains "ababab"
// ---------------------------------------------------------------------------
#[test]
fn test_str_repeat() {
    let src = r#"
def f() -> str {
    repeat("ab", 3)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("ababab"), "repeat should produce ababab, got: {}", out.trim());
}
