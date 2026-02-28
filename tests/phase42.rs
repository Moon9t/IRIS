//! Phase 42 integration tests: string indexing and slicing.
//!
//! str_index(s, i) -> i64     byte value at position i
//! slice(s, start, end) -> str  substring [start..end)
//! find(s, sub) -> option<i64>  index of first match or none
//! str_replace(s, old, new) -> str  replace all occurrences

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. str_index returns byte value at position
// ---------------------------------------------------------------------------
#[test]
fn test_str_index() {
    let src = r#"
def f() -> i64 {
    val s = "hello"
    str_index(s, 0)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // 'h' = 104 in ASCII
    assert_eq!(out.trim(), "104", "str_index of 'h' should be 104, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 2. slice returns substring
// ---------------------------------------------------------------------------
#[test]
fn test_slice_basic() {
    let src = r#"
def f() -> str {
    val s = "hello world"
    slice(s, 6, 11)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("world"), "slice should return 'world', got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. slice of empty range returns empty string
// ---------------------------------------------------------------------------
#[test]
fn test_slice_empty() {
    let src = r#"
def f() -> str {
    val s = "hello"
    slice(s, 2, 2)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // Empty string — output is blank (no quotes in eval mode)
    assert!(out.trim().is_empty(), "slice of empty range, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. find returns some(index) when found
// ---------------------------------------------------------------------------
#[test]
fn test_find_found() {
    let src = r#"
def f() -> i64 {
    val s = "hello world"
    val r = find(s, "world")
    when r {
        some(i) => i,
        none => -1,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "6", "find should return index 6, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. find returns none when not found
// ---------------------------------------------------------------------------
#[test]
fn test_find_not_found() {
    let src = r#"
def f() -> i64 {
    val s = "hello"
    val r = find(s, "xyz")
    when r {
        some(i) => i,
        none => -1,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "-1", "find of missing sub should be none (-1), got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. str_replace replaces all occurrences
// ---------------------------------------------------------------------------
#[test]
fn test_str_replace() {
    let src = r#"
def f() -> str {
    val s = "hello world"
    str_replace(s, "o", "0")
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert!(out.contains("hell0 w0rld"), "str_replace should give 'hell0 w0rld', got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. str_index appears in IR
// ---------------------------------------------------------------------------
#[test]
fn test_str_index_ir() {
    let src = r#"
def f() -> i64 {
    str_index("abc", 1)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should emit IR");
    assert!(out.contains("str_index"), "IR should contain str_index, got:\n{}", out);
}

// ---------------------------------------------------------------------------
// 8. find appears in LLVM stub
// ---------------------------------------------------------------------------
#[test]
fn test_find_llvm() {
    let src = r#"
def f() -> i64 {
    val r = find("hello", "ll")
    when r {
        some(i) => i,
        none => -1,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Llvm).expect("should emit LLVM stub");
    assert!(out.contains("iris_str_find"), "LLVM stub should call iris_str_find, got:\n{}", out);
}
