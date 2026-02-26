//! Phase 44: HashMap type
//!
//! Tests for: map(), map_set(), map_get(), map_contains(), map_remove(), map_len()

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// empty map has length 0
// ---------------------------------------------------------------------------

#[test]
fn test_map_new_len_zero() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_len(m)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ---------------------------------------------------------------------------
// map_set increases length
// ---------------------------------------------------------------------------

#[test]
fn test_map_set_len() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "a", 1);
    map_set(m, "b", 2);
    map_len(m)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// map_get returns some(v) for existing key
// ---------------------------------------------------------------------------

#[test]
fn test_map_get_some() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "x", 42);
    val opt = map_get(m, "x")
    if is_some(opt) { unwrap(opt) } else { -1 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// map_get returns none for missing key
// ---------------------------------------------------------------------------

#[test]
fn test_map_get_none() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "x", 42);
    val opt = map_get(m, "y")
    if is_some(opt) { unwrap(opt) } else { -1 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "-1");
}

// ---------------------------------------------------------------------------
// map_contains returns true for existing key
// ---------------------------------------------------------------------------

#[test]
fn test_map_contains_true() {
    let src = r#"
def f() -> bool {
    val m = map()
    map_set(m, "hello", 99);
    map_contains(m, "hello")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "true");
}

// ---------------------------------------------------------------------------
// map_contains returns false for missing key
// ---------------------------------------------------------------------------

#[test]
fn test_map_contains_false() {
    let src = r#"
def f() -> bool {
    val m = map()
    map_set(m, "hello", 99);
    map_contains(m, "world")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "false");
}

// ---------------------------------------------------------------------------
// IR text emission
// ---------------------------------------------------------------------------

#[test]
fn test_map_ir_text() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "k", 1);
    map_len(m)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("map_new"), "expected map_new in IR:\n{}", ir);
    assert!(ir.contains("map_set"), "expected map_set in IR:\n{}", ir);
    assert!(ir.contains("map_len"), "expected map_len in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// LLVM stub emission
// ---------------------------------------------------------------------------

#[test]
fn test_map_llvm() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "k", 1);
    map_len(m)
}
"#;
    let ll = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(ll.contains("iris_map_new"), "expected iris_map_new in LLVM:\n{}", ll);
    assert!(ll.contains("iris_map_set"), "expected iris_map_set in LLVM:\n{}", ll);
    assert!(ll.contains("iris_map_len"), "expected iris_map_len in LLVM:\n{}", ll);
}
