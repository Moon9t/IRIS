//! Phase 58: Extended collection operations
//!
//! Tests for: list_contains, list_sort, map_keys, map_values, list_concat, list_slice

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// list_contains: present element returns true
// ---------------------------------------------------------------------------

#[test]
fn test_list_contains_true() {
    let src = r#"
def f() -> bool {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    list_contains(xs, 20)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "true");
}

// ---------------------------------------------------------------------------
// list_contains: absent element returns false
// ---------------------------------------------------------------------------

#[test]
fn test_list_contains_false() {
    let src = r#"
def f() -> bool {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    list_contains(xs, 99)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "false");
}

// ---------------------------------------------------------------------------
// list_sort: sorts elements ascending
// ---------------------------------------------------------------------------

#[test]
fn test_list_sort() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 5);
    push(xs, 1);
    push(xs, 3);
    list_sort(xs);
    list_get(xs, 0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// list_concat: concatenates two lists
// ---------------------------------------------------------------------------

#[test]
fn test_list_concat() {
    let src = r#"
def f() -> i64 {
    val a = list()
    push(a, 1);
    push(a, 2);
    val b = list()
    push(b, 3);
    push(b, 4);
    val c = list_concat(a, b)
    list_len(c)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "4");
}

// ---------------------------------------------------------------------------
// list_slice: slices a subrange
// ---------------------------------------------------------------------------

#[test]
fn test_list_slice() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    push(xs, 40);
    val sl = list_slice(xs, 1, 3)
    list_len(sl)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// map_keys: returns keys of a map
// ---------------------------------------------------------------------------

#[test]
fn test_map_keys() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "a", 1);
    map_set(m, "b", 2);
    map_set(m, "c", 3);
    val ks = map_keys(m)
    list_len(ks)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ---------------------------------------------------------------------------
// map_values: returns values of a map
// ---------------------------------------------------------------------------

#[test]
fn test_map_values() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "x", 100);
    map_set(m, "y", 200);
    val vs = map_values(m)
    list_len(vs)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// IR text contains new collection instructions
// ---------------------------------------------------------------------------

#[test]
fn test_collection_ir_text() {
    let src = r#"
def f() -> bool {
    val xs = list()
    push(xs, 42);
    list_contains(xs, 42)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("list_contains"), "expected list_contains in IR:\n{}", ir);
}
