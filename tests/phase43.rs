//! Phase 43: List/Vec dynamic array type
//!
//! Tests for: list(), push(), list_len(), list_get(), list_set(), list_pop()

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// list_new + list_len: empty list has length 0
// ---------------------------------------------------------------------------

#[test]
fn test_list_new_len_zero() {
    let src = r#"
def f() -> i64 {
    val v = list()
    list_len(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ---------------------------------------------------------------------------
// push increases length
// ---------------------------------------------------------------------------

#[test]
fn test_list_push_len() {
    let src = r#"
def f() -> i64 {
    val v = list()
    push(v, 10);
    push(v, 20);
    push(v, 30);
    list_len(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ---------------------------------------------------------------------------
// list_get retrieves the correct element
// ---------------------------------------------------------------------------

#[test]
fn test_list_get() {
    let src = r#"
def f() -> i64 {
    val v = list()
    push(v, 100);
    push(v, 200);
    push(v, 300);
    list_get(v, 1)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "200");
}

// ---------------------------------------------------------------------------
// list_set mutates a slot
// ---------------------------------------------------------------------------

#[test]
fn test_list_set() {
    let src = r#"
def f() -> i64 {
    val v = list()
    push(v, 1);
    push(v, 2);
    push(v, 3);
    list_set(v, 0, 99);
    list_get(v, 0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "99");
}

// ---------------------------------------------------------------------------
// list_pop returns the last element
// ---------------------------------------------------------------------------

#[test]
fn test_list_pop() {
    let src = r#"
def f() -> i64 {
    val v = list()
    push(v, 7);
    push(v, 8);
    push(v, 9);
    list_pop(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "9");
}

// ---------------------------------------------------------------------------
// list_pop reduces length by 1
// ---------------------------------------------------------------------------

#[test]
fn test_list_pop_reduces_len() {
    let src = r#"
def f() -> i64 {
    val v = list()
    push(v, 1);
    push(v, 2);
    val _ = list_pop(v)
    list_len(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// IR text emission contains list_new / list_push / list_len
// ---------------------------------------------------------------------------

#[test]
fn test_list_ir_text() {
    let src = r#"
def f() -> i64 {
    val v = list()
    push(v, 42);
    list_len(v)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("list_new"), "expected list_new in IR:\n{}", ir);
    assert!(ir.contains("list_push"), "expected list_push in IR:\n{}", ir);
    assert!(ir.contains("list_len"), "expected list_len in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// LLVM stub contains iris_list_* calls
// ---------------------------------------------------------------------------

#[test]
fn test_list_llvm() {
    let src = r#"
def f() -> i64 {
    val v = list()
    push(v, 1);
    list_len(v)
}
"#;
    let ll = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(ll.contains("iris_list_new"), "expected iris_list_new in LLVM:\n{}", ll);
    assert!(ll.contains("iris_list_push"), "expected iris_list_push in LLVM:\n{}", ll);
    assert!(ll.contains("iris_list_len"), "expected iris_list_len in LLVM:\n{}", ll);
}
