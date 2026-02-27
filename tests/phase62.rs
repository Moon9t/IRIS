//! Phase 62 integration tests: iterator functional operations on lists.
//! Covers list.map, list.filter, list.fold, list.any, list.all.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. list.map — double each element
// ---------------------------------------------------------------------------
#[test]
fn test_list_map_doubles() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    val ys = xs.map(|x: i64| x * 2)
    list_get(ys, 0) + list_get(ys, 1) + list_get(ys, 2)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "12");
}

// ---------------------------------------------------------------------------
// 2. list.map — empty list yields empty list
// ---------------------------------------------------------------------------
#[test]
fn test_list_map_empty() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    val ys = xs.map(|x: i64| x * 10)
    list_len(ys)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ---------------------------------------------------------------------------
// 3. list.filter — keep elements greater than 3
// ---------------------------------------------------------------------------
#[test]
fn test_list_filter_even() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    push(xs, 5);
    val big = xs.filter(|x: i64| x > 3)
    list_len(big)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// 4. list.filter — always-true predicate keeps all elements
// ---------------------------------------------------------------------------
#[test]
fn test_list_filter_all() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    val ys = xs.filter(|x: i64| x > 0)
    list_get(ys, 0) + list_get(ys, 1) + list_get(ys, 2)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ---------------------------------------------------------------------------
// 5. list.fold — sum
// ---------------------------------------------------------------------------
#[test]
fn test_list_fold_sum() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    xs.fold(0, |acc: i64, x: i64| acc + x)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ---------------------------------------------------------------------------
// 6. list.fold — product
// ---------------------------------------------------------------------------
#[test]
fn test_list_fold_product() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    xs.fold(1, |acc: i64, x: i64| acc * x)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "24");
}

// ---------------------------------------------------------------------------
// 7. list.any — true when element matches, false when none match
// ---------------------------------------------------------------------------
#[test]
fn test_list_any() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 {
    when b { true => 1, false => 0 }
}
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 5);
    push(xs, 15);
    val has_big = xs.any(|x: i64| x > 10)
    val none_neg = xs.any(|x: i64| x < 0)
    bool_to_i64(has_big) * 10 + bool_to_i64(none_neg)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ---------------------------------------------------------------------------
// 8. list.all — true when all match, false when some don't
// ---------------------------------------------------------------------------
#[test]
fn test_list_all() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 {
    when b { true => 1, false => 0 }
}
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    val all_pos = xs.all(|x: i64| x > 0)
    val all_big = xs.all(|x: i64| x > 2)
    bool_to_i64(all_pos) * 10 + bool_to_i64(all_big)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}
