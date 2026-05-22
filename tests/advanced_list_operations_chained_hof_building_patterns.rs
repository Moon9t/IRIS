//! Phase 110 integration tests: advanced list operations — chained HOF, building patterns.

use iris::{compile, EmitKind};

// ── 1. Map-filter-fold pipeline ─────────────────────────────────────────────
#[test]
fn test_list_pipeline() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    push(xs, 5);
    val doubled = xs.map(|x: i64| x * 2)
    val big = doubled.filter(|x: i64| x > 4)
    big.fold(0, |acc: i64, x: i64| acc + x)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // doubled = [2,4,6,8,10], big = [6,8,10], sum = 24
    assert_eq!(result.trim(), "24");
}

// ── 2. List sort ascending ──────────────────────────────────────────────────
#[test]
fn test_list_sort_ascending() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 3);
    push(xs, 1);
    push(xs, 2);
    list_sort(xs);
    list_get(xs, 0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 3. List concat combines two lists ───────────────────────────────────────
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

// ── 4. List slice extracts sub-list ─────────────────────────────────────────
#[test]
fn test_list_slice() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    push(xs, 40);
    push(xs, 50);
    val sub = list_slice(xs, 1, 4)
    list_get(sub, 0) + list_get(sub, 1) + list_get(sub, 2)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // slice [20, 30, 40], sum = 90
    assert_eq!(result.trim(), "90");
}

// ── 5. List contains check ──────────────────────────────────────────────────
#[test]
fn test_list_contains() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    bool_to_i64(list_contains(xs, 20))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 6. List contains returns false for missing ──────────────────────────────
#[test]
fn test_list_not_contains() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    bool_to_i64(list_contains(xs, 99))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ── 7. All returns true when all elements match ─────────────────────────────
#[test]
fn test_list_all_true() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val xs = list()
    push(xs, 2);
    push(xs, 4);
    push(xs, 6);
    bool_to_i64(xs.all(|x: i64| x > 0))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 8. All returns false when any element fails ─────────────────────────────
#[test]
fn test_list_all_false() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val xs = list()
    push(xs, 2);
    push(xs, 0 - 1);
    push(xs, 6);
    bool_to_i64(xs.all(|x: i64| x > 0))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}
