//! Phase 104 integration tests: closures as higher-order function arguments.

use iris::{compile, EmitKind};

// ── 1. Pass inline closure to list.map ──────────────────────────────────────
#[test]
fn test_closure_inline_map() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    val ys = xs.map(|x: i64| x + 10)
    list_get(ys, 0) + list_get(ys, 1) + list_get(ys, 2)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "36");
}

// ── 2. Pass inline closure to list.filter ───────────────────────────────────
#[test]
fn test_closure_inline_filter() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    push(xs, 5);
    val evens = xs.filter(|x: i64| x > 2)
    list_len(evens)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 3. Pass inline closure to list.fold for string concat ───────────────────
#[test]
fn test_closure_fold_string() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    val total = xs.fold(0, |acc: i64, x: i64| acc + x)
    total
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ── 4. Closure capturing outer variable in map ──────────────────────────────
#[test]
fn test_closure_capture_in_map() {
    let src = r#"
def f() -> i64 {
    val offset = 100
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    val ys = xs.map(|x: i64| x + offset)
    list_get(ys, 2)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "103");
}

// ── 5. Closure capturing outer variable in filter ───────────────────────────
#[test]
fn test_closure_capture_in_filter() {
    let src = r#"
def f() -> i64 {
    val threshold = 3
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    push(xs, 5);
    val big = xs.filter(|x: i64| x > threshold)
    list_len(big)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 6. Chained map then filter ──────────────────────────────────────────────
#[test]
fn test_closure_map_then_filter() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    val doubled = xs.map(|x: i64| x * 2)
    val big = doubled.filter(|x: i64| x > 4)
    list_len(big)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 7. Chained filter then fold ─────────────────────────────────────────────
#[test]
fn test_closure_filter_then_fold() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    push(xs, 5);
    val big = xs.filter(|x: i64| x > 2)
    big.fold(0, |acc: i64, x: i64| acc + x)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "12");
}

// ── 8. Any with closure ─────────────────────────────────────────────────────
#[test]
fn test_closure_any() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    bool_to_i64(xs.any(|x: i64| x > 2))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}
