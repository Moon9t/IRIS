//! Phase 118 integration tests: advanced option patterns.

use iris::{compile, EmitKind};

// ── 1. Option with map lookup default ───────────────────────────────────────
#[test]
fn test_option_map_default() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "x", 42);
    val opt_x = map_get(m, "x")
    val opt_y = map_get(m, "y")
    val a = if is_some(opt_x) { unwrap(opt_x) } else { 0 }
    val b = if is_some(opt_y) { unwrap(opt_y) } else { 99 }
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "141");
}

// ── 2. Multiple some values ─────────────────────────────────────────────────
#[test]
fn test_multiple_somes() {
    let src = r#"
def f() -> i64 {
    val a = some(10)
    val b = some(20)
    val c = some(30)
    unwrap(a) + unwrap(b) + unwrap(c)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ── 3. Option in conditional chain ──────────────────────────────────────────
#[test]
fn test_option_conditional_chain() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    val a = if 1 < list_len(xs) { list_get(xs, 1) } else { 0 - 1 }
    val b = if 5 < list_len(xs) { list_get(xs, 5) } else { 0 - 1 }
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "19");
}

// ── 4. Option some with string ──────────────────────────────────────────────
#[test]
fn test_option_some_string() {
    let src = r#"
def f() -> i64 {
    val opt = some("hello")
    if is_some(opt) { len(unwrap(opt)) } else { 0 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "5");
}

// ── 5. Option none default to integer ────────────────────────────────────────
#[test]
fn test_option_none_default_str() {
    let src = r#"
def f() -> i64 {
    val opt = some(77)
    val opt2 = some(0)
    val a = if is_some(opt) { unwrap(opt) } else { 0 }
    val b = if is_some(opt2) { unwrap(opt2) } else { 99 }
    a + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "77");
}

// ── 6. Chained option checks ───────────────────────────────────────────────
#[test]
fn test_chained_option_checks() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "a", 1);
    map_set(m, "b", 2);
    val opt_a = map_get(m, "a")
    val a = if is_some(opt_a) { unwrap(opt_a) } else { 0 }
    val opt_b = map_get(m, "b")
    val b = if is_some(opt_b) { unwrap(opt_b) } else { 0 }
    val opt_c = map_get(m, "c")
    val c = if is_some(opt_c) { unwrap(opt_c) } else { 0 }
    a + b + c
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 7. Option-like values from list sum ─────────────────────────────────────
#[test]
fn test_option_in_loop() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    var sum = 0
    for x in xs {
        val opt = some(x)
        sum = sum + unwrap(opt)
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ── 8. Option with boolean value ────────────────────────────────────────────
#[test]
fn test_option_bool() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val a = some(true)
    val b = none()
    val va = if is_some(a) { bool_to_i64(unwrap(a)) } else { 0 - 1 }
    val vb = if is_some(b) { 99 } else { 0 - 1 }
    va + vb
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}
