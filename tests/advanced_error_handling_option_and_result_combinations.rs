//! Phase 106 integration tests: advanced error handling — option and result combinations.

use iris::{compile, EmitKind};

// ── 1. Nested option: some wrapping a value ─────────────────────────────────
#[test]
fn test_option_some_unwrap() {
    let src = r#"
def f() -> i64 {
    val x = some(42)
    if is_some(x) { unwrap(x) } else { 0 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 2. Option none check ───────────────────────────────────────────────────
#[test]
fn test_option_none_check() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val x = none()
    bool_to_i64(is_some(x))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ── 3. Function returning option with conditional ───────────────────────────
#[test]
fn test_option_conditional_return() {
    let src = r#"
def find_positive(x: i64) -> i64 {
    val opt = if x > 0 { some(x) } else { none() }
    if is_some(opt) { unwrap(opt) } else { -1 }
}
def f() -> i64 {
    find_positive(5) + find_positive(0 - 3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "4");
}

// ── 4. Result ok unwrap path ────────────────────────────────────────────────
#[test]
fn test_result_ok_path() {
    let src = r#"
def f() -> i64 {
    val r = ok(100)
    if is_ok(r) { 100 } else { 0 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "100");
}

// ── 5. Result err path ─────────────────────────────────────────────────────
#[test]
fn test_result_err_path() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val r = err(42)
    bool_to_i64(is_ok(r))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ── 6. Map lookup returns option, chain check ───────────────────────────────
#[test]
fn test_map_get_option_chain() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "key", 99);
    val opt = map_get(m, "key")
    val present = if is_some(opt) { unwrap(opt) } else { 0 }
    val opt2 = map_get(m, "missing")
    val absent = if is_some(opt2) { unwrap(opt2) } else { -1 }
    present + absent
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "98");
}

// ── 7. Multiple option checks in sequence ───────────────────────────────────
#[test]
fn test_multiple_option_checks() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val a = some(10)
    val b = none()
    val c = some(30)
    bool_to_i64(is_some(a)) + bool_to_i64(is_some(b)) + bool_to_i64(is_some(c))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 8. Result used in if-else chain ─────────────────────────────────────────
#[test]
fn test_result_if_chain() {
    let src = r#"
def safe_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        0
    } else {
        a / b
    }
}
def f() -> i64 {
    safe_div(100, 5) + safe_div(10, 0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "20");
}
