//! Phase 109 integration tests: advanced map operations.

use iris::{compile, EmitKind};

// ── 1. Map overwrite existing key ───────────────────────────────────────────
#[test]
fn test_map_overwrite_key() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "x", 10);
    map_set(m, "x", 99);
    val opt = map_get(m, "x")
    if is_some(opt) { unwrap(opt) } else { -1 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "99");
}

// ── 2. Map with multiple keys ───────────────────────────────────────────────
#[test]
fn test_map_multiple_keys() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "a", 1);
    map_set(m, "b", 2);
    map_set(m, "c", 3);
    map_len(m)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 3. Map contains returns true for existing ───────────────────────────────
#[test]
fn test_map_contains_true() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val m = map()
    map_set(m, "hello", 42);
    bool_to_i64(map_contains(m, "hello"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 4. Map contains returns false for missing ───────────────────────────────
#[test]
fn test_map_contains_false() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val m = map()
    map_set(m, "hello", 42);
    bool_to_i64(map_contains(m, "world"))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ── 5. Map keys returns correct count ───────────────────────────────────────
#[test]
fn test_map_keys_count() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "x", 1);
    map_set(m, "y", 2);
    map_set(m, "z", 3);
    list_len(map_keys(m))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 6. Map values returns correct count ─────────────────────────────────────
#[test]
fn test_map_values_count() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "a", 10);
    map_set(m, "b", 20);
    list_len(map_values(m))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 7. Map used as frequency counter ────────────────────────────────────────
#[test]
fn test_map_frequency_counter() {
    let src = r#"
def f() -> i64 {
    val m = map()
    map_set(m, "apple", 1);
    map_set(m, "banana", 2);
    map_set(m, "apple", 3);
    val opt = map_get(m, "apple")
    if is_some(opt) { unwrap(opt) } else { 0 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 8. Empty map has no keys ────────────────────────────────────────────────
#[test]
fn test_map_empty_keys() {
    let src = r#"
def f() -> i64 {
    val m = map()
    list_len(map_keys(m))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}
