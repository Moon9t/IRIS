//! Phase 24 integration tests: `option<T>` type (`some`, `none`, `is_some`, `unwrap`, `when`).

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. some(42) compiles to IR containing make_some
// ---------------------------------------------------------------------------
#[test]
fn test_make_some_ir() {
    let src = r#"
def f() -> i64 {
    val x = some(42)
    unwrap(x)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("make_some"),
        "IR should contain make_some, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 2. none compiles to IR containing make_none
// ---------------------------------------------------------------------------
#[test]
fn test_make_none_ir() {
    let src = r#"
def f() -> bool {
    val x = none
    is_some(x)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("make_none"),
        "IR should contain make_none, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 3. some(100) evaluates to some(100)
// ---------------------------------------------------------------------------
#[test]
fn test_some_eval() {
    let src = r#"
def f() -> i64 {
    val x = some(100)
    unwrap(x)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "100", "unwrap(some(100)) should be 100, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. is_some(some(5)) == true
// ---------------------------------------------------------------------------
#[test]
fn test_is_some_true() {
    let src = r#"
def f() -> bool {
    val x = some(5)
    is_some(x)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "true", "is_some(some(5)) should be true, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. is_some(none) == false
// ---------------------------------------------------------------------------
#[test]
fn test_is_some_false() {
    let src = r#"
def f() -> bool {
    val x = none
    is_some(x)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "false", "is_some(none) should be false, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. unwrap(some(42)) == 42
// ---------------------------------------------------------------------------
#[test]
fn test_unwrap_some() {
    let src = r#"
def f() -> i64 {
    unwrap(some(42))
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "unwrap(some(42)) should be 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. Function returning option<i64>
// ---------------------------------------------------------------------------
#[test]
fn test_option_in_function() {
    let src = r#"
def f() -> i64 {
    val x = some(21)
    val inner = unwrap(x)
    inner * 2
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "unwrap(some(21)) * 2 should be 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. when some(5) { some(x) => x, none => 0 } == 5
// ---------------------------------------------------------------------------
#[test]
fn test_option_when() {
    let src = r#"
def f() -> i64 {
    val x = some(5)
    when x { some(v) => v, none => 0 }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "5", "when some(5) {{ some(v) => v, none => 0 }} should be 5, got: {}", out.trim());
}
