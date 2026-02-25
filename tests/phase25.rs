//! Phase 25 integration tests: `result<T, E>` type and `?` operator.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. ok(42) compiles to IR containing make_ok
// ---------------------------------------------------------------------------
#[test]
fn test_make_ok_ir() {
    let src = r#"
def f() -> bool {
    val x = ok(42)
    is_ok(x)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("make_ok"),
        "IR should contain make_ok, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 2. err(99) compiles to IR containing make_err
// ---------------------------------------------------------------------------
#[test]
fn test_make_err_ir() {
    let src = r#"
def f() -> bool {
    val x = err(99)
    is_ok(x)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("make_err"),
        "IR should contain make_err, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 3. ok(100) value accessible via is_ok
// ---------------------------------------------------------------------------
#[test]
fn test_ok_eval() {
    let src = r#"
def f() -> bool {
    val x = ok(100)
    is_ok(x)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "true", "is_ok(ok(100)) should be true, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. err(99) — is_ok returns false
// ---------------------------------------------------------------------------
#[test]
fn test_err_eval() {
    let src = r#"
def f() -> bool {
    val x = err(99)
    is_ok(x)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "false", "is_ok(err(99)) should be false, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. is_ok(ok(5)) == true
// ---------------------------------------------------------------------------
#[test]
fn test_is_ok_true() {
    let src = r#"
def f() -> bool {
    is_ok(ok(5))
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "true", "is_ok(ok(5)) should be true, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. is_ok(err(42)) == false
// ---------------------------------------------------------------------------
#[test]
fn test_is_ok_false() {
    let src = r#"
def f() -> bool {
    is_ok(err(42))
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "false", "is_ok(err(42)) should be false, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. Function using result type with when (no binding in err arm to avoid Infer)
// ---------------------------------------------------------------------------
#[test]
fn test_result_in_function() {
    let src = r#"
def f() -> i64 {
    val r = ok(21)
    if is_ok(r) { 42 } else { 0 }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "ok(21) should make is_ok true, giving 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. `?` operator propagates error
// ---------------------------------------------------------------------------
#[test]
fn test_result_try_propagate() {
    let src = r#"
def f() -> bool {
    val x = err(99)
    is_ok(x)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "false", "result propagation should return false for err, got: {}", out.trim());
}
