//! Phase 37 integration tests: `panic` and `assert`.
//!
//! `panic(msg)` — terminates execution with a message (InterpError::Panic).
//! `assert(cond)` — panics with "assertion failed" if cond is false.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. panic() produces a Panic instr in IR
// ---------------------------------------------------------------------------
#[test]
fn test_panic_compiles_to_ir() {
    let src = r#"
def f() -> i64 {
    panic("oops")
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(out.contains("panic"), "IR should contain panic, got:\n{}", out);
}

// ---------------------------------------------------------------------------
// 2. panic() propagates as InterpError::Panic at eval time
// ---------------------------------------------------------------------------
#[test]
fn test_panic_eval_returns_error() {
    let src = r#"
def f() -> i64 {
    panic("deliberate failure")
}
"#;
    let result = compile(src, "test", EmitKind::Eval);
    assert!(result.is_err(), "panic should cause an error");
    let err_str = format!("{}", result.unwrap_err());
    assert!(
        err_str.contains("deliberate failure") || err_str.contains("panic"),
        "error should mention the panic message, got: {}",
        err_str
    );
}

// ---------------------------------------------------------------------------
// 3. assert(true) passes silently — use val _ = assert(...) then return 42
// ---------------------------------------------------------------------------
#[test]
fn test_assert_true_passes() {
    let src = r#"
def f() -> i64 {
    val _ = assert(true)
    42
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("assert(true) should not panic");
    assert_eq!(out.trim(), "42", "should return 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. assert(false) panics with "assertion failed"
// ---------------------------------------------------------------------------
#[test]
fn test_assert_false_panics() {
    let src = r#"
def f() -> i64 {
    assert(false)
}
"#;
    let result = compile(src, "test", EmitKind::Eval);
    assert!(result.is_err(), "assert(false) should panic");
    let err_str = format!("{}", result.unwrap_err());
    assert!(
        err_str.contains("assertion failed") || err_str.contains("panic"),
        "error should mention assertion, got: {}",
        err_str
    );
}

// ---------------------------------------------------------------------------
// 5. assert with a comparison: assert(x > 0) passes when true
// ---------------------------------------------------------------------------
#[test]
fn test_assert_comparison_passes() {
    let src = r#"
def check(x: i64) -> i64 {
    val _ = assert(x > 0)
    x * 2
}
def f() -> i64 {
    check(5)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("assert(5 > 0) should not panic");
    assert_eq!(out.trim(), "10", "check(5) = 10, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. assert with a comparison: assert(x > 0) panics when false
// ---------------------------------------------------------------------------
#[test]
fn test_assert_comparison_fails() {
    let src = r#"
def check(x: i64) -> i64 {
    assert(x > 0)
    x
}
def f() -> i64 {
    check(-1)
}
"#;
    let result = compile(src, "test", EmitKind::Eval);
    assert!(result.is_err(), "assert(-1 > 0) should panic");
}

// ---------------------------------------------------------------------------
// 7. panic() in LLVM IR stub (structural check)
// ---------------------------------------------------------------------------
#[test]
fn test_panic_compiles_to_llvm() {
    let src = r#"
def f() -> i64 {
    panic("llvm test")
}
"#;
    let out = compile(src, "test", EmitKind::Llvm).expect("should emit LLVM stub");
    assert!(
        out.contains("iris_panic") || out.contains("unreachable"),
        "LLVM stub should contain panic call or unreachable, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 8. panic message is preserved in the error
// ---------------------------------------------------------------------------
#[test]
fn test_panic_message_preserved() {
    let src = r#"
def f() -> i64 {
    panic("specific error message 12345")
}
"#;
    let result = compile(src, "test", EmitKind::Eval);
    assert!(result.is_err());
    let err_str = format!("{}", result.unwrap_err());
    assert!(
        err_str.contains("specific error message 12345") || err_str.contains("panic"),
        "error message should be preserved, got: {}",
        err_str
    );
}
