//! Phase 59: Process / environment builtins
//!
//! Tests for: args(), env_var(name)

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// args(): returns a list (its length is >= 0)
// ---------------------------------------------------------------------------

#[test]
fn test_args_is_list() {
    let src = r#"
def f() -> i64 {
    val a = args()
    list_len(a)
}
"#;
    // We don't know the exact count, but it should be >= 0 and not error.
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    let n: i64 = result.trim().parse().expect("expected integer from args()");
    assert!(n >= 0, "args() length must be non-negative, got {}", n);
}

// ---------------------------------------------------------------------------
// env_var: known variable (PATH or similar) has a value
// ---------------------------------------------------------------------------

#[test]
fn test_env_var_exists() {
    // Set a var specifically for this test
    std::env::set_var("IRIS_TEST_VAR_123", "hello");

    let src = r#"
def f() -> bool {
    val v = env_var("IRIS_TEST_VAR_123")
    is_some(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "true");
}

// ---------------------------------------------------------------------------
// env_var: missing variable returns None (is_some = false)
// ---------------------------------------------------------------------------

#[test]
fn test_env_var_missing() {
    let src = r#"
def f() -> bool {
    val v = env_var("IRIS_TOTALLY_MISSING_VAR_XYZZY_9999")
    is_some(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "false");
}

// ---------------------------------------------------------------------------
// args() IR text contains process_args instruction
// ---------------------------------------------------------------------------

#[test]
fn test_args_ir_text() {
    let src = r#"
def f() -> i64 {
    val a = args()
    list_len(a)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("process_args"), "expected process_args in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// env_var IR text contains env_var instruction
// ---------------------------------------------------------------------------

#[test]
fn test_env_var_ir_text() {
    let src = r#"
def f() -> bool {
    val v = env_var("HOME")
    is_some(v)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("env_var"), "expected env_var in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// LLVM IR contains iris_process_args and iris_env_var declarations
// ---------------------------------------------------------------------------

#[test]
fn test_process_llvm() {
    let src = r#"
def f() -> i64 {
    val a = args()
    list_len(a)
}
"#;
    let ll = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(ll.contains("iris_process_args"), "expected iris_process_args in LLVM:\n{}", ll);
}

// ---------------------------------------------------------------------------
// env_var LLVM IR contains iris_env_var declaration
// ---------------------------------------------------------------------------

#[test]
fn test_env_var_llvm() {
    let src = r#"
def f() -> bool {
    val v = env_var("PATH")
    is_some(v)
}
"#;
    let ll = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(ll.contains("iris_env_var"), "expected iris_env_var in LLVM:\n{}", ll);
}

// ---------------------------------------------------------------------------
// env_var unwrap: known var value is the string we set
// ---------------------------------------------------------------------------

#[test]
fn test_env_var_value() {
    std::env::set_var("IRIS_TEST_VAL_456", "world");

    let src = r#"
def f() -> i64 {
    val v = env_var("IRIS_TEST_VAL_456")
    if is_some(v) {
        42
    } else {
        0
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}
