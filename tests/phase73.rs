//! Phase 73 integration tests: dead variable warnings.

use iris::{compile_with_warnings, EmitKind};

// ---------------------------------------------------------------------------
// 1. Unused variable produces a warning
// ---------------------------------------------------------------------------
#[test]
fn test_warn_unused_var() {
    let src = r#"
def f() -> i64 {
    val unused = 99
    42
}
"#;
    let (result, warnings) = compile_with_warnings(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
    assert!(!warnings.is_empty(), "expected at least one warning");
    assert!(warnings.iter().any(|w| w.message.contains("unused")));
}

// ---------------------------------------------------------------------------
// 2. Used variable produces no warning
// ---------------------------------------------------------------------------
#[test]
fn test_no_warn_used_var() {
    let src = r#"
def f() -> i64 {
    val x = 21
    x * 2
}
"#;
    let (result, warnings) = compile_with_warnings(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
    assert!(warnings.is_empty(), "expected no warnings, got: {:?}", warnings);
}

// ---------------------------------------------------------------------------
// 3. Multiple unused variables
// ---------------------------------------------------------------------------
#[test]
fn test_warn_multiple_unused() {
    let src = r#"
def f() -> i64 {
    val a = 1
    val b = 2
    val c = 3
    c
}
"#;
    let (_result, warnings) = compile_with_warnings(src, "test", EmitKind::Eval).unwrap();
    let msgs: Vec<&str> = warnings.iter().map(|w| w.message.as_str()).collect();
    // 'a' and 'b' are unused; 'c' is used as the return value
    assert!(msgs.iter().any(|m| m.contains("'a'")), "expected warning for 'a'");
    assert!(msgs.iter().any(|m| m.contains("'b'")), "expected warning for 'b'");
    assert!(!msgs.iter().any(|m| m.contains("'c'")), "should NOT warn for used 'c'");
}

// ---------------------------------------------------------------------------
// 4. Underscore-prefixed variable suppresses warning
// ---------------------------------------------------------------------------
#[test]
fn test_no_warn_underscore_prefix() {
    let src = r#"
def f() -> i64 {
    val _intentionally_unused = 99
    42
}
"#;
    let (result, warnings) = compile_with_warnings(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
    assert!(warnings.is_empty(), "underscore-prefixed vars should not warn");
}

// ---------------------------------------------------------------------------
// 5. Variable used in a nested expression
// ---------------------------------------------------------------------------
#[test]
fn test_no_warn_used_in_expr() {
    let src = r#"
def f() -> i64 {
    val x = 10
    val y = 5
    x + y
}
"#;
    let (result, warnings) = compile_with_warnings(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15");
    assert!(warnings.is_empty(), "all vars are used");
}

// ---------------------------------------------------------------------------
// 6. Variable used in a when expression
// ---------------------------------------------------------------------------
#[test]
fn test_no_warn_used_in_when() {
    let src = r#"
def f() -> i64 {
    val x = 5
    when x > 3 {
        true  => x * 2,
        false => x,
    }
}
"#;
    let (result, warnings) = compile_with_warnings(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
    assert!(warnings.is_empty(), "x is used in when");
}

// ---------------------------------------------------------------------------
// 7. Warning includes function name
// ---------------------------------------------------------------------------
#[test]
fn test_warn_includes_function_name() {
    let src = r#"
def my_func() -> i64 {
    val dead = 0
    1
}
"#;
    let (_result, warnings) = compile_with_warnings(src, "test", EmitKind::Eval).unwrap();
    assert!(!warnings.is_empty());
    assert!(warnings.iter().any(|w| w.func == "my_func"));
}

// ---------------------------------------------------------------------------
// 8. Variable used in a function call argument
// ---------------------------------------------------------------------------
#[test]
fn test_no_warn_used_as_arg() {
    let src = r#"
def double(x: i64) -> i64 { x * 2 }
def f() -> i64 {
    val n = 21
    double(n)
}
"#;
    let (result, warnings) = compile_with_warnings(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
    assert!(warnings.is_empty(), "n is used as argument to double");
}
