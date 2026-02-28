//! Phase 75 integration tests: better error messages with source context.
//!
//! Verifies that `compile_with_diagnostics` produces human-readable output
//! with line numbers, source excerpts, and caret pointers on failure.

use iris::{compile_with_diagnostics, diagnostics, EmitKind};

// ---------------------------------------------------------------------------
// 1. Successful compilation returns Ok
// ---------------------------------------------------------------------------
#[test]
fn test_success_returns_ok() {
    let src = r#"
def f() -> i64 { 42 }
"#;
    let result = compile_with_diagnostics(src, "test", EmitKind::Eval);
    assert!(result.is_ok(), "expected Ok, got: {:?}", result);
    assert_eq!(result.unwrap().trim(), "42");
}

// ---------------------------------------------------------------------------
// 2. Parse error includes "error:" prefix
// ---------------------------------------------------------------------------
#[test]
fn test_parse_error_has_prefix() {
    let src = "def f() -> i64 { @bad }";
    let err = compile_with_diagnostics(src, "test", EmitKind::Eval).unwrap_err();
    assert!(err.starts_with("error:"), "expected 'error:' prefix, got: {}", err);
}

// ---------------------------------------------------------------------------
// 3. Parse error includes line number
// ---------------------------------------------------------------------------
#[test]
fn test_parse_error_has_line_number() {
    // The @bad token is on line 3
    let src = "\ndef f() -> i64 {\n    @bad\n}";
    let err = compile_with_diagnostics(src, "test", EmitKind::Eval).unwrap_err();
    // The diagnostic should mention line 3
    assert!(err.contains("3:"), "expected line 3 reference, got: {}", err);
}

// ---------------------------------------------------------------------------
// 4. Parse error includes --> location marker
// ---------------------------------------------------------------------------
#[test]
fn test_parse_error_has_location_marker() {
    let src = "def f() -> i64 { @bad }";
    let err = compile_with_diagnostics(src, "test", EmitKind::Eval).unwrap_err();
    assert!(err.contains("-->"), "expected '-->' in diagnostic, got: {}", err);
}

// ---------------------------------------------------------------------------
// 5. Parse error includes source line content
// ---------------------------------------------------------------------------
#[test]
fn test_parse_error_includes_source_line() {
    let src = "def f() -> i64 { @bad }";
    let err = compile_with_diagnostics(src, "test", EmitKind::Eval).unwrap_err();
    // The source line should appear in the diagnostic
    assert!(err.contains("@bad"), "expected source content in diagnostic, got: {}", err);
}

// ---------------------------------------------------------------------------
// 6. Parse error includes caret pointer
// ---------------------------------------------------------------------------
#[test]
fn test_parse_error_has_caret() {
    let src = "def f() -> i64 { @bad }";
    let err = compile_with_diagnostics(src, "test", EmitKind::Eval).unwrap_err();
    assert!(err.contains('^'), "expected caret '^' in diagnostic, got: {}", err);
}

// ---------------------------------------------------------------------------
// 7. Lower error (undefined variable) includes source context
// ---------------------------------------------------------------------------
#[test]
fn test_lower_error_has_context() {
    let src = r#"
def f() -> i64 {
    undefined_var
}
"#;
    let err = compile_with_diagnostics(src, "test", EmitKind::Eval).unwrap_err();
    assert!(err.starts_with("error:"), "expected error prefix");
    // Should mention the undefined variable
    assert!(err.contains("undefined_var"), "expected variable name in error, got: {}", err);
}

// ---------------------------------------------------------------------------
// 8. byte_to_line_col works correctly
// ---------------------------------------------------------------------------
#[test]
fn test_byte_to_line_col() {
    let source = "hello\nworld\nfoo";
    // 'w' is at byte 6 → line 2, col 1
    let (line, col) = diagnostics::byte_to_line_col(source, 6);
    assert_eq!(line, 2, "expected line 2");
    assert_eq!(col, 1, "expected col 1");

    // 'f' is at byte 12 → line 3, col 1
    let (line, col) = diagnostics::byte_to_line_col(source, 12);
    assert_eq!(line, 3, "expected line 3");
    assert_eq!(col, 1, "expected col 1");

    // 'o' (second) is at byte 14 → line 3, col 3
    let (line, col) = diagnostics::byte_to_line_col(source, 14);
    assert_eq!(line, 3, "expected line 3");
    assert_eq!(col, 3, "expected col 3");
}
