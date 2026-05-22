//! Phase 128 integration tests: enhanced error diagnostics.
//!
//! Validates: error codes in output, span underlines, filename display,
//! colored rendering, and help hints.

use iris::diagnostics::{
    render_error, render_error_colored, render_error_colored_with_file, render_error_with_file,
};
use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Error codes appear in render_error output
// ---------------------------------------------------------------------------

#[test]
fn test_error_code_in_parse_error() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error(src, &err);
    // The parser recovers past '@' and reports UnexpectedToken (E0005)
    assert!(
        rendered.contains("[E0"),
        "parse error should include an error code:\n{}",
        rendered
    );
}

#[test]
fn test_error_code_in_lower_error() {
    let src = "def main() -> i64 { undefined_var }";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error(src, &err);
    // Should contain an error code like E0100 (UndefinedVariable)
    assert!(
        rendered.contains("[E01"),
        "lower error should include error code:\n{}",
        rendered
    );
}

// ---------------------------------------------------------------------------
// 2. Filename appears in render_error_with_file output
// ---------------------------------------------------------------------------

#[test]
fn test_render_error_shows_filename() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error_with_file(src, &err, "examples/test.iris");
    assert!(
        rendered.contains("examples/test.iris"),
        "filename should appear in output:\n{}",
        rendered
    );
}

#[test]
fn test_render_error_shows_filename_with_line_col() {
    // Put the error-causing token on line 1 so we know the line number
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error_with_file(src, &err, "test.iris");
    // Should contain "test.iris:LINE:COL"
    assert!(
        rendered.contains("test.iris:"),
        "should show filename:line:col:\n{}",
        rendered
    );
}

// ---------------------------------------------------------------------------
// 3. Colored output contains ANSI sequences
// ---------------------------------------------------------------------------

#[test]
fn test_colored_output_has_ansi_red() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error_colored(src, &err);
    // Bold red for "error"
    assert!(
        rendered.contains("\x1b[1;31m"),
        "colored output should contain bold-red:\n{:?}",
        rendered
    );
}

#[test]
fn test_colored_output_has_ansi_blue() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error_colored(src, &err);
    // Bold blue for line numbers and pipes
    assert!(
        rendered.contains("\x1b[1;34m"),
        "colored output should contain bold-blue:\n{:?}",
        rendered
    );
}

#[test]
fn test_colored_output_has_reset() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error_colored(src, &err);
    assert!(
        rendered.contains("\x1b[0m"),
        "colored output should contain ANSI reset:\n{:?}",
        rendered
    );
}

#[test]
fn test_colored_with_file_shows_both() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error_colored_with_file(src, &err, "main.iris");
    assert!(rendered.contains("main.iris"));
    assert!(rendered.contains("\x1b["));
}

// ---------------------------------------------------------------------------
// 4. Help hints appear for known error patterns
// ---------------------------------------------------------------------------

#[test]
fn test_help_hint_for_at_sign() {
    // Use a direct UnexpectedChar error to test the '@' hint
    use iris::error::ParseError;
    let src = "@decorator";
    let err = iris::error::Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 0 });
    let rendered = render_error(src, &err);
    assert!(
        rendered.contains("= help:"),
        "expected help hint for '@':\n{}",
        rendered
    );
}

#[test]
fn test_help_hint_for_hash_comment() {
    let src = "# this is a comment";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error(src, &err);
    assert!(
        rendered.contains("= help:"),
        "expected help hint for '#':\n{}",
        rendered
    );
    assert!(rendered.contains("//"));
}

// ---------------------------------------------------------------------------
// 5. Caret / underline still present
// ---------------------------------------------------------------------------

#[test]
fn test_caret_present() {
    let src = "@x";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error(src, &err);
    assert!(
        rendered.contains("^"),
        "expected caret underline:\n{}",
        rendered
    );
}

// ---------------------------------------------------------------------------
// 6. Plain (non-colored) output does NOT contain ANSI escapes
// ---------------------------------------------------------------------------

#[test]
fn test_plain_output_no_ansi() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error(src, &err);
    assert!(
        !rendered.contains("\x1b["),
        "plain render_error should not have ANSI:\n{:?}",
        rendered
    );
}

#[test]
fn test_plain_with_file_no_ansi() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error_with_file(src, &err, "test.iris");
    assert!(
        !rendered.contains("\x1b["),
        "plain render_error_with_file should not have ANSI:\n{:?}",
        rendered
    );
}
