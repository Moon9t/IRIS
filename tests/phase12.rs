//! Phase 12 integration tests: diagnostics, CLI polish.

use std::path::PathBuf;

use iris::cli::{parse_args, ParseArgsResult};
use iris::diagnostics::{byte_to_line_col, render_error, span_to_line_col};
use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. byte_to_line_col — "abc\ndef\n", byte 4 → (2, 1)
// ---------------------------------------------------------------------------
#[test]
fn test_byte_to_line_col_basic() {
    let src = "abc\ndef\n";
    assert_eq!(byte_to_line_col(src, 4), (2, 1)); // 'd'
    assert_eq!(byte_to_line_col(src, 5), (2, 2)); // 'e'
    assert_eq!(byte_to_line_col(src, 6), (2, 3)); // 'f'
}

// ---------------------------------------------------------------------------
// 2. byte_to_line_col — "hello", byte 2 → (1, 3)
// ---------------------------------------------------------------------------
#[test]
fn test_byte_to_line_col_first_line() {
    let src = "hello";
    assert_eq!(byte_to_line_col(src, 0), (1, 1)); // 'h'
    assert_eq!(byte_to_line_col(src, 2), (1, 3)); // 'l'
    assert_eq!(byte_to_line_col(src, 4), (1, 5)); // 'o'
}

// ---------------------------------------------------------------------------
// 3. render_error contains '^'
// ---------------------------------------------------------------------------
#[test]
fn test_render_error_contains_caret() {
    // Force a parse error at a known location by putting '@' on line 1.
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail to parse");
    let rendered = render_error(src, &err);
    assert!(rendered.contains('^'), "expected '^' in:\n{}", rendered);
}

// ---------------------------------------------------------------------------
// 4. render_error contains the line number
// ---------------------------------------------------------------------------
#[test]
fn test_render_error_contains_line_number() {
    let src = "@bad";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error(src, &err);
    // Should contain something like "1:1" or " --> 1:"
    assert!(
        rendered.contains("1:") || rendered.contains(" 1 "),
        "expected line 1 reference in:\n{}",
        rendered
    );
}

// ---------------------------------------------------------------------------
// 5. Parse error for '@' on line 3 — rendered output references line 3
// ---------------------------------------------------------------------------
#[test]
fn test_parse_error_unknown_char_message() {
    // '@' is on line 3
    let src = "def a() -> f32 {\n  val x = 1.0\n  @bad\n}";
    let err = compile(src, "test", EmitKind::Ir).expect_err("should fail");
    let rendered = render_error(src, &err);
    assert!(
        rendered.contains("3:") || rendered.contains(" 3 "),
        "expected line 3 in rendered output:\n{}",
        rendered
    );
}

// ---------------------------------------------------------------------------
// 6. --help flag returns Help variant
// ---------------------------------------------------------------------------
#[test]
fn test_help_flag() {
    let args: Vec<String> = ["iris", "--help"].iter().map(|s| s.to_string()).collect();
    let result = parse_args(&args).expect("parse_args should not error");
    assert!(
        matches!(result, ParseArgsResult::Help),
        "expected Help variant"
    );
}

// ---------------------------------------------------------------------------
// 7. -o flag captures the output path
// ---------------------------------------------------------------------------
#[test]
fn test_output_flag() {
    let args: Vec<String> =
        ["iris", "-o", "out.ll", "file.iris"].iter().map(|s| s.to_string()).collect();
    let result = parse_args(&args).expect("parse_args should succeed");
    if let ParseArgsResult::Args(cli) = result {
        assert_eq!(cli.output, Some(PathBuf::from("out.ll")));
        assert_eq!(cli.path, PathBuf::from("file.iris"));
    } else {
        panic!("expected Args variant");
    }
}

// ---------------------------------------------------------------------------
// 8. span_to_line_col across multiple lines
// ---------------------------------------------------------------------------
#[test]
fn test_span_to_line_col_multiline() {
    // 5 lines:
    // line 1: "hello\n"       bytes 0-5   (6 bytes)
    // line 2: "world\n"       bytes 6-11  (6 bytes)
    // line 3: "foo\n"         bytes 12-15 (4 bytes)
    // line 4: "bar\n"         bytes 16-19 (4 bytes)
    // line 5: "baz"           bytes 20-22 (3 bytes)
    let src = "hello\nworld\nfoo\nbar\nbaz";

    assert_eq!(span_to_line_col(src, 0), (1, 1));  // 'h'
    assert_eq!(span_to_line_col(src, 5), (1, 6));  // '\n' position
    assert_eq!(span_to_line_col(src, 6), (2, 1));  // 'w'
    assert_eq!(span_to_line_col(src, 11), (2, 6)); // '\n'
    assert_eq!(span_to_line_col(src, 12), (3, 1)); // 'f'
    assert_eq!(span_to_line_col(src, 16), (4, 1)); // 'b'
    assert_eq!(span_to_line_col(src, 20), (5, 1)); // 'b' of "baz"
    assert_eq!(span_to_line_col(src, 22), (5, 3)); // 'z'
}
