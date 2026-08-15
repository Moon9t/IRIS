//! Source diagnostics: byte-to-line/col mapping and human-readable error rendering.
//!
//! Provides both plain-text and ANSI-colored rendering of compiler errors
//! in a rustc-style format with source excerpts, span underlines, error
//! codes, and optional help/hint notes.

use crate::error::{Error, InterpError, LowerError, ParseError, PassError};

// ---------------------------------------------------------------------------
// ANSI color helpers (no external dependency)
// ---------------------------------------------------------------------------

/// ANSI escape codes for terminal coloring.
mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const BOLD_RED: &str = "\x1b[1;31m";
    pub const BOLD_BLUE: &str = "\x1b[1;34m";
    pub const BOLD_GREEN: &str = "\x1b[1;32m";
}

// ---------------------------------------------------------------------------
// Core line/col mapping
// ---------------------------------------------------------------------------

/// Converts a byte offset within `source` to a 1-based `(line, col)` pair.
///
/// # Examples
/// ```text
/// "abc\ndef\n", byte 4  → (2, 1)   // 'd' is first char of line 2
/// "hello",     byte 2  → (1, 3)   // 'l' at column 3 on line 1
/// ```
pub fn byte_to_line_col(source: &str, byte: u32) -> (u32, u32) {
    let byte = byte as usize;
    let mut line = 1u32;
    let mut col = 1u32;
    for (i, ch) in source.char_indices() {
        if i == byte {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Returns the 1-based `(line, col)` for the start of the given byte span.
pub fn span_to_line_col(source: &str, start_byte: u32) -> (u32, u32) {
    byte_to_line_col(source, start_byte)
}

// ---------------------------------------------------------------------------
// Error span/byte extraction
// ---------------------------------------------------------------------------

/// Extracts the starting byte offset from errors that carry location info.
///
/// Returns `None` for errors without source position (pass errors, runtime errors).
pub fn error_byte_offset(err: &Error) -> Option<u32> {
    extract_byte(err)
}

fn extract_byte(err: &Error) -> Option<u32> {
    extract_span(err).map(|(start, _)| start)
}

/// Extracts the full `(start, end)` byte span from errors that carry one.
///
/// For errors that only store a single position (e.g. `UnexpectedChar`),
/// the returned span is one byte wide: `(pos, pos + 1)`.
fn extract_span(err: &Error) -> Option<(u32, u32)> {
    match err {
        Error::Parse(pe) => match pe {
            ParseError::UnexpectedChar { pos, .. } => Some((*pos, *pos + 1)),
            ParseError::UnterminatedString { pos, .. } => Some((*pos, *pos + 1)),
            ParseError::InvalidEscape { pos, .. } => Some((*pos, *pos + 2)),
            ParseError::InvalidLiteral { span, .. } => Some((span.start.0, span.end.0)),
            ParseError::UnexpectedToken { span, .. } => Some((span.start.0, span.end.0)),
            ParseError::UnexpectedEof { .. } => None,
        },
        Error::Lower(le) => match le {
            LowerError::UndefinedVariable { span, .. } => Some((span.start.0, span.end.0)),
            LowerError::TypeMismatch { span, .. } => Some((span.start.0, span.end.0)),
            LowerError::DuplicateFunction { span, .. } => Some((span.start.0, span.end.0)),
            LowerError::Unsupported { span, .. } => Some((span.start.0, span.end.0)),
            LowerError::Rejected { span, .. } => Some((span.start.0, span.end.0)),
            LowerError::UndefinedLayer { span, .. } => Some((span.start.0, span.end.0)),
            LowerError::DuplicateNode { span, .. } => Some((span.start.0, span.end.0)),
            LowerError::InvalidLayerParam { span, .. } => Some((span.start.0, span.end.0)),
            LowerError::UnknownOp { .. } => None,
        },
        Error::Interp(InterpError::Located { byte, byte_end, .. }) => {
            Some((*byte, byte_end.unwrap_or(*byte + 1)))
        }
        _ => None,
    }
}

/// Returns a contextual help note for common errors, or `None`.
fn error_hint(err: &Error) -> Option<&'static str> {
    match err {
        Error::Parse(pe) => match pe {
            ParseError::UnexpectedChar { ch: '@', .. } => {
                Some("IRIS does not use '@' — decorators are not supported")
            }
            ParseError::UnexpectedChar { ch: '#', .. } => {
                Some("comments in IRIS start with '//', not '#'")
            }
            ParseError::UnterminatedString { .. } => {
                Some("make sure every '\"' has a matching closing '\"'")
            }
            ParseError::InvalidEscape { .. } => {
                Some("valid escape sequences: \\n, \\t, \\r, \\\\, \\\"")
            }
            ParseError::UnexpectedEof { .. } => {
                Some("check for unmatched braces '{}' or parentheses '()'")
            }
            _ => None,
        },
        Error::Lower(le) => match le {
            LowerError::UndefinedVariable { name, .. } if name == "struct" => {
                Some("IRIS uses 'record' instead of 'struct'")
            }
            LowerError::UndefinedVariable { name, .. } if name == "enum" => {
                Some("IRIS uses 'choice' instead of 'enum'")
            }
            LowerError::UndefinedVariable { name, .. } if name == "match" => {
                Some("IRIS uses 'when' instead of 'match'")
            }
            LowerError::UndefinedVariable { name, .. } if name == "import" => {
                Some("IRIS uses 'bring \"file.iris\"' instead of 'import'")
            }
            LowerError::TypeMismatch { .. } => {
                Some("try adding an explicit type annotation to clarify the expected type")
            }
            LowerError::DuplicateFunction { .. } => {
                Some("rename one of the functions, or move it to a separate module")
            }
            _ => None,
        },
        Error::Pass(pe) => match pe {
            PassError::UnresolvedInfer { .. } => {
                Some("add a type annotation — the compiler cannot infer the type automatically")
            }
            PassError::MissingTerminator { .. } => {
                Some("every block must end with a return value or branch — check for missing 'return' or 'else' clauses")
            }
            _ => None,
        },
        Error::Interp(ie) => {
            // Unwrap Located to get hints for the inner error.
            let inner = match ie {
                InterpError::Located { inner, .. } => inner.as_ref(),
                other => other,
            };
            match inner {
                InterpError::DivisionByZero => {
                    Some("check that the divisor is not zero before dividing")
                }
                InterpError::IndexOutOfBounds { .. } => {
                    Some("use 'len(list)' to check bounds before indexing")
                }
                InterpError::TypeError { .. } => {
                    Some("check that operand types match — use `to_i64()`, `to_f64()`, or `to_str()` for conversions")
                }
                InterpError::Panic { .. } => {
                    Some("use `try` blocks to catch panics, or check conditions before the panicking operation")
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Plain-text rendering (backward-compatible)
// ---------------------------------------------------------------------------

/// Renders a rustc-style diagnostic for `err`, with a source excerpt, span
/// underline, error code, and optional help note.
///
/// ```text
/// error[E0001]: [syntax error] unexpected character '@' …
///  --> 3:3
///   |
/// 3 |   @invalid
///   |   ^
///   = help: IRIS does not use '@' — decorators are not supported
/// ```
pub fn render_error(source: &str, err: &Error) -> String {
    render_error_inner(source, err, None, false)
}

/// Like [`render_error`] but includes the filename in the location arrow.
///
/// ```text
///  --> src/main.iris:3:3
/// ```
pub fn render_error_with_file(source: &str, err: &Error, filename: &str) -> String {
    render_error_inner(source, err, Some(filename), false)
}

/// Renders a colored (ANSI) diagnostic for terminal output.
pub fn render_error_colored(source: &str, err: &Error) -> String {
    render_error_inner(source, err, None, true)
}

/// Renders a colored (ANSI) diagnostic with filename.
pub fn render_error_colored_with_file(source: &str, err: &Error, filename: &str) -> String {
    render_error_inner(source, err, Some(filename), true)
}

fn render_error_inner(source: &str, err: &Error, filename: Option<&str>, colored: bool) -> String {
    let code = err.diagnostic_code();
    let mut out = if colored {
        format!(
            "{}error[{}]{}: {}{}{}\n",
            ansi::BOLD_RED,
            code,
            ansi::RESET,
            ansi::BOLD,
            err,
            ansi::RESET,
        )
    } else {
        format!("error[{}]: {}\n", code, err)
    };

    if let Some((start_byte, end_byte)) = extract_span(err) {
        let (line, col) = byte_to_line_col(source, start_byte);
        let source_line = source.lines().nth((line - 1) as usize).unwrap_or("");

        // Compute underline width: clamp to the current source line
        let span_len = (end_byte.saturating_sub(start_byte)).max(1) as usize;
        // Don't underline past the end of the source line
        let max_underline = source_line
            .len()
            .saturating_sub((col as usize).saturating_sub(1));
        let underline_len = span_len.min(max_underline).max(1);

        let indent = (col as usize).saturating_sub(1);
        let pointer = format!("{}{}", " ".repeat(indent), "^".repeat(underline_len));
        let line_num = line.to_string();
        let gutter = " ".repeat(line_num.len());

        if colored {
            let loc = if let Some(f) = filename {
                format!("{}:{}:{}", f, line, col)
            } else {
                format!("{}:{}", line, col)
            };
            out.push_str(&format!(" {}-->{} {}\n", ansi::BOLD_BLUE, ansi::RESET, loc));
            out.push_str(&format!("{} {}|{}\n", gutter, ansi::BOLD_BLUE, ansi::RESET));
            out.push_str(&format!(
                "{}{} |{} {}\n",
                ansi::BOLD_BLUE,
                line_num,
                ansi::RESET,
                source_line
            ));
            out.push_str(&format!(
                "{} {}|{} {}{}{}\n",
                gutter,
                ansi::BOLD_BLUE,
                ansi::RESET,
                ansi::BOLD_RED,
                pointer,
                ansi::RESET,
            ));
        } else {
            let loc = if let Some(f) = filename {
                format!("{}:{}:{}", f, line, col)
            } else {
                format!("{}:{}", line, col)
            };
            out.push_str(&format!(" --> {}\n", loc));
            out.push_str(&format!("{}  |\n", gutter));
            out.push_str(&format!("{} | {}\n", line_num, source_line));
            out.push_str(&format!("{}  | {}\n", gutter, pointer));
        }
    }

    // Append help note if available
    if let Some(hint) = error_hint(err) {
        if colored {
            out.push_str(&format!(
                "   {}= help:{} {}\n",
                ansi::BOLD_GREEN,
                ansi::RESET,
                hint
            ));
        } else {
            out.push_str(&format!("   = help: {}\n", hint));
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ParseError;
    use crate::parser::lexer::{BytePos, Span};

    // -- byte_to_line_col -----------------------------------------------------

    #[test]
    fn byte_to_line_col_start_of_file() {
        assert_eq!(byte_to_line_col("hello", 0), (1, 1));
    }

    #[test]
    fn byte_to_line_col_first_line() {
        assert_eq!(byte_to_line_col("hello", 3), (1, 4));
    }

    #[test]
    fn byte_to_line_col_second_line() {
        assert_eq!(byte_to_line_col("abc\ndef", 4), (2, 1));
    }

    #[test]
    fn byte_to_line_col_second_line_middle() {
        assert_eq!(byte_to_line_col("abc\ndef", 5), (2, 2));
    }

    #[test]
    fn byte_to_line_col_third_line() {
        assert_eq!(byte_to_line_col("a\nb\nc", 4), (3, 1));
    }

    #[test]
    fn byte_to_line_col_empty_string() {
        assert_eq!(byte_to_line_col("", 0), (1, 1));
    }

    #[test]
    fn byte_to_line_col_newline_only() {
        assert_eq!(byte_to_line_col("\n", 1), (2, 1));
    }

    // -- span_to_line_col -----------------------------------------------------

    #[test]
    fn span_to_line_col_delegates() {
        assert_eq!(
            span_to_line_col("abc\ndef", 4),
            byte_to_line_col("abc\ndef", 4)
        );
    }

    // -- error_byte_offset ----------------------------------------------------

    #[test]
    fn error_byte_offset_parse_unexpected_char() {
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 42 });
        assert_eq!(error_byte_offset(&err), Some(42));
    }

    #[test]
    fn error_byte_offset_parse_eof() {
        let err = Error::Parse(ParseError::UnexpectedEof {
            context: "test".into(),
        });
        assert_eq!(error_byte_offset(&err), None);
    }

    #[test]
    fn error_byte_offset_lower_undefined() {
        let span = Span {
            start: BytePos(10),
            end: BytePos(15),
        };
        let err = Error::Lower(crate::error::LowerError::UndefinedVariable {
            name: "x".into(),
            span,
            suggestion: None,
        });
        assert_eq!(error_byte_offset(&err), Some(10));
    }

    #[test]
    fn error_byte_offset_pass_none() {
        let err = Error::Pass(crate::error::PassError::UseBeforeDef {
            func: "f".into(),
            value: "v".into(),
        });
        assert_eq!(error_byte_offset(&err), None);
    }

    #[test]
    fn error_byte_offset_interp_none() {
        let err = Error::Interp(crate::error::InterpError::DivisionByZero);
        assert_eq!(error_byte_offset(&err), None);
    }

    // -- render_error ---------------------------------------------------------

    #[test]
    fn render_error_with_location() {
        let src = "def main() {\n  @invalid\n}";
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 15 });
        let rendered = render_error(src, &err);
        assert!(rendered.contains("error[E0001]"));
        assert!(rendered.contains("-->"));
        assert!(rendered.contains("^"));
    }

    #[test]
    fn render_error_without_location() {
        let src = "def main() {}";
        let err = Error::Parse(ParseError::UnexpectedEof {
            context: "test".into(),
        });
        let rendered = render_error(src, &err);
        assert!(rendered.contains("error[E0006]"));
        assert!(!rendered.contains("-->"));
    }

    #[test]
    fn render_error_line_number_correct() {
        let src = "line1\nline2\nline3";
        // byte 12 = start of "line3"
        let err = Error::Parse(ParseError::UnexpectedChar { ch: 'x', pos: 12 });
        let rendered = render_error(src, &err);
        assert!(rendered.contains("3 |"));
    }

    // -- error codes in output ------------------------------------------------

    #[test]
    fn render_error_includes_error_code() {
        let src = "@bad";
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 0 });
        let rendered = render_error(src, &err);
        assert!(
            rendered.contains("error[E0001]"),
            "expected error code in output:\n{}",
            rendered
        );
    }

    // -- span underline -------------------------------------------------------

    #[test]
    fn render_error_underlines_span() {
        let src = "val x = badtoken";
        let span = Span {
            start: BytePos(8),
            end: BytePos(16), // "badtoken" = 8 chars
        };
        let err = Error::Parse(ParseError::InvalidLiteral {
            text: "badtoken".into(),
            span,
        });
        let rendered = render_error(src, &err);
        // Should contain "^^^^^^^^" (8 carets)
        assert!(
            rendered.contains("^^^^^^^^"),
            "expected multi-char underline in:\n{}",
            rendered
        );
    }

    #[test]
    fn render_error_single_char_underline() {
        let src = "@x";
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 0 });
        let rendered = render_error(src, &err);
        assert!(
            rendered.contains("^"),
            "expected single caret in:\n{}",
            rendered
        );
    }

    // -- filename in location -------------------------------------------------

    #[test]
    fn render_error_with_filename() {
        let src = "@bad";
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 0 });
        let rendered = render_error_with_file(src, &err, "test.iris");
        assert!(
            rendered.contains("test.iris:1:1"),
            "expected filename in location:\n{}",
            rendered
        );
    }

    // -- help hints -----------------------------------------------------------

    #[test]
    fn render_error_help_hint_at_sign() {
        let src = "@bad";
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 0 });
        let rendered = render_error(src, &err);
        assert!(
            rendered.contains("= help:"),
            "expected help note for '@':\n{}",
            rendered
        );
        assert!(rendered.contains("decorators"));
    }

    #[test]
    fn render_error_help_hint_hash() {
        let src = "# comment";
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '#', pos: 0 });
        let rendered = render_error(src, &err);
        assert!(rendered.contains("= help:"));
        assert!(rendered.contains("//"));
    }

    #[test]
    fn render_error_help_hint_unterminated_string() {
        let src = "\"hello";
        let err = Error::Parse(ParseError::UnterminatedString { pos: 0 });
        let rendered = render_error(src, &err);
        assert!(rendered.contains("= help:"));
    }

    // -- colored output -------------------------------------------------------

    #[test]
    fn render_error_colored_contains_ansi() {
        let src = "@bad";
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 0 });
        let rendered = render_error_colored(src, &err);
        assert!(
            rendered.contains("\x1b["),
            "expected ANSI escape codes in colored output:\n{}",
            rendered
        );
    }

    #[test]
    fn render_error_colored_with_filename() {
        let src = "@bad";
        let err = Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 0 });
        let rendered = render_error_colored_with_file(src, &err, "main.iris");
        assert!(rendered.contains("main.iris:1:1"));
        assert!(rendered.contains("\x1b["));
    }

    // -- extract_span ---------------------------------------------------------

    #[test]
    fn extract_span_parse_char() {
        let err = Error::Parse(ParseError::UnexpectedChar { ch: 'x', pos: 5 });
        assert_eq!(extract_span(&err), Some((5, 6)));
    }

    #[test]
    fn extract_span_lower_variable() {
        let span = Span {
            start: BytePos(10),
            end: BytePos(15),
        };
        let err = Error::Lower(LowerError::UndefinedVariable {
            name: "foo".into(),
            span,
            suggestion: None,
        });
        assert_eq!(extract_span(&err), Some((10, 15)));
    }

    #[test]
    fn extract_span_pass_returns_none() {
        let err = Error::Pass(PassError::UseBeforeDef {
            func: "f".into(),
            value: "v".into(),
        });
        assert_eq!(extract_span(&err), None);
    }

    // -- error_hint -----------------------------------------------------------

    #[test]
    fn hint_division_by_zero() {
        let err = Error::Interp(InterpError::DivisionByZero);
        assert!(error_hint(&err).is_some());
    }

    #[test]
    fn hint_index_out_of_bounds() {
        let err = Error::Interp(InterpError::IndexOutOfBounds { idx: 5, len: 3 });
        assert!(error_hint(&err).is_some());
    }

    #[test]
    fn hint_unresolved_infer() {
        let err = Error::Pass(PassError::UnresolvedInfer {
            func: "main".into(),
        });
        let h = error_hint(&err).unwrap();
        assert!(h.contains("type annotation"));
    }

    #[test]
    fn hint_missing_terminator() {
        let err = Error::Pass(PassError::MissingTerminator {
            func: "main".into(),
            block: "bb0".into(),
        });
        assert!(error_hint(&err).is_some());
    }

    #[test]
    fn hint_type_mismatch() {
        let span = Span {
            start: BytePos(0),
            end: BytePos(1),
        };
        let err = Error::Lower(LowerError::TypeMismatch {
            expected: "i64".into(),
            found: "str".into(),
            span,
        });
        assert!(error_hint(&err).is_some());
    }

    #[test]
    fn hint_none_for_generic_error() {
        let err = Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(error_hint(&err).is_none());
    }
}
