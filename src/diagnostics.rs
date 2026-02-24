//! Source diagnostics: byte-to-line/col mapping and human-readable error rendering.

use crate::error::{Error, LowerError, ParseError};

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

/// Extracts the starting byte offset from errors that carry location info.
fn extract_byte(err: &Error) -> Option<u32> {
    match err {
        Error::Parse(pe) => match pe {
            ParseError::UnexpectedChar { pos, .. } => Some(*pos),
            ParseError::UnterminatedString { pos, .. } => Some(*pos),
            ParseError::InvalidEscape { pos, .. } => Some(*pos),
            ParseError::InvalidLiteral { span, .. } => Some(span.start.0),
            ParseError::UnexpectedToken { span, .. } => Some(span.start.0),
            ParseError::UnexpectedEof { .. } => None,
        },
        Error::Lower(le) => match le {
            LowerError::UndefinedVariable { span, .. } => Some(span.start.0),
            LowerError::TypeMismatch { span, .. } => Some(span.start.0),
            LowerError::DuplicateFunction { span, .. } => Some(span.start.0),
            LowerError::Unsupported { span, .. } => Some(span.start.0),
            LowerError::UndefinedLayer { span, .. } => Some(span.start.0),
            LowerError::DuplicateNode { span, .. } => Some(span.start.0),
            LowerError::InvalidLayerParam { span, .. } => Some(span.start.0),
            LowerError::UnknownOp { .. } => None,
        },
        _ => None,
    }
}

/// Renders a rustc-style diagnostic for `err`, with a source excerpt and caret.
///
/// ```text
/// error: unexpected character '@' at byte 33
///  --> 3:3
///   |
/// 3 |   @invalid
///   |   ^
/// ```
pub fn render_error(source: &str, err: &Error) -> String {
    let mut out = format!("error: {}\n", err);

    if let Some(byte) = extract_byte(err) {
        let (line, col) = byte_to_line_col(source, byte);
        let source_line = source.lines().nth((line - 1) as usize).unwrap_or("");

        // Build pointer: col is 1-based, so col-1 spaces then '^'
        let indent = (col as usize).saturating_sub(1);
        let pointer = format!("{}{}", " ".repeat(indent), "^");
        let line_num = line.to_string();
        let gutter = " ".repeat(line_num.len());

        out.push_str(&format!(" --> {}:{}\n", line, col));
        out.push_str(&format!("{}  |\n", gutter));
        out.push_str(&format!("{} | {}\n", line_num, source_line));
        out.push_str(&format!("{}  | {}\n", gutter, pointer));
    }

    out
}
