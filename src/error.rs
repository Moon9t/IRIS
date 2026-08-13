use thiserror::Error;

use crate::parser::lexer::Span;

fn format_undef(name: &str, suggestion: Option<&str>) -> String {
    let base = format!(
        "cannot find '{}' — this variable or function is not defined in the current scope. \
         Check for typos or make sure it is declared before use",
        name
    );
    if let Some(s) = suggestion {
        format!(
            "{}
  help: did you mean '{}'?",
            base, s
        )
    } else {
        base
    }
}

/// Top-level error type for the IRIS compiler pipeline.
#[derive(Debug, Error)]
pub enum Error {
    #[error("{}", format_error_pretty("syntax error", &format!("{}", _0)))]
    Parse(#[from] ParseError),

    #[error("{}", format_error_pretty("compile error", &format!("{}", _0)))]
    Lower(#[from] LowerError),

    #[error("{}", format_error_pretty("type/pass error", &format!("{}", _0)))]
    Pass(#[from] PassError),

    #[error("{}", format_error_pretty("codegen error", &format!("{}", _0)))]
    Codegen(#[from] CodegenError),

    #[error("{}", format_error_pretty("runtime error", &format!("{}", _0)))]
    Interp(#[from] InterpError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{}", format_error_pretty("preprocessor error", _0))]
    Preprocessor(String),
}

/// Formats a compiler error in a human-friendly style.
fn format_error_pretty(category: &str, msg: &str) -> String {
    format!("[{}] {}", category, msg)
}

/// Utility: describes the byte offset as a human-readable source location.
pub fn describe_location(source: Option<&str>, byte: u32) -> String {
    if let Some(src) = source {
        let byte = byte as usize;
        let prefix = if byte <= src.len() { &src[..byte] } else { src };
        let line = prefix.bytes().filter(|&b| b == b'\n').count() + 1;
        let col = prefix.rfind('\n').map(|i| byte - i).unwrap_or(byte + 1);
        format!("line {}, column {}", line, col)
    } else {
        format!("byte {}", byte)
    }
}

// ---------------------------------------------------------------------------
// Parse errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Error)]
pub enum ParseError {
    #[error("unexpected character '{ch}' — this character is not valid in IRIS source code (at byte {pos})")]
    UnexpectedChar { ch: char, pos: u32 },

    #[error("unterminated string — you opened a string literal but never closed it with a matching quote (at byte {pos})")]
    UnterminatedString { pos: u32 },

    #[error("invalid escape sequence '\\{escaped}' — valid escapes are \\n, \\t, \\r, \\\\ and \\\" (at byte {pos})", escaped = ch.map(|c| c.to_string()).unwrap_or_else(|| "EOF".into()))]
    InvalidEscape { ch: Option<char>, pos: u32 },

    #[error(
        "invalid literal '{text}' — this does not look like a valid number, string, or boolean"
    )]
    InvalidLiteral { text: String, span: Span },

    #[error("expected {expected}, but found '{found}' — the compiler was looking for {expected} at this point")]
    UnexpectedToken {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("unexpected end of file while parsing {context} — you may be missing a closing brace '}}', parenthesis ')', or semicolon ';'")]
    UnexpectedEof { context: String },
}

// ---------------------------------------------------------------------------
// Lowering errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum LowerError {
    #[error("{}", format_undef(name, suggestion.as_deref()))]
    UndefinedVariable {
        name: String,
        span: Span,
        suggestion: Option<String>,
    },

    #[error("type mismatch — expected '{expected}' but found '{found}'. The types on both sides of this expression must agree")]
    TypeMismatch {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("'{name}' is already defined — each function name must be unique within its module. Consider renaming one of them")]
    DuplicateFunction { name: String, span: Span },

    #[error("unsupported expression — {detail}. This construct is not yet supported by the IRIS compiler")]
    Unsupported { detail: String, span: Span },

    #[error("cannot find layer or input '{name}' — make sure it is defined earlier in the model")]
    UndefinedLayer { name: String, span: Span },

    #[error("duplicate node name '{name}' — each node in a model must have a unique name")]
    DuplicateNode { name: String, span: Span },

    #[error(
        "invalid layer parameter — {detail}. Check the documentation for valid hyperparameters"
    )]
    InvalidLayerParam { detail: String, span: Span },

    #[error("unknown operation '{op}' — there is no shape inference rule for this layer type")]
    UnknownOp { op: String },
}

// ---------------------------------------------------------------------------
// Pass errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum PassError {
    #[error(
        "in function '{func}': variable '{value}' is used before it has been assigned a value"
    )]
    UseBeforeDef { func: String, value: String },

    #[error("in function '{func}': variable '{value}' is defined more than once — each variable can only be assigned once in SSA form")]
    MultipleDefinition { func: String, value: String },

    #[error("type error in function '{func}' — {detail}")]
    TypeError { func: String, detail: String },

    #[error("in function '{func}': block '{block}' does not end with a return or branch — every code path must have an explicit ending")]
    MissingTerminator { func: String, block: String },

    #[error(
        "shape mismatch in function '{func}' — {detail}. Tensor dimensions must be compatible"
    )]
    ShapeMismatch { func: String, detail: String },

    #[error("could not determine the type of a value in function '{func}' — try adding an explicit type annotation")]
    UnresolvedInfer { func: String },
}

// ---------------------------------------------------------------------------
// Interpreter errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum InterpError {
    #[error("internal error: undefined value %{id} — this is a compiler bug, please report it")]
    UndefinedValue { id: u32 },

    #[error("division by zero — cannot divide a number by zero")]
    DivisionByZero,

    #[error("index out of bounds — tried to access index {idx} but the collection only has {len} elements (valid indices: 0 to {max_idx})", max_idx = if *len > 0 { *len as i64 - 1 } else { 0 })]
    IndexOutOfBounds { idx: i64, len: usize },

    #[error("type error — {detail}")]
    TypeError { detail: String },

    #[error("not yet supported — {detail}")]
    Unsupported { detail: String },

    #[error("program panicked: {msg}")]
    Panic { msg: String },

    #[error("{0}")]
    SecurityViolation(#[from] crate::security::SecurityError),

    /// Wraps any runtime error with the source byte offset of the instruction
    /// that triggered it, enabling source-excerpt rendering in diagnostics.
    #[error("{inner}")]
    Located {
        inner: Box<InterpError>,
        /// Byte offset into the original source of the failing instruction.
        byte: u32,
        /// Optional end byte offset for wider underlines
        byte_end: Option<u32>,
        /// Name of the function in which the error occurred.
        func: String,
    },
}

// ---------------------------------------------------------------------------
// Codegen errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("the {backend} backend does not support this construct — {detail}")]
    Unsupported { backend: String, detail: String },

    #[error("I/O error during code generation: {0}")]
    Io(#[from] std::io::Error),
}

impl From<std::fmt::Error> for CodegenError {
    fn from(e: std::fmt::Error) -> Self {
        CodegenError::Unsupported {
            backend: "codegen".into(),
            detail: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Error byte-offset extraction (for LSP and CLI error reporting)
// ---------------------------------------------------------------------------

impl Error {
    /// Returns a diagnostic code string for this error (used by LSP).
    pub fn diagnostic_code(&self) -> &'static str {
        match self {
            Error::Parse(p) => match p {
                ParseError::UnexpectedChar { .. } => "E0001",
                ParseError::UnterminatedString { .. } => "E0002",
                ParseError::InvalidEscape { .. } => "E0003",
                ParseError::InvalidLiteral { .. } => "E0004",
                ParseError::UnexpectedToken { .. } => "E0005",
                ParseError::UnexpectedEof { .. } => "E0006",
            },
            Error::Lower(l) => match l {
                LowerError::UndefinedVariable { .. } => "E0100",
                LowerError::TypeMismatch { .. } => "E0101",
                LowerError::DuplicateFunction { .. } => "E0102",
                LowerError::Unsupported { .. } => "E0103",
                LowerError::UndefinedLayer { .. } => "E0104",
                LowerError::DuplicateNode { .. } => "E0105",
                LowerError::InvalidLayerParam { .. } => "E0106",
                LowerError::UnknownOp { .. } => "E0107",
            },
            Error::Pass(p) => match p {
                PassError::UseBeforeDef { .. } => "E0200",
                PassError::MultipleDefinition { .. } => "E0201",
                PassError::TypeError { .. } => "E0202",
                PassError::MissingTerminator { .. } => "E0203",
                PassError::ShapeMismatch { .. } => "E0204",
                PassError::UnresolvedInfer { .. } => "E0205",
            },
            Error::Codegen(_) => "E0300",
            Error::Interp(ie) => {
                // Unwrap Located to get the code of the inner error.
                let inner = match ie {
                    InterpError::Located { inner, .. } => inner.as_ref(),
                    other => other,
                };
                match inner {
                    InterpError::UndefinedValue { .. } => "E0401",
                    InterpError::DivisionByZero => "E0402",
                    InterpError::IndexOutOfBounds { .. } => "E0403",
                    InterpError::TypeError { .. } => "E0404",
                    InterpError::Unsupported { .. } => "E0405",
                    InterpError::Panic { .. } => "E0406",
                    InterpError::SecurityViolation(_) => "E0407",
                    InterpError::Located { .. } => "E0400",
                }
            }
            Error::Io(_) => "E0500",
            Error::Preprocessor(_) => "E0501",
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span {
            start: crate::parser::lexer::BytePos(0),
            end: crate::parser::lexer::BytePos(1),
        }
    }

    // -- format_error_pretty --------------------------------------------------

    #[test]
    fn format_error_pretty_basic() {
        let s = format_error_pretty("syntax error", "unexpected '@'");
        assert_eq!(s, "[syntax error] unexpected '@'");
    }

    // -- describe_location ----------------------------------------------------

    #[test]
    fn describe_location_with_source() {
        let src = "abc\ndef\nghi";
        assert_eq!(describe_location(Some(src), 0), "line 1, column 1");
        assert_eq!(describe_location(Some(src), 4), "line 2, column 1");
        assert_eq!(describe_location(Some(src), 6), "line 2, column 3");
    }

    #[test]
    fn describe_location_no_source() {
        assert_eq!(describe_location(None, 42), "byte 42");
    }

    #[test]
    fn describe_location_past_end() {
        let src = "ab";
        // byte offset past end — uses full source length
        let result = describe_location(Some(src), 100);
        assert!(result.contains("line"));
    }

    // -- ParseError Display ---------------------------------------------------

    #[test]
    fn parse_error_unexpected_char() {
        let e = ParseError::UnexpectedChar { ch: '@', pos: 5 };
        let msg = format!("{}", e);
        assert!(msg.contains("@"));
        assert!(msg.contains("byte 5"));
    }

    #[test]
    fn parse_error_unterminated_string() {
        let e = ParseError::UnterminatedString { pos: 10 };
        let msg = format!("{}", e);
        assert!(msg.contains("unterminated string"));
    }

    #[test]
    fn parse_error_invalid_escape() {
        let e = ParseError::InvalidEscape {
            ch: Some('q'),
            pos: 3,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("\\q"));
    }

    #[test]
    fn parse_error_invalid_escape_eof() {
        let e = ParseError::InvalidEscape { ch: None, pos: 3 };
        let msg = format!("{}", e);
        assert!(msg.contains("EOF"));
    }

    #[test]
    fn parse_error_unexpected_token() {
        let e = ParseError::UnexpectedToken {
            expected: "';'".into(),
            found: "def".into(),
            span: dummy_span(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("';'"));
        assert!(msg.contains("def"));
    }

    #[test]
    fn parse_error_unexpected_eof() {
        let e = ParseError::UnexpectedEof {
            context: "function body".into(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("function body"));
    }

    // -- LowerError Display ---------------------------------------------------

    #[test]
    fn lower_error_undefined_variable() {
        let e = LowerError::UndefinedVariable {
            name: "foo".into(),
            span: dummy_span(),
            suggestion: None,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("foo"));
        assert!(msg.contains("not defined"));
    }

    #[test]
    fn lower_error_type_mismatch() {
        let e = LowerError::TypeMismatch {
            expected: "i64".into(),
            found: "str".into(),
            span: dummy_span(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("i64"));
        assert!(msg.contains("str"));
    }

    #[test]
    fn lower_error_duplicate_function() {
        let e = LowerError::DuplicateFunction {
            name: "main".into(),
            span: dummy_span(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("main"));
        assert!(msg.contains("already defined"));
    }

    // -- PassError Display ----------------------------------------------------

    #[test]
    fn pass_error_use_before_def() {
        let e = PassError::UseBeforeDef {
            func: "foo".into(),
            value: "%5".into(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("foo"));
        assert!(msg.contains("%5"));
    }

    #[test]
    fn pass_error_missing_terminator() {
        let e = PassError::MissingTerminator {
            func: "main".into(),
            block: "bb0".into(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("main"));
        assert!(msg.contains("bb0"));
    }

    // -- InterpError Display --------------------------------------------------

    #[test]
    fn interp_error_division_by_zero() {
        let e = InterpError::DivisionByZero;
        let msg = format!("{}", e);
        assert!(msg.contains("division by zero"));
    }

    #[test]
    fn interp_error_index_out_of_bounds() {
        let e = InterpError::IndexOutOfBounds { idx: 10, len: 5 };
        let msg = format!("{}", e);
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn interp_error_index_out_of_bounds_empty() {
        let e = InterpError::IndexOutOfBounds { idx: 0, len: 0 };
        let msg = format!("{}", e);
        assert!(msg.contains("0 elements"));
    }

    // -- CodegenError Display -------------------------------------------------

    #[test]
    fn codegen_error_unsupported() {
        let e = CodegenError::Unsupported {
            backend: "LLVM".into(),
            detail: "closures".into(),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("LLVM"));
        assert!(msg.contains("closures"));
    }

    #[test]
    fn codegen_error_from_fmt() {
        let e: CodegenError = std::fmt::Error.into();
        let msg = format!("{}", e);
        assert!(msg.contains("codegen"));
    }

    // -- Error wrapper Display ------------------------------------------------

    #[test]
    fn error_wraps_parse() {
        let e = Error::Parse(ParseError::UnexpectedChar { ch: '#', pos: 0 });
        let msg = format!("{}", e);
        assert!(msg.contains("[syntax error]"));
    }

    #[test]
    fn error_wraps_lower() {
        let e = Error::Lower(LowerError::UnknownOp {
            op: "conv3d".into(),
        });
        let msg = format!("{}", e);
        assert!(msg.contains("[compile error]"));
    }

    #[test]
    fn error_wraps_interp() {
        let e = Error::Interp(InterpError::DivisionByZero);
        let msg = format!("{}", e);
        assert!(msg.contains("[runtime error]"));
    }

    // -- Diagnostic codes -----------------------------------------------------

    #[test]
    fn diagnostic_code_parse() {
        assert_eq!(
            Error::Parse(ParseError::UnexpectedChar { ch: '@', pos: 0 }).diagnostic_code(),
            "E0001"
        );
        assert_eq!(
            Error::Parse(ParseError::UnterminatedString { pos: 0 }).diagnostic_code(),
            "E0002"
        );
        assert_eq!(
            Error::Parse(ParseError::InvalidEscape { ch: None, pos: 0 }).diagnostic_code(),
            "E0003"
        );
        assert_eq!(
            Error::Parse(ParseError::InvalidLiteral {
                text: "".into(),
                span: dummy_span()
            })
            .diagnostic_code(),
            "E0004"
        );
        assert_eq!(
            Error::Parse(ParseError::UnexpectedToken {
                expected: "".into(),
                found: "".into(),
                span: dummy_span()
            })
            .diagnostic_code(),
            "E0005"
        );
        assert_eq!(
            Error::Parse(ParseError::UnexpectedEof { context: "".into() }).diagnostic_code(),
            "E0006"
        );
    }

    #[test]
    fn diagnostic_code_lower() {
        assert_eq!(
            Error::Lower(LowerError::UndefinedVariable {
                name: "".into(),
                span: dummy_span(),
                suggestion: None,
            })
            .diagnostic_code(),
            "E0100"
        );
        assert_eq!(
            Error::Lower(LowerError::UnknownOp { op: "".into() }).diagnostic_code(),
            "E0107"
        );
    }

    #[test]
    fn diagnostic_code_pass() {
        assert_eq!(
            Error::Pass(PassError::UseBeforeDef {
                func: "".into(),
                value: "".into()
            })
            .diagnostic_code(),
            "E0200"
        );
        assert_eq!(
            Error::Pass(PassError::UnresolvedInfer { func: "".into() }).diagnostic_code(),
            "E0205"
        );
    }

    #[test]
    fn diagnostic_code_codegen() {
        assert_eq!(
            Error::Codegen(CodegenError::Unsupported {
                backend: "".into(),
                detail: "".into()
            })
            .diagnostic_code(),
            "E0300"
        );
    }

    #[test]
    fn diagnostic_code_interp() {
        assert_eq!(
            Error::Interp(InterpError::DivisionByZero).diagnostic_code(),
            "E0402"
        );
        assert_eq!(
            Error::Interp(InterpError::IndexOutOfBounds { idx: 0, len: 0 }).diagnostic_code(),
            "E0403"
        );
        // Located unwraps to the inner error code.
        assert_eq!(
            Error::Interp(InterpError::Located {
                inner: Box::new(InterpError::DivisionByZero),
                byte: 10,
                byte_end: None,
                func: "f".into(),
            })
            .diagnostic_code(),
            "E0402"
        );
    }

    #[test]
    fn diagnostic_code_io() {
        let e = Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert_eq!(e.diagnostic_code(), "E0500");
    }
}
