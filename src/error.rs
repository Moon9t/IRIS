use thiserror::Error;

use crate::parser::lexer::Span;

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

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("unexpected character '{ch}' — this character is not valid in IRIS source code (at byte {pos})")]
    UnexpectedChar { ch: char, pos: u32 },

    #[error("unterminated string — you opened a string literal but never closed it with a matching quote (at byte {pos})")]
    UnterminatedString { pos: u32 },

    #[error("invalid escape sequence '\\{escaped}' — valid escapes are \\n, \\t, \\r, \\\\ and \\\" (at byte {pos})", escaped = ch.map(|c| c.to_string()).unwrap_or_else(|| "EOF".into()))]
    InvalidEscape { ch: Option<char>, pos: u32 },

    #[error("invalid literal '{text}' — this does not look like a valid number, string, or boolean")]
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
    #[error("cannot find '{name}' — this variable or function is not defined in the current scope. Check for typos or make sure it is declared before use")]
    UndefinedVariable { name: String, span: Span },

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

    #[error("invalid layer parameter — {detail}. Check the documentation for valid hyperparameters")]
    InvalidLayerParam { detail: String, span: Span },

    #[error("unknown operation '{op}' — there is no shape inference rule for this layer type")]
    UnknownOp { op: String },
}

// ---------------------------------------------------------------------------
// Pass errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum PassError {
    #[error("in function '{func}': variable '{value}' is used before it has been assigned a value")]
    UseBeforeDef { func: String, value: String },

    #[error("in function '{func}': variable '{value}' is defined more than once — each variable can only be assigned once in SSA form")]
    MultipleDefinition { func: String, value: String },

    #[error("type error in function '{func}' — {detail}")]
    TypeError { func: String, detail: String },

    #[error("in function '{func}': block '{block}' does not end with a return or branch — every code path must have an explicit ending")]
    MissingTerminator { func: String, block: String },

    #[error("shape mismatch in function '{func}' — {detail}. Tensor dimensions must be compatible")]
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
            Error::Interp(_) => "E0400",
            Error::Io(_) => "E0500",
        }
    }
}
