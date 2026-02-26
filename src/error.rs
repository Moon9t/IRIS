use thiserror::Error;

use crate::parser::lexer::Span;

/// Top-level error type for the IRIS compiler pipeline.
#[derive(Debug, Error)]
pub enum Error {
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("lowering error: {0}")]
    Lower(#[from] LowerError),

    #[error("pass error: {0}")]
    Pass(#[from] PassError),

    #[error("codegen error: {0}")]
    Codegen(#[from] CodegenError),

    #[error("interpreter error: {0}")]
    Interp(#[from] InterpError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Parse errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("unexpected character '{ch}' at byte {pos}")]
    UnexpectedChar { ch: char, pos: u32 },

    #[error("unterminated string literal starting at byte {pos}")]
    UnterminatedString { pos: u32 },

    #[error("invalid escape sequence '{ch:?}' at byte {pos}")]
    InvalidEscape { ch: Option<char>, pos: u32 },

    #[error("invalid literal '{text}'")]
    InvalidLiteral { text: String, span: Span },

    #[error("expected {expected}, found '{found}' at {span:?}")]
    UnexpectedToken {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("unexpected end of file while parsing {context}")]
    UnexpectedEof { context: String },
}

// ---------------------------------------------------------------------------
// Lowering errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum LowerError {
    #[error("undefined variable '{name}' at {span:?}")]
    UndefinedVariable { name: String, span: Span },

    #[error("type mismatch: expected {expected}, found {found} at {span:?}")]
    TypeMismatch {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("function '{name}' already defined")]
    DuplicateFunction { name: String, span: Span },

    #[error("unsupported expression kind at {span:?}: {detail}")]
    Unsupported { detail: String, span: Span },

    #[error("undefined layer or input '{name}' at {span:?}")]
    UndefinedLayer { name: String, span: Span },

    #[error("duplicate node name '{name}' at {span:?}")]
    DuplicateNode { name: String, span: Span },

    #[error("invalid layer hyperparameter at {span:?}: {detail}")]
    InvalidLayerParam { detail: String, span: Span },

    #[error("no shape inference rule for layer op '{op}'")]
    UnknownOp { op: String },
}

// ---------------------------------------------------------------------------
// Pass errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum PassError {
    #[error("SSA violation in function '{func}': value {value} used before definition")]
    UseBeforeDef { func: String, value: String },

    #[error("SSA violation in function '{func}': value {value} defined more than once")]
    MultipleDefinition { func: String, value: String },

    #[error("type error in function '{func}': {detail}")]
    TypeError { func: String, detail: String },

    #[error("block '{block}' in function '{func}' has no terminator")]
    MissingTerminator { func: String, block: String },

    #[error("shape mismatch in function '{func}': {detail}")]
    ShapeMismatch { func: String, detail: String },

    #[error("unresolved type inference variable in function '{func}'")]
    UnresolvedInfer { func: String },
}

// ---------------------------------------------------------------------------
// Interpreter errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum InterpError {
    #[error("undefined value %{id}")]
    UndefinedValue { id: u32 },

    #[error("division by zero")]
    DivisionByZero,

    #[error("index {idx} out of bounds for length {len}")]
    IndexOutOfBounds { idx: i64, len: usize },

    #[error("type error: {detail}")]
    TypeError { detail: String },

    #[error("unsupported: {detail}")]
    Unsupported { detail: String },

    #[error("panic: {msg}")]
    Panic { msg: String },
}

// ---------------------------------------------------------------------------
// Codegen errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("unsupported IR construct in backend '{backend}': {detail}")]
    Unsupported { backend: String, detail: String },

    #[error("I/O error during code emission: {0}")]
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
