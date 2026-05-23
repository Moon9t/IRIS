use crate::error::ParseError;

/// A byte offset within the source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BytePos(pub u32);

/// A half-open `[start, end)` byte range in the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: BytePos,
    pub end: BytePos,
}

impl Span {
    pub fn new(start: u32, end: u32) -> Self {
        Self {
            start: BytePos(start),
            end: BytePos(end),
        }
    }

    /// Creates a zero-length span at position `pos` (for synthetic nodes).
    pub fn at(pos: u32) -> Self {
        Self::new(pos, pos)
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Def,
    Val,
    Var,
    Return,
    If,
    Else,
    While,
    Loop,
    Break,
    Continue,
    // Record (struct) keyword
    Record,
    // Reserved for future use
    Bring,
    When,
    Choice,
    /// `for` keyword in range loops
    For,
    /// `in` keyword in range loops
    In,
    /// `spawn` keyword for concurrent task creation
    Spawn,
    /// `par` keyword for parallel for-loop
    Par,
    /// `async` keyword for async function definitions
    Async,
    /// `await` keyword for awaiting async expressions
    Await,
    /// `const` keyword for global constant declarations
    Const,
    /// `type` keyword for type alias declarations
    Type,
    /// `trait` keyword for trait definitions
    Trait,
    /// `impl` keyword for trait implementations
    Impl,
    /// `pub` visibility modifier
    Pub,
    /// `extern` keyword for FFI declarations
    Extern,
    /// `mod` keyword for inline module blocks
    Mod,
    // Model DSL keywords
    Model,
    Layer,
    Input,
    Output,

    // Type keywords
    F32,
    F64,
    I32,
    I64,
    Bool,
    Tensor,
    /// `str` type keyword
    Str,

    // Literals
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),
    Ident(String),

    // Punctuation
    LParen,   // (
    RParen,   // )
    LBrace,   // {
    RBrace,   // }
    LBracket, // [
    RBracket, // ]
    LAngle,   // <
    RAngle,   // >
    Comma,    // ,
    Colon,    // :
    Semi,     // ;
    Arrow,    // ->
    Eq,       // =
    EqEq,     // ==
    NotEq,    // !=
    LtEq,     // <=
    GtEq,     // >=

    // Arithmetic operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent, // %

    /// `!` (logical NOT prefix)
    Bang,

    /// `to` cast operator
    To,

    /// `.` field access
    Dot,

    /// `..` range separator used in `for i in start..end`
    DotDot,

    /// `..=` inclusive range used in range patterns
    DotDotEq,

    /// `=>` fat arrow used in `when` arms
    FatArrow,

    /// `&&` logical AND
    AmpAmp,
    /// `||` logical OR
    PipePipe,
    /// `|` single pipe (for lambda parameter delimiters)
    Pipe,

    /// `?` try / error-propagation operator
    Question,

    /// `@` attribute sigil (e.g. `@differentiable`)
    At,

    /// `f"..."` string with `{ident}` interpolation placeholders.
    /// The payload is the raw content (with `{...}` markers preserved).
    FStringLit(String),

    Eof,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Def => write!(f, "def"),
            Token::Val => write!(f, "val"),
            Token::Var => write!(f, "var"),
            Token::Return => write!(f, "return"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::While => write!(f, "while"),
            Token::Loop => write!(f, "loop"),
            Token::Break => write!(f, "break"),
            Token::Continue => write!(f, "continue"),
            Token::Record => write!(f, "record"),
            Token::Trait => write!(f, "trait"),
            Token::Impl => write!(f, "impl"),
            Token::Pub => write!(f, "pub"),
            Token::Extern => write!(f, "extern"),
            Token::Mod => write!(f, "mod"),
            Token::Bring => write!(f, "bring"),
            Token::When => write!(f, "when"),
            Token::Choice => write!(f, "choice"),
            Token::For => write!(f, "for"),
            Token::In => write!(f, "in"),
            Token::Spawn => write!(f, "spawn"),
            Token::Par => write!(f, "par"),
            Token::Async => write!(f, "async"),
            Token::Await => write!(f, "await"),
            Token::Const => write!(f, "const"),
            Token::Type => write!(f, "type"),
            Token::Model => write!(f, "model"),
            Token::Layer => write!(f, "layer"),
            Token::Input => write!(f, "input"),
            Token::Output => write!(f, "output"),
            Token::F32 => write!(f, "f32"),
            Token::F64 => write!(f, "f64"),
            Token::I32 => write!(f, "i32"),
            Token::I64 => write!(f, "i64"),
            Token::Bool => write!(f, "bool"),
            Token::Tensor => write!(f, "tensor"),
            Token::Str => write!(f, "str"),
            Token::IntLit(n) => write!(f, "{}", n),
            Token::FloatLit(n) => write!(f, "{}", n),
            Token::BoolLit(b) => write!(f, "{}", b),
            Token::StringLit(s) => write!(f, "\"{}\"", s),
            Token::Ident(s) => write!(f, "{}", s),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LAngle => write!(f, "<"),
            Token::RAngle => write!(f, ">"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semi => write!(f, ";"),
            Token::Arrow => write!(f, "->"),
            Token::Eq => write!(f, "="),
            Token::EqEq => write!(f, "=="),
            Token::NotEq => write!(f, "!="),
            Token::LtEq => write!(f, "<="),
            Token::GtEq => write!(f, ">="),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::Bang => write!(f, "!"),
            Token::To => write!(f, "to"),
            Token::Dot => write!(f, "."),
            Token::DotDot => write!(f, ".."),
            Token::DotDotEq => write!(f, "..="),
            Token::FatArrow => write!(f, "=>"),
            Token::AmpAmp => write!(f, "&&"),
            Token::PipePipe => write!(f, "||"),
            Token::Pipe => write!(f, "|"),
            Token::Question => write!(f, "?"),
            Token::At => write!(f, "@"),
            Token::FStringLit(s) => write!(f, "f\"{}\"", s),
            Token::Eof => write!(f, "<eof>"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

pub struct Lexer<'src> {
    src: &'src str,
    pos: usize,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Self { src, pos: 0 }
    }

    /// Tokenizes the full source and returns a flat `Vec` of spanned tokens.
    /// Returns an error on any unrecognized character.
    pub fn tokenize(&mut self) -> Result<Vec<Spanned<Token>>, ParseError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.src.len() {
                let end = self.pos as u32;
                tokens.push(Spanned {
                    node: Token::Eof,
                    span: Span::at(end),
                });
                break;
            }
            let tok = self.next_token()?;
            tokens.push(tok);
        }
        Ok(tokens)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.pos < self.src.len() && self.src.as_bytes()[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.src[self.pos..].starts_with("//") {
                // Line comment: skip to end of line.
                while self.pos < self.src.len() && self.src.as_bytes()[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else if self.src[self.pos..].starts_with("/*") {
                // Block comment: skip until matching `*/`.  Supports nesting.
                self.pos += 2;
                let mut depth = 1usize;
                while self.pos < self.src.len() && depth > 0 {
                    if self.src[self.pos..].starts_with("/*") {
                        depth += 1;
                        self.pos += 2;
                    } else if self.src[self.pos..].starts_with("*/") {
                        depth -= 1;
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> u8 {
        let b = self.src.as_bytes()[self.pos];
        self.pos += 1;
        b
    }

    fn next_token(&mut self) -> Result<Spanned<Token>, ParseError> {
        let start = self.pos as u32;
        let Some(ch) = self.peek() else {
            return Err(ParseError::UnexpectedEof {
                context: "lexer: unexpected end of input".into(),
            });
        };

        // Two-character tokens
        if ch == b'-' && self.peek2() == Some(b'>') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::Arrow,
                span: Span::new(start, self.pos as u32),
            });
        }
        if ch == b'=' && self.peek2() == Some(b'=') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::EqEq,
                span: Span::new(start, self.pos as u32),
            });
        }
        if ch == b'=' && self.peek2() == Some(b'>') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::FatArrow,
                span: Span::new(start, self.pos as u32),
            });
        }
        if ch == b'!' && self.peek2() == Some(b'=') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::NotEq,
                span: Span::new(start, self.pos as u32),
            });
        }
        if ch == b'<' && self.peek2() == Some(b'=') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::LtEq,
                span: Span::new(start, self.pos as u32),
            });
        }
        if ch == b'>' && self.peek2() == Some(b'=') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::GtEq,
                span: Span::new(start, self.pos as u32),
            });
        }

        // `&&` logical AND
        if ch == b'&' && self.peek2() == Some(b'&') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::AmpAmp,
                span: Span::new(start, self.pos as u32),
            });
        }
        // `||` logical OR
        if ch == b'|' && self.peek2() == Some(b'|') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::PipePipe,
                span: Span::new(start, self.pos as u32),
            });
        }

        // `..=` inclusive range — must come before `..`
        if ch == b'.'
            && self.peek2() == Some(b'.')
            && self.src.as_bytes().get(self.pos + 2) == Some(&b'=')
        {
            self.pos += 3;
            return Ok(Spanned {
                node: Token::DotDotEq,
                span: Span::new(start, self.pos as u32),
            });
        }
        // `..` range separator — must come before single `.`
        if ch == b'.' && self.peek2() == Some(b'.') {
            self.pos += 2;
            return Ok(Spanned {
                node: Token::DotDot,
                span: Span::new(start, self.pos as u32),
            });
        }

        // Single-character punctuation
        let maybe_punct = match ch {
            b'(' => Some(Token::LParen),
            b')' => Some(Token::RParen),
            b'{' => Some(Token::LBrace),
            b'}' => Some(Token::RBrace),
            b'[' => Some(Token::LBracket),
            b']' => Some(Token::RBracket),
            b'<' => Some(Token::LAngle),
            b'>' => Some(Token::RAngle),
            b',' => Some(Token::Comma),
            b':' => Some(Token::Colon),
            b';' => Some(Token::Semi),
            b'=' => Some(Token::Eq),
            b'+' => Some(Token::Plus),
            b'-' => Some(Token::Minus),
            b'*' => Some(Token::Star),
            b'/' => Some(Token::Slash),
            b'%' => Some(Token::Percent),
            b'!' => Some(Token::Bang),
            b'.' => Some(Token::Dot),
            b'|' => Some(Token::Pipe),
            b'?' => Some(Token::Question),
            b'@' => Some(Token::At),
            _ => None,
        };
        if let Some(tok) = maybe_punct {
            self.pos += 1;
            return Ok(Spanned {
                node: tok,
                span: Span::new(start, self.pos as u32),
            });
        }

        if ch == b'"' {
            return self.lex_string(start);
        }

        if ch.is_ascii_digit() {
            return self.lex_number(start);
        }

        if ch.is_ascii_alphabetic() || ch == b'_' {
            // Detect f"..." string interpolation prefix before normal identifier
            if ch == b'f' && self.peek2() == Some(b'"') {
                self.pos += 1; // consume 'f'
                return self.lex_fstring(start);
            }
            return Ok(self.lex_ident_or_keyword(start));
        }

        Err(ParseError::UnexpectedChar {
            ch: ch as char,
            pos: start,
        })
    }

    fn lex_string(&mut self, start: u32) -> Result<Spanned<Token>, ParseError> {
        self.advance(); // consume opening `"`
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err(ParseError::UnterminatedString { pos: start }),
                Some(b'"') => {
                    self.advance();
                    break;
                }
                Some(b'\\') => {
                    self.advance();
                    match self.peek() {
                        Some(b'n') => {
                            self.advance();
                            s.push('\n');
                        }
                        Some(b't') => {
                            self.advance();
                            s.push('\t');
                        }
                        Some(b'r') => {
                            self.advance();
                            s.push('\r');
                        }
                        Some(b'"') => {
                            self.advance();
                            s.push('"');
                        }
                        Some(b'\\') => {
                            self.advance();
                            s.push('\\');
                        }
                        other => {
                            return Err(ParseError::InvalidEscape {
                                ch: other.map(|b| b as char),
                                pos: self.pos as u32,
                            });
                        }
                    }
                }
                Some(b) => {
                    self.advance();
                    s.push(b as char);
                }
            }
        }
        Ok(Spanned {
            node: Token::StringLit(s),
            span: Span::new(start, self.pos as u32),
        })
    }

    /// Lex an f-string literal `f"..."` — the `f` has already been consumed.
    /// The raw content (including `{...}` markers) is stored verbatim.
    fn lex_fstring(&mut self, start: u32) -> Result<Spanned<Token>, ParseError> {
        self.advance(); // consume opening `"`
        let mut raw = String::new();
        loop {
            match self.peek() {
                None => return Err(ParseError::UnterminatedString { pos: start }),
                Some(b'"') => {
                    self.advance();
                    break;
                }
                Some(b'\\') => {
                    self.advance();
                    match self.peek() {
                        Some(b'n') => {
                            self.advance();
                            raw.push('\n');
                        }
                        Some(b't') => {
                            self.advance();
                            raw.push('\t');
                        }
                        Some(b'r') => {
                            self.advance();
                            raw.push('\r');
                        }
                        Some(b'"') => {
                            self.advance();
                            raw.push('"');
                        }
                        Some(b'\\') => {
                            self.advance();
                            raw.push('\\');
                        }
                        other => {
                            return Err(ParseError::InvalidEscape {
                                ch: other.map(|b| b as char),
                                pos: self.pos as u32,
                            });
                        }
                    }
                }
                Some(b) => {
                    self.advance();
                    raw.push(b as char);
                }
            }
        }
        Ok(Spanned {
            node: Token::FStringLit(raw),
            span: Span::new(start, self.pos as u32),
        })
    }

    fn lex_number(&mut self, start: u32) -> Result<Spanned<Token>, ParseError> {
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.advance();
        }
        let is_float =
            self.peek() == Some(b'.') && self.peek2().is_some_and(|b| b.is_ascii_digit());
        if is_float {
            self.advance(); // consume '.'
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.advance();
            }
            // Optional exponent: e/E followed by optional +/- and digits
            if self.peek().is_some_and(|b| b == b'e' || b == b'E') {
                self.advance();
                if self.peek().is_some_and(|b| b == b'+' || b == b'-') {
                    self.advance();
                }
                while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                    self.advance();
                }
            }
            let text = &self.src[start as usize..self.pos];
            let value: f64 = text.parse().map_err(|_| ParseError::InvalidLiteral {
                text: text.to_owned(),
                span: Span::new(start, self.pos as u32),
            })?;
            Ok(Spanned {
                node: Token::FloatLit(value),
                span: Span::new(start, self.pos as u32),
            })
        } else {
            let text = &self.src[start as usize..self.pos];
            let value: i64 = text.parse().map_err(|_| ParseError::InvalidLiteral {
                text: text.to_owned(),
                span: Span::new(start, self.pos as u32),
            })?;
            Ok(Spanned {
                node: Token::IntLit(value),
                span: Span::new(start, self.pos as u32),
            })
        }
    }

    fn lex_ident_or_keyword(&mut self, start: u32) -> Spanned<Token> {
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            self.advance();
        }
        let text = &self.src[start as usize..self.pos];
        let tok = match text {
            "def" => Token::Def,
            "val" => Token::Val,
            "var" => Token::Var,
            "return" => Token::Return,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "loop" => Token::Loop,
            "break" => Token::Break,
            "continue" => Token::Continue,
            "record" => Token::Record,
            "bring" => Token::Bring,
            "when" => Token::When,
            "choice" => Token::Choice,
            "for" => Token::For,
            "in" => Token::In,
            "to" => Token::To,
            "spawn" => Token::Spawn,
            "par" => Token::Par,
            "async" => Token::Async,
            "await" => Token::Await,
            "const" => Token::Const,
            "type" => Token::Type,
            "trait" => Token::Trait,
            "impl" => Token::Impl,
            "pub" => Token::Pub,
            "extern" => Token::Extern,
            "model" => Token::Model,
            // "layer", "input", "output" are contextual — parsed as Ident inside model blocks
            "f32" => Token::F32,
            "f64" => Token::F64,
            "i32" => Token::I32,
            "i64" => Token::I64,
            "bool" => Token::Bool,
            "tensor" => Token::Tensor,
            "str" => Token::Str,
            "true" => Token::BoolLit(true),
            "false" => Token::BoolLit(false),
            _ => Token::Ident(text.to_owned()),
        };
        Spanned {
            node: tok,
            span: Span::new(start, self.pos as u32),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: tokenize source and return the token nodes (without spans).
    fn toks(src: &str) -> Vec<Token> {
        Lexer::new(src)
            .tokenize()
            .expect("tokenize failed")
            .into_iter()
            .map(|s| s.node)
            .collect()
    }

    // -- Span tests -------------------------------------------------------

    #[test]
    fn span_merge() {
        let a = Span::new(2, 5);
        let b = Span::new(8, 12);
        let m = a.merge(b);
        assert_eq!(m.start, BytePos(2));
        assert_eq!(m.end, BytePos(12));
    }

    #[test]
    fn span_at_zero_length() {
        let s = Span::at(7);
        assert_eq!(s.start, BytePos(7));
        assert_eq!(s.end, BytePos(7));
    }

    // -- Keywords ---------------------------------------------------------

    #[test]
    fn lex_keywords() {
        let tokens = toks("def val var return if else while loop break continue");
        assert_eq!(tokens[0], Token::Def);
        assert_eq!(tokens[1], Token::Val);
        assert_eq!(tokens[2], Token::Var);
        assert_eq!(tokens[3], Token::Return);
        assert_eq!(tokens[4], Token::If);
        assert_eq!(tokens[5], Token::Else);
        assert_eq!(tokens[6], Token::While);
        assert_eq!(tokens[7], Token::Loop);
        assert_eq!(tokens[8], Token::Break);
        assert_eq!(tokens[9], Token::Continue);
    }

    #[test]
    fn lex_type_keywords() {
        let tokens = toks("f32 f64 i32 i64 bool tensor str");
        assert_eq!(tokens[0], Token::F32);
        assert_eq!(tokens[1], Token::F64);
        assert_eq!(tokens[2], Token::I32);
        assert_eq!(tokens[3], Token::I64);
        assert_eq!(tokens[4], Token::Bool);
        assert_eq!(tokens[5], Token::Tensor);
        assert_eq!(tokens[6], Token::Str);
    }

    #[test]
    fn lex_advanced_keywords() {
        let tokens = toks("record bring when choice for in spawn par async await const type trait impl pub extern");
        assert_eq!(tokens[0], Token::Record);
        assert_eq!(tokens[1], Token::Bring);
        assert_eq!(tokens[2], Token::When);
        assert_eq!(tokens[3], Token::Choice);
        assert_eq!(tokens[4], Token::For);
        assert_eq!(tokens[5], Token::In);
        assert_eq!(tokens[6], Token::Spawn);
        assert_eq!(tokens[7], Token::Par);
        assert_eq!(tokens[8], Token::Async);
        assert_eq!(tokens[9], Token::Await);
        assert_eq!(tokens[10], Token::Const);
        assert_eq!(tokens[11], Token::Type);
        assert_eq!(tokens[12], Token::Trait);
        assert_eq!(tokens[13], Token::Impl);
        assert_eq!(tokens[14], Token::Pub);
        assert_eq!(tokens[15], Token::Extern);
    }

    // -- Literals ---------------------------------------------------------

    #[test]
    fn lex_integer_literals() {
        let tokens = toks("0 42 1000000");
        assert_eq!(tokens[0], Token::IntLit(0));
        assert_eq!(tokens[1], Token::IntLit(42));
        assert_eq!(tokens[2], Token::IntLit(1000000));
    }

    #[test]
    fn lex_float_literals() {
        let tokens = toks("3.14 0.5 1.0e10 2.5E3");
        assert!(
            matches!(tokens[0], Token::FloatLit(v) if (v - std::f64::consts::PI).abs() < 1e-10)
        );
        assert!(matches!(tokens[1], Token::FloatLit(v) if (v - 0.5).abs() < 1e-10));
        assert!(matches!(tokens[2], Token::FloatLit(v) if (v - 1.0e10).abs() < 1e5));
        assert!(matches!(tokens[3], Token::FloatLit(v) if (v - 2.5e3).abs() < 1e-10));
    }

    #[test]
    fn lex_bool_literals() {
        let tokens = toks("true false");
        assert_eq!(tokens[0], Token::BoolLit(true));
        assert_eq!(tokens[1], Token::BoolLit(false));
    }

    #[test]
    fn lex_string_literal() {
        let tokens = toks(r#""hello world""#);
        assert_eq!(tokens[0], Token::StringLit("hello world".into()));
    }

    #[test]
    fn lex_string_escape_sequences() {
        let tokens = toks(r#""line1\nline2\ttab\\backslash\"quote""#);
        assert_eq!(
            tokens[0],
            Token::StringLit("line1\nline2\ttab\\backslash\"quote".into())
        );
    }

    #[test]
    fn lex_fstring() {
        let tokens = toks(r#"f"hello {name}""#);
        assert_eq!(tokens[0], Token::FStringLit("hello {name}".into()));
    }

    // -- Punctuation and operators ----------------------------------------

    #[test]
    fn lex_two_char_operators() {
        let tokens = toks("-> == != <= >= => && || .. ..=");
        assert_eq!(tokens[0], Token::Arrow);
        assert_eq!(tokens[1], Token::EqEq);
        assert_eq!(tokens[2], Token::NotEq);
        assert_eq!(tokens[3], Token::LtEq);
        assert_eq!(tokens[4], Token::GtEq);
        assert_eq!(tokens[5], Token::FatArrow);
        assert_eq!(tokens[6], Token::AmpAmp);
        assert_eq!(tokens[7], Token::PipePipe);
        assert_eq!(tokens[8], Token::DotDot);
        assert_eq!(tokens[9], Token::DotDotEq);
    }

    #[test]
    fn lex_single_char_punctuation() {
        let tokens = toks("( ) { } [ ] < > , : ; = + - * / % ! . | ? @");
        assert_eq!(tokens[0], Token::LParen);
        assert_eq!(tokens[1], Token::RParen);
        assert_eq!(tokens[2], Token::LBrace);
        assert_eq!(tokens[3], Token::RBrace);
        assert_eq!(tokens[4], Token::LBracket);
        assert_eq!(tokens[5], Token::RBracket);
        assert_eq!(tokens[6], Token::LAngle);
        assert_eq!(tokens[7], Token::RAngle);
        assert_eq!(tokens[8], Token::Comma);
        assert_eq!(tokens[9], Token::Colon);
        assert_eq!(tokens[10], Token::Semi);
        assert_eq!(tokens[11], Token::Eq);
        assert_eq!(tokens[12], Token::Plus);
        assert_eq!(tokens[13], Token::Minus);
        assert_eq!(tokens[14], Token::Star);
        assert_eq!(tokens[15], Token::Slash);
        assert_eq!(tokens[16], Token::Percent);
        assert_eq!(tokens[17], Token::Bang);
        assert_eq!(tokens[18], Token::Dot);
        assert_eq!(tokens[19], Token::Pipe);
        assert_eq!(tokens[20], Token::Question);
        assert_eq!(tokens[21], Token::At);
    }

    // -- Comments ---------------------------------------------------------

    #[test]
    fn lex_skips_line_comments() {
        let tokens = toks("def // this is a comment\nmain");
        assert_eq!(tokens[0], Token::Def);
        assert_eq!(tokens[1], Token::Ident("main".into()));
        assert_eq!(tokens[2], Token::Eof);
    }

    // -- Identifiers ------------------------------------------------------

    #[test]
    fn lex_identifiers() {
        let tokens = toks("foo _bar baz_123 CamelCase");
        assert_eq!(tokens[0], Token::Ident("foo".into()));
        assert_eq!(tokens[1], Token::Ident("_bar".into()));
        assert_eq!(tokens[2], Token::Ident("baz_123".into()));
        assert_eq!(tokens[3], Token::Ident("CamelCase".into()));
    }

    // -- Complex expressions ----------------------------------------------

    #[test]
    fn lex_function_definition() {
        let tokens = toks("def add(a: i64, b: i64) -> i64 { a + b }");
        assert_eq!(tokens[0], Token::Def);
        assert_eq!(tokens[1], Token::Ident("add".into()));
        assert_eq!(tokens[2], Token::LParen);
        assert_eq!(tokens[3], Token::Ident("a".into()));
        assert_eq!(tokens[4], Token::Colon);
        assert_eq!(tokens[5], Token::I64);
        assert_eq!(tokens[6], Token::Comma);
    }

    // -- Error conditions -------------------------------------------------

    #[test]
    fn lex_unterminated_string_is_err() {
        let result = Lexer::new(r#""hello"#).tokenize();
        assert!(result.is_err());
    }

    #[test]
    fn lex_invalid_escape_is_err() {
        let result = Lexer::new(r#""\z""#).tokenize();
        assert!(result.is_err());
    }

    #[test]
    fn lex_unrecognized_char_is_err() {
        let result = Lexer::new("def main() { # }").tokenize();
        assert!(result.is_err());
    }

    // -- EOF token --------------------------------------------------------

    #[test]
    fn lex_empty_source() {
        let tokens = toks("");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], Token::Eof);
    }

    #[test]
    fn lex_whitespace_only() {
        let tokens = toks("   \n\t  ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], Token::Eof);
    }

    // -- Token Display ----------------------------------------------------

    #[test]
    fn token_display() {
        assert_eq!(format!("{}", Token::Def), "def");
        assert_eq!(format!("{}", Token::Arrow), "->");
        assert_eq!(format!("{}", Token::IntLit(42)), "42");
        assert_eq!(format!("{}", Token::BoolLit(true)), "true");
        assert_eq!(format!("{}", Token::StringLit("hi".into())), "\"hi\"");
        assert_eq!(format!("{}", Token::Eof), "<eof>");
    }
}
