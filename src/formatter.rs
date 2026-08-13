use std::path::{Path, PathBuf};

use crate::parser::lexer::{Lexer, Token};

/// Options controlling code formatting.
#[derive(Debug, Clone)]
pub struct FormatOptions {
    /// Number of spaces per indentation level.
    pub indent: usize,
    /// Maximum line width before wrapping (currently unused by the formatter).
    pub max_line_width: usize,
}

impl Default for FormatOptions {
    fn default() -> Self {
        FormatOptions {
            indent: 4,
            max_line_width: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Formats an IRIS source string according to style rules.
/// Returns `Err` only on lexer error; on parse errors the original source is returned.
pub fn format_source(source: &str, options: &FormatOptions) -> Result<String, String> {
    Ok(format_iris(source, options))
}

/// Formats a single file.  When `check_only` is true the file is not modified;
/// returns `Ok(true)` if the file would change.
pub fn format_file(
    path: &Path,
    options: &FormatOptions,
    check_only: bool,
) -> Result<bool, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read '{}': {}", path.display(), e))?;
    let formatted = format_source(&source, options)?;
    let changed = formatted != source;
    if !check_only && changed {
        std::fs::write(path, &formatted)
            .map_err(|e| format!("cannot write '{}': {}", path.display(), e))?;
    }
    Ok(changed)
}

/// Formats every `*.iris` file in `dir` (non-recursive).
/// Returns `(total_files, changed_files)`.
pub fn format_directory(
    dir: &Path,
    options: &FormatOptions,
    check_only: bool,
) -> Result<(usize, usize), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read directory '{}': {}", dir.display(), e))?;
    let mut total = 0usize;
    let mut changed = 0usize;
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map_or(false, |ext| ext == "iris")
                && p.file_name().map_or(false, |n| {
                    n.to_str().map_or(false, |s| !s.starts_with('.'))
                })
        })
        .collect();
    files.sort();
    for path in &files {
        total += 1;
        if format_file(path, options, check_only)? {
            changed += 1;
            if !check_only {
                eprintln!("  formatted {}", path.display());
            }
        }
    }
    Ok((total, changed))
}

// ---------------------------------------------------------------------------
// Core formatter (extracted from src/lsp.rs format_iris)
// ---------------------------------------------------------------------------

/// Token-stream based IRIS formatter.  Normalises indentation and spacing.
fn format_iris(source: &str, options: &FormatOptions) -> String {
    let spanned_tokens = match Lexer::new(source).tokenize() {
        Ok(t) => t,
        Err(_) => return source.to_owned(),
    };

    let indent_width = options.indent;
    let indent_str = |depth: usize| " ".repeat(indent_width * depth);

    let mut out = String::with_capacity(source.len() + 64);
    let mut indent = 0usize;
    let mut at_line_start = true;
    let mut prev_was_newline = false;
    let mut blank_lines = 0usize;
    let mut prev_tok_was_pub = false;

    let is_top_level_kw = |t: &Token| {
        matches!(
            t,
            Token::Def
                | Token::Record
                | Token::Choice
                | Token::Model
                | Token::Const
                | Token::Type
                | Token::Extern
                | Token::Trait
                | Token::Impl
                | Token::Mod
                | Token::Pub
        )
    };

    let is_stmt_kw = |t: &Token| {
        matches!(
            t,
            Token::Val
                | Token::Var
                | Token::For
                | Token::While
                | Token::Loop
                | Token::Return
                | Token::Break
                | Token::Continue
                | Token::Spawn
                | Token::Par
        )
    };

    for (idx, spanned) in spanned_tokens.iter().enumerate() {
        let tok = &spanned.node;
        let tok_str = token_to_str(tok, source);
        if tok_str.is_empty() {
            continue;
        }

        if is_top_level_kw(tok)
            && indent == 0
            && !out.is_empty()
            && blank_lines == 0
            && !at_line_start
            && !(matches!(tok, Token::Def) && prev_tok_was_pub)
        {
            out.push('\n');
            blank_lines = 1;
        }

        if indent > 0 && !at_line_start && is_stmt_kw(tok) {
            out.push('\n');
            at_line_start = true;
        }

        if at_line_start {
            let ind = indent_str(indent);
            out.push_str(&ind);
            at_line_start = false;
        }

        if tok_str == "{" {
            if !out.ends_with(' ') && !out.ends_with('\n') {
                out.push(' ');
            }
            out.push('{');
            indent += 1;
            out.push('\n');
            at_line_start = true;
            blank_lines = 0;
            prev_was_newline = true;
            continue;
        }

        if tok_str == "}" {
            indent = indent.saturating_sub(1);
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&indent_str(indent));
            out.push('}');
            out.push('\n');
            at_line_start = true;
            blank_lines = 0;
            prev_was_newline = true;
            continue;
        }

        if tok_str == ";" {
            out.push(';');
            out.push('\n');
            at_line_start = true;
            blank_lines = 0;
            prev_was_newline = false;
            continue;
        }

        if tok_str == "," {
            if out.ends_with(' ') {
                out.pop();
            }
            out.push(',');
            out.push(' ');
            prev_was_newline = false;
            continue;
        }

        let needs_space = matches!(
            tok_str.as_str(),
            "=" | "=="
                | "!="
                | "<="
                | ">="
                | "+"
                | "-"
                | "*"
                | "/"
                | "%"
                | "&&"
                | "||"
                | "->"
                | "=>"
                | ".."
                | "..="
                | ":"
                | "to"
        );

        if needs_space {
            if !out.ends_with(' ') && !out.ends_with('\n') {
                out.push(' ');
            }
            out.push_str(&tok_str);
            out.push(' ');
        } else if tok_str == "(" || tok_str == "[" || tok_str == "<" {
            out.push_str(&tok_str);
        } else if tok_str == ")" || tok_str == "]" || tok_str == ">" {
            if out.ends_with(' ') {
                out.pop();
            }
            out.push_str(&tok_str);
        } else {
            let last = out.chars().last();
            let needs_sep = matches!(last, Some(c) if c.is_alphanumeric() || c == '_' || c == '"');
            if needs_sep && !tok_str.starts_with(['.', '(', '[']) {
                out.push(' ');
            }
            out.push_str(&tok_str);
        }

        let _ = (idx, prev_was_newline, spanned.span);
        prev_was_newline = false;
        blank_lines = 0;
        prev_tok_was_pub = matches!(tok, Token::Pub);
    }

    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Returns the source text for a token (for formatting).
fn token_to_str(tok: &Token, _source: &str) -> String {
    match tok {
        Token::Def => "def".into(),
        Token::DefMacro => "defmacro".into(),
        Token::Val => "val".into(),
        Token::Var => "var".into(),
        Token::Let => "let".into(),
        Token::If => "if".into(),
        Token::Else => "else".into(),
        Token::Match => "match".into(),
        Token::When => "when".into(),
        Token::For => "for".into(),
        Token::While => "while".into(),
        Token::Loop => "loop".into(),
        Token::Break => "break".into(),
        Token::Continue => "continue".into(),
        Token::Return => "return".into(),
        Token::Record => "record".into(),
        Token::Choice => "choice".into(),
        Token::Model => "model".into(),
        Token::Layer => "layer".into(),
        Token::Input => "input".into(),
        Token::Output => "output".into(),
        Token::Const => "const".into(),
        Token::Type => "type".into(),
        Token::Extern => "extern".into(),
        Token::Trait => "trait".into(),
        Token::Impl => "impl".into(),
        Token::Mod => "mod".into(),
        Token::Pub => "pub".into(),
        Token::Bring => "bring".into(),
        Token::Async => "async".into(),
        Token::Await => "await".into(),
        Token::Spawn => "spawn".into(),
        Token::Par => "par".into(),
        Token::In => "in".into(),
        Token::To => "to".into(),
        Token::BoolLit(b) => {
            if *b {
                "true".into()
            } else {
                "false".into()
            }
        }
        Token::I64 => "i64".into(),
        Token::I32 => "i32".into(),
        Token::I8 => "i8".into(),
        Token::U8 => "u8".into(),
        Token::U32 => "u32".into(),
        Token::U64 => "u64".into(),
        Token::Usize => "usize".into(),
        Token::F64 => "f64".into(),
        Token::F32 => "f32".into(),
        Token::Bool => "bool".into(),
        Token::Str => "str".into(),
        Token::Tensor => "tensor".into(),
        Token::LBrace => "{".into(),
        Token::RBrace => "}".into(),
        Token::LParen => "(".into(),
        Token::RParen => ")".into(),
        Token::LBracket => "[".into(),
        Token::RBracket => "]".into(),
        Token::LAngle => "<".into(),
        Token::RAngle => ">".into(),
        Token::Comma => ",".into(),
        Token::Semi => ";".into(),
        Token::Colon => ":".into(),
        Token::DoubleColon => "::".into(),
        Token::Dot => ".".into(),
        Token::DotDot => "..".into(),
        Token::DotDotEq => "..=".into(),
        Token::Arrow => "->".into(),
        Token::FatArrow => "=>".into(),
        Token::Eq => "=".into(),
        Token::EqEq => "==".into(),
        Token::NotEq => "!=".into(),
        Token::LtGt => "<>".into(),
        Token::LtEq => "<=".into(),
        Token::GtEq => ">=".into(),
        Token::PlusEq => "+=".into(),
        Token::MinusEq => "-=".into(),
        Token::StarEq => "*=".into(),
        Token::SlashEq => "/=".into(),
        Token::PercentEq => "%=".into(),
        Token::Plus => "+".into(),
        Token::Minus => "-".into(),
        Token::Star => "*".into(),
        Token::Slash => "/".into(),
        Token::Percent => "%".into(),
        Token::Pipe => "|".into(),
        Token::AmpAmp => "&&".into(),
        Token::PipePipe => "||".into(),
        Token::Bang => "!".into(),
        Token::At => "@".into(),
        Token::Question => "?".into(),
        Token::QuestionQuestion => "??".into(),
        Token::Ident(s) => s.clone(),
        Token::IntLit(n) => n.to_string(),
        Token::FloatLit(f) => {
            if f.fract() == 0.0 {
                format!("{:.1}", f)
            } else {
                f.to_string()
            }
        }
        Token::StringLit(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Token::CharLit(c) => format!("'{}'", *c as u8 as char),
        Token::FStringLit(s) => format!("f\"{}\"", s),
        Token::Eof => String::new(),
        Token::Effect => "effect".to_owned(),
        Token::With => "with".to_owned(),
        Token::Yield => "yield".to_owned(),
        Token::Dyn => "dyn".to_owned(),
        Token::Resume => "resume".to_owned(),
        Token::By => "by".to_owned(),
        Token::Defer => "defer".to_owned(),
        Token::Try => "try".to_owned(),
        Token::Catch => "catch".to_owned(),
        Token::Raise => "raise".to_owned(),
        Token::Amp => "&".to_owned(),
        Token::Move => "move".to_owned(),
        Token::Unsafe => "unsafe".to_owned(),
        Token::Select => "select".to_owned(),
        Token::DocComment(s) => format!("/// {}", s),
    }
}
