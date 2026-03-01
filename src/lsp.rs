//! Language Server Protocol (LSP) implementation for IRIS.
//!
//! [`LspState`] provides the core document/diagnostic API, testable without I/O.
//! [`run_lsp_server`] wraps it in a JSON-RPC Content-Length message loop on
//! stdin/stdout, compatible with any LSP client (VSCode, Neovim, etc.).

use std::collections::HashMap;

use crate::diagnostics::{byte_to_line_col, error_byte_offset};
use crate::EmitKind;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single diagnostic (error or warning) for the LSP client.
#[derive(Debug, Clone)]
pub struct LspDiagnostic {
    /// 0-based start line.
    pub line: u32,
    /// 0-based start character (UTF-16 code unit index).
    pub character: u32,
    /// 0-based end line (same as `line` for single-line errors).
    pub end_line: u32,
    /// 0-based end character.
    pub end_character: u32,
    /// Human-readable message.
    pub message: String,
    /// 1 = Error, 2 = Warning, 3 = Information, 4 = Hint.
    pub severity: u8,
}

/// Persistent LSP server state: one entry per open document.
#[derive(Default)]
pub struct LspState {
    /// URI → source text.
    documents: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Static keyword + builtin completion list
// ---------------------------------------------------------------------------

static STATIC_COMPLETIONS: &[&str] = &[
    "def", "val", "var", "for", "while", "loop", "if", "else", "when",
    "return", "break", "continue", "choice", "record", "const", "type",
    "extern", "trait", "impl", "bring", "pub", "async", "await", "spawn",
    "par", "in",
    // builtins
    "print", "panic", "assert", "len", "concat", "sin", "cos", "sqrt",
    "abs", "floor", "ceil", "exp", "log", "min", "max", "pow", "clamp",
    "some", "none", "ok", "err", "is_some", "is_ok", "unwrap",
    "list", "push", "map", "cell", "cell_get", "cell_set",
    "to_str", "format", "read_line", "read_i64", "read_f64",
    "grad", "grad_of", "zeros", "ones", "fill", "linspace",
    "split", "join", "contains", "starts_with", "ends_with",
    "to_upper", "to_lower", "trim", "repeat",
];

// ---------------------------------------------------------------------------
// LspState implementation
// ---------------------------------------------------------------------------

impl LspState {
    /// Creates an empty LSP state.
    pub fn new() -> Self { Self::default() }

    /// Called when the editor opens a document. Returns initial diagnostics.
    pub fn open_document(&mut self, uri: &str, text: &str) -> Vec<LspDiagnostic> {
        self.documents.insert(uri.to_owned(), text.to_owned());
        self.diagnose(uri)
    }

    /// Called when the editor changes a document. Returns updated diagnostics.
    pub fn update_document(&mut self, uri: &str, text: &str) -> Vec<LspDiagnostic> {
        self.documents.insert(uri.to_owned(), text.to_owned());
        self.diagnose(uri)
    }

    /// Called when the editor closes a document.
    pub fn close_document(&mut self, uri: &str) {
        self.documents.remove(uri);
    }

    /// Returns hover information (type signature) for the identifier at the given position.
    pub fn hover(&self, uri: &str, line: u32, character: u32) -> Option<String> {
        let source = self.documents.get(uri)?;
        let module_name = uri_to_module_name(uri);
        let module = crate::compile_to_module(source, &module_name).ok()?;

        let byte = line_col_to_byte(source, line, character);
        let ident = ident_at_byte(source, byte)?;

        for func in module.functions() {
            let bare = func.name.split("__").next().unwrap_or(&func.name);
            if bare == ident {
                let params: Vec<String> = func.params.iter()
                    .map(|p| format!("{}: {:?}", p.name, p.ty))
                    .collect();
                return Some(format!("def {}({}) -> {:?}", bare, params.join(", "), func.return_ty));
            }
        }
        None
    }

    /// Returns completion candidates for the given position.
    pub fn completions(&self, uri: &str) -> Vec<String> {
        let mut items: Vec<String> = STATIC_COMPLETIONS.iter().map(|s| s.to_string()).collect();

        if let Some(source) = self.documents.get(uri) {
            let module_name = uri_to_module_name(uri);
            if let Ok(module) = crate::compile_to_module(source, &module_name) {
                for func in module.functions() {
                    let bare = func.name.split("__").next().unwrap_or(&func.name);
                    if !bare.starts_with("__") {
                        items.push(bare.to_owned());
                    }
                }
            }
        }
        items.sort();
        items.dedup();
        items
    }

    /// Returns the definition location for the identifier at the given position.
    /// Returns `(uri, start_line, start_char, end_line, end_char)` on success.
    pub fn definition(&self, uri: &str, line: u32, character: u32) -> Option<(String, u32, u32, u32, u32)> {
        let source = self.documents.get(uri)?;
        let byte = line_col_to_byte(source, line, character);
        let ident = ident_at_byte(source, byte)?;

        let ast = parse_source(source)?;
        let def_byte = definition_byte_of(&ast, ident)?;
        let (start_line, start_char) = byte_to_lsp_pos(source, def_byte);
        let end_char = start_char + ident.len() as u32;
        Some((uri.to_owned(), start_line, start_char, start_line, end_char))
    }

    /// Returns document symbols (outline) for the given document.
    /// Each entry is `(name, kind, start_line, start_char, end_line, end_char)`.
    /// SymbolKind: 12=Function, 23=Struct, 10=Enum, 14=Constant, 26=TypeParameter
    pub fn document_symbols(&self, uri: &str) -> Vec<(String, u32, u32, u32, u32, u32)> {
        let source = match self.documents.get(uri) { Some(s) => s, None => return vec![] };
        let ast = match parse_source(source) { Some(a) => a, None => return vec![] };
        let mut symbols = Vec::new();

        for func in &ast.functions {
            let (sl, sc) = byte_to_lsp_pos(source, func.name.span.start.0);
            let end_byte = func.span.end.0.min(source.len() as u32);
            let (el, ec) = byte_to_lsp_pos(source, end_byte);
            symbols.push((func.name.name.clone(), 12u32, sl, sc, el, ec));
        }
        for s in &ast.structs {
            let (sl, sc) = byte_to_lsp_pos(source, s.name.span.start.0);
            let end_byte = s.span.end.0.min(source.len() as u32);
            let (el, ec) = byte_to_lsp_pos(source, end_byte);
            symbols.push((s.name.name.clone(), 23u32, sl, sc, el, ec));
        }
        for e in &ast.enums {
            let (sl, sc) = byte_to_lsp_pos(source, e.name.span.start.0);
            let end_byte = e.span.end.0.min(source.len() as u32);
            let (el, ec) = byte_to_lsp_pos(source, end_byte);
            symbols.push((e.name.name.clone(), 10u32, sl, sc, el, ec));
        }
        for c in &ast.consts {
            let (sl, sc) = byte_to_lsp_pos(source, c.name.span.start.0);
            let (el, ec) = (sl, sc + c.name.name.len() as u32);
            symbols.push((c.name.name.clone(), 14u32, sl, sc, el, ec));
        }
        for ta in &ast.type_aliases {
            let (sl, sc) = byte_to_lsp_pos(source, ta.span.start.0);
            let (el, ec) = (sl, sc + ta.name.len() as u32);
            symbols.push((ta.name.clone(), 26u32, sl, sc, el, ec));
        }

        // Sort by line for a predictable outline order.
        symbols.sort_by_key(|s| s.2);
        symbols
    }

    /// Returns signature help for a function call at the given position.
    /// Returns `(label, parameter_labels, active_parameter_index)`.
    pub fn signature_help(&self, uri: &str, line: u32, character: u32) -> Option<(String, Vec<String>, usize)> {
        let source = self.documents.get(uri)?;
        let (func_name, active_param) = find_call_context(source, line, character)?;
        let ast = parse_source(source)?;

        // Find the function definition.
        let func = ast.functions.iter().find(|f| f.name.name == func_name)?;

        let param_labels: Vec<String> = func.params.iter()
            .map(|p| format!("{}: {}", p.name.name, ast_type_str(&p.ty)))
            .collect();
        let ret = ast_type_str(&func.return_ty);
        let label = format!("def {}({}) -> {}", func_name, param_labels.join(", "), ret);

        Some((label, param_labels, active_param))
    }

    /// Returns a formatted version of the document source.
    pub fn format(&self, uri: &str) -> Option<String> {
        let source = self.documents.get(uri)?;
        Some(format_iris(source))
    }

    // ------------------------------------------------------------------
    // Private
    // ------------------------------------------------------------------

    fn diagnose(&self, uri: &str) -> Vec<LspDiagnostic> {
        let Some(source) = self.documents.get(uri) else { return Vec::new() };
        let module_name = uri_to_module_name(uri);
        let mut diags = Vec::new();

        // Try file-based compilation first (resolves bring declarations).
        // Fall back to in-memory compilation for unsaved / untitled files.
        let compile_result = if let Some(path) = uri_to_file_path(uri) {
            crate::compile_file_text(source, &path, EmitKind::Ir)
        } else {
            crate::compile(source, &module_name, EmitKind::Ir)
        };

        if let Err(e) = compile_result {
            let (line, character) = if let Some(byte) = error_byte_offset(&e) {
                let (l, c) = byte_to_line_col(source, byte);
                (l.saturating_sub(1), c.saturating_sub(1))
            } else {
                (0, 0)
            };
            diags.push(LspDiagnostic {
                line,
                character,
                end_line: line,
                end_character: character + 1,
                message: format!("{}", e),
                severity: 1,
            });
        }

        // Collect dead-variable warnings directly from the single-file AST.
        // This works even when bring resolution fails (warnings are per-file).
        if let Some(ast) = parse_source(source) {
            for w in crate::pass::find_unused_vars(&ast) {
                let (line, character) = if let Some(sp) = w.span {
                    let (l, c) = byte_to_line_col(source, sp.start.0);
                    (l.saturating_sub(1), c.saturating_sub(1))
                } else {
                    (0u32, 0u32)
                };
                diags.push(LspDiagnostic {
                    line,
                    character,
                    end_line: line,
                    end_character: character + 1,
                    message: w.message,
                    severity: 2,
                });
            }
        }

        diags
    }
}

// ---------------------------------------------------------------------------
// AST helpers
// ---------------------------------------------------------------------------

fn parse_source(source: &str) -> Option<crate::parser::ast::AstModule> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;
    let tokens = Lexer::new(source).tokenize().ok()?;
    Parser::new(&tokens).parse_module().ok()
}

/// Finds the byte offset of the definition of `name` in the AST.
fn definition_byte_of(ast: &crate::parser::ast::AstModule, name: &str) -> Option<u32> {
    for func in &ast.functions {
        if func.name.name == name {
            return Some(func.name.span.start.0);
        }
    }
    for s in &ast.structs {
        if s.name.name == name {
            return Some(s.name.span.start.0);
        }
    }
    for e in &ast.enums {
        if e.name.name == name {
            return Some(e.name.span.start.0);
        }
    }
    for c in &ast.consts {
        if c.name.name == name {
            return Some(c.name.span.start.0);
        }
    }
    for ta in &ast.type_aliases {
        if ta.name == name {
            return Some(ta.span.start.0);
        }
    }
    None
}

/// Converts a byte offset to a 0-based (line, character) LSP position.
fn byte_to_lsp_pos(source: &str, byte: u32) -> (u32, u32) {
    let byte = byte as usize;
    let prefix = if byte <= source.len() { &source[..byte] } else { source };
    let line = prefix.bytes().filter(|&b| b == b'\n').count() as u32;
    let col = prefix.rfind('\n').map(|i| byte - i - 1).unwrap_or(byte) as u32;
    (line, col)
}

/// Converts an AstType to a display string.
fn ast_type_str(ty: &crate::parser::ast::AstType) -> String {
    use crate::parser::ast::{AstType, AstScalarKind};
    match ty {
        AstType::Scalar(k, _) => match k {
            AstScalarKind::I64 => "i64",
            AstScalarKind::I32 => "i32",
            AstScalarKind::F64 => "f64",
            AstScalarKind::F32 => "f32",
            AstScalarKind::Bool => "bool",
            AstScalarKind::U8 => "u8",
            AstScalarKind::I8 => "i8",
            AstScalarKind::U32 => "u32",
            AstScalarKind::U64 => "u64",
            AstScalarKind::USize => "usize",
        }.to_owned(),
        AstType::Named(n, _) => n.clone(),
        AstType::Tuple(ts, _) => {
            let inner: Vec<String> = ts.iter().map(ast_type_str).collect();
            format!("({})", inner.join(", "))
        }
        AstType::List(t, _) => format!("list<{}>", ast_type_str(t)),
        AstType::Map(k, v, _) => format!("map<{}, {}>", ast_type_str(k), ast_type_str(v)),
        AstType::Option(t, _) => format!("option<{}>", ast_type_str(t)),
        AstType::Result(t, e, _) => format!("result<{}, {}>", ast_type_str(t), ast_type_str(e)),
        AstType::Fn { params, ret, .. } => {
            let ps: Vec<String> = params.iter().map(ast_type_str).collect();
            format!("({}) -> {}", ps.join(", "), ast_type_str(ret))
        }
        _ => "?".to_owned(),
    }
}

/// Finds the function name and active parameter index at the cursor position.
/// Scans backwards to find the innermost unclosed `(` and the identifier before it.
fn find_call_context(source: &str, line: u32, character: u32) -> Option<(String, usize)> {
    let cursor_byte = line_col_to_byte(source, line, character) as usize;
    let prefix = &source[..cursor_byte.min(source.len())];

    let mut depth = 0i32;
    let mut open_pos = None;
    for (i, ch) in prefix.char_indices().rev() {
        match ch {
            ')' => depth += 1,
            '(' => {
                if depth == 0 {
                    open_pos = Some(i);
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
    }

    let open = open_pos?;
    // Count commas between open_pos and cursor to find active parameter.
    let between = &prefix[open + 1..];
    let mut depth2 = 0i32;
    let active_param = between.chars().filter(|&c| {
        if c == '(' || c == '[' { depth2 += 1; }
        if c == ')' || c == ']' { depth2 -= 1; }
        c == ',' && depth2 == 0
    }).count();

    // Find identifier immediately before `(`.
    let before_paren = prefix[..open].trim_end();
    let ident_end = before_paren.len();
    let ident_start = before_paren.rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);
    let func_name = &before_paren[ident_start..ident_end];
    if func_name.is_empty() { return None; }

    Some((func_name.to_owned(), active_param))
}

// ---------------------------------------------------------------------------
// Simple formatter
// ---------------------------------------------------------------------------

/// Token-stream based IRIS formatter. Normalises indentation and spacing.
fn format_iris(source: &str) -> String {
    use crate::parser::lexer::{Lexer, Token};

    let spanned_tokens = match Lexer::new(source).tokenize() {
        Ok(t) => t,
        Err(_) => return source.to_owned(),
    };

    let mut out = String::with_capacity(source.len() + 64);
    let mut indent = 0usize;
    let mut at_line_start = true;
    let mut prev_was_newline = false;
    let mut blank_lines = 0usize;

    // Helper: emit current indentation.
    let indent_str = |depth: usize| "    ".repeat(depth);

    // Top-level item starters that get a blank line before them.
    let is_top_level_kw = |t: &Token| matches!(
        t,
        Token::Def | Token::Record | Token::Choice | Token::Const | Token::Type
        | Token::Extern | Token::Trait | Token::Impl | Token::Pub
    );

    for (idx, spanned) in spanned_tokens.iter().enumerate() {
        let tok = &spanned.node;
        let tok_str = token_to_str(tok, source);
        if tok_str.is_empty() { continue; }

        // Emit blank line before top-level keywords (except at very start).
        if is_top_level_kw(tok) && indent == 0 && !out.is_empty() && blank_lines == 0 && !at_line_start {
            out.push('\n');
            blank_lines = 1;
        }

        // Newlines and indentation.
        if at_line_start {
            let ind = indent_str(indent);
            out.push_str(&ind);
            at_line_start = false;
        }

        // Opening brace: emit, then newline + increase indent.
        if tok_str == "{" {
            // Space before `{` if not at line start.
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

        // Closing brace: decrease indent, then emit on its own line.
        if tok_str == "}" {
            if indent > 0 { indent -= 1; }
            if !out.ends_with('\n') { out.push('\n'); }
            out.push_str(&indent_str(indent));
            out.push('}');
            out.push('\n');
            at_line_start = true;
            blank_lines = 0;
            prev_was_newline = true;
            continue;
        }

        // Semicolons and angle brackets for generics — pass through.
        if tok_str == ";" {
            out.push(';');
            out.push('\n');
            at_line_start = true;
            blank_lines = 0;
            prev_was_newline = false;
            continue;
        }

        // Commas — no leading space, one trailing space.
        if tok_str == "," {
            // Remove trailing space before comma.
            if out.ends_with(' ') { out.pop(); }
            out.push(',');
            out.push(' ');
            prev_was_newline = false;
            continue;
        }

        // Operators that need surrounding spaces.
        let needs_space = matches!(tok_str.as_str(),
            "=" | "==" | "!=" | "<=" | ">=" |
            "+" | "-" | "*" | "/" | "%" | "&&" | "||" |
            "->" | "=>" | ".." | "..=" | ":" | "to"
        );

        if needs_space {
            if !out.ends_with(' ') && !out.ends_with('\n') { out.push(' '); }
            out.push_str(&tok_str);
            out.push(' ');
        } else if tok_str == "(" || tok_str == "[" || tok_str == "<" {
            // No space before open paren/bracket (function calls, indexing, generics).
            out.push_str(&tok_str);
        } else if tok_str == ")" || tok_str == "]" || tok_str == ">" {
            if out.ends_with(' ') { out.pop(); }
            out.push_str(&tok_str);
        } else {
            // Default: keyword or identifier — space between tokens unless at line start.
            let last = out.chars().last();
            let needs_sep = matches!(last, Some(c) if c.is_alphanumeric() || c == '_' || c == '"');
            if needs_sep && !tok_str.starts_with(|c: char| c == '.' || c == '(' || c == '[') {
                out.push(' ');
            }
            out.push_str(&tok_str);
        }

        let _ = (idx, prev_was_newline, spanned.span); // suppress unused warnings
        prev_was_newline = false;
        blank_lines = 0;
    }

    if !out.ends_with('\n') { out.push('\n'); }
    out
}

/// Returns the source text for a token (for formatting).
fn token_to_str(tok: &crate::parser::lexer::Token, _source: &str) -> String {
    use crate::parser::lexer::Token;
    match tok {
        Token::Def => "def".into(),
        Token::Val => "val".into(),
        Token::Var => "var".into(),
        Token::If => "if".into(),
        Token::Else => "else".into(),
        Token::When => "when".into(),
        Token::For => "for".into(),
        Token::While => "while".into(),
        Token::Loop => "loop".into(),
        Token::Break => "break".into(),
        Token::Continue => "continue".into(),
        Token::Return => "return".into(),
        Token::Record => "record".into(),
        Token::Choice => "choice".into(),
        Token::Const => "const".into(),
        Token::Type => "type".into(),
        Token::Extern => "extern".into(),
        Token::Trait => "trait".into(),
        Token::Impl => "impl".into(),
        Token::Pub => "pub".into(),
        Token::Bring => "bring".into(),
        Token::Async => "async".into(),
        Token::Await => "await".into(),
        Token::Spawn => "spawn".into(),
        Token::Par => "par".into(),
        Token::In => "in".into(),
        Token::To => "to".into(),
        Token::BoolLit(b) => if *b { "true".into() } else { "false".into() },
        // Type keywords
        Token::I64 => "i64".into(),
        Token::I32 => "i32".into(),
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
        Token::Dot => ".".into(),
        Token::DotDot => "..".into(),
        Token::DotDotEq => "..=".into(),
        Token::Arrow => "->".into(),
        Token::FatArrow => "=>".into(),
        Token::Eq => "=".into(),
        Token::EqEq => "==".into(),
        Token::NotEq => "!=".into(),
        Token::LtEq => "<=".into(),
        Token::GtEq => ">=".into(),
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
        Token::Ident(s) => s.clone(),
        Token::IntLit(n) => n.to_string(),
        Token::FloatLit(f) => {
            if f.fract() == 0.0 { format!("{:.1}", f) } else { f.to_string() }
        }
        Token::StringLit(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Token::FStringLit(s) => format!("f\"{}\"", s),
        Token::Eof => String::new(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// LSP protocol server (JSON-RPC over stdin/stdout)
// ---------------------------------------------------------------------------

/// Runs the LSP server, reading JSON-RPC messages from stdin and writing
/// responses to stdout. Blocks until the client sends `exit`.
pub fn run_lsp_server() -> std::io::Result<()> {
    use std::io::Read;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut state = LspState::new();
    #[allow(unused_assignments)]
    let mut request_id: Option<serde_json::Value> = None;

    loop {
        // Read Content-Length header.
        let mut content_length: usize = 0;
        loop {
            let mut byte = [0u8];
            let mut chars = String::new();
            loop {
                stdin.lock().read_exact(&mut byte)?;
                if byte[0] == b'\r' { continue; }
                if byte[0] == b'\n' { break; }
                chars.push(byte[0] as char);
            }
            if chars.is_empty() { break; }
            if chars.to_lowercase().starts_with("content-length:") {
                let val = chars["content-length:".len()..].trim();
                content_length = val.parse().unwrap_or(0);
            }
        }
        if content_length == 0 { continue; }

        // Read body.
        let mut body = vec![0u8; content_length];
        stdin.lock().read_exact(&mut body)?;
        let body_str = String::from_utf8_lossy(&body);

        let msg: serde_json::Value = match serde_json::from_str(&body_str) {
            Ok(v) => v,
            Err(_) => continue,
        };

        request_id = msg.get("id").cloned();
        let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(serde_json::Value::Null);

        match method {
            "initialize" => {
                let resp = make_response(request_id.clone(), serde_json::json!({
                    "capabilities": {
                        "textDocumentSync": 1,
                        "hoverProvider": true,
                        "completionProvider": { "triggerCharacters": ["."] },
                        "definitionProvider": true,
                        "documentSymbolProvider": true,
                        "signatureHelpProvider": { "triggerCharacters": ["(", ","] },
                        "documentFormattingProvider": true
                    },
                    "serverInfo": { "name": "iris-lsp", "version": "0.1.0" }
                }));
                write_message(&mut stdout.lock(), &resp)?;
            }
            "initialized" => { /* no-op */ }
            "textDocument/didOpen" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_owned();
                let text = params["textDocument"]["text"].as_str().unwrap_or("").to_owned();
                let diags = state.open_document(&uri, &text);
                let notif = make_diagnostics_notification(&uri, &diags);
                write_message(&mut stdout.lock(), &notif)?;
            }
            "textDocument/didChange" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_owned();
                let text = params["contentChanges"][0]["text"].as_str()
                    .or_else(|| params["contentChanges"].as_array()
                        .and_then(|a| a.first())
                        .and_then(|c| c["text"].as_str()))
                    .unwrap_or("").to_owned();
                let diags = state.update_document(&uri, &text);
                let notif = make_diagnostics_notification(&uri, &diags);
                write_message(&mut stdout.lock(), &notif)?;
            }
            "textDocument/didSave" => { /* diagnostics already sent on change */ }
            "textDocument/didClose" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
                state.close_document(uri);
            }
            "textDocument/hover" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_owned();
                let line = params["position"]["line"].as_u64().unwrap_or(0) as u32;
                let character = params["position"]["character"].as_u64().unwrap_or(0) as u32;
                let hover_text = state.hover(&uri, line, character);
                let result = hover_text.map(|t| serde_json::json!({
                    "contents": { "kind": "markdown", "value": format!("`{}`", t) }
                })).unwrap_or(serde_json::Value::Null);
                write_message(&mut stdout.lock(), &make_response(request_id.clone(), result))?;
            }
            "textDocument/completion" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_owned();
                let items: Vec<serde_json::Value> = state.completions(&uri).into_iter()
                    .map(|label| serde_json::json!({ "label": label, "kind": 1 }))
                    .collect();
                write_message(&mut stdout.lock(), &make_response(
                    request_id.clone(),
                    serde_json::json!({ "isIncomplete": false, "items": items }),
                ))?;
            }
            "textDocument/definition" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_owned();
                let line = params["position"]["line"].as_u64().unwrap_or(0) as u32;
                let character = params["position"]["character"].as_u64().unwrap_or(0) as u32;
                let result = state.definition(&uri, line, character)
                    .map(|(def_uri, sl, sc, el, ec)| serde_json::json!({
                        "uri": def_uri,
                        "range": {
                            "start": { "line": sl, "character": sc },
                            "end":   { "line": el, "character": ec }
                        }
                    }))
                    .unwrap_or(serde_json::Value::Null);
                write_message(&mut stdout.lock(), &make_response(request_id.clone(), result))?;
            }
            "textDocument/documentSymbol" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_owned();
                let syms: Vec<serde_json::Value> = state.document_symbols(&uri).into_iter()
                    .map(|(name, kind, sl, sc, el, ec)| serde_json::json!({
                        "name": name,
                        "kind": kind,
                        "range": {
                            "start": { "line": sl, "character": sc },
                            "end":   { "line": el, "character": ec }
                        },
                        "selectionRange": {
                            "start": { "line": sl, "character": sc },
                            "end":   { "line": sl, "character": sc + name.len() as u32 }
                        }
                    }))
                    .collect();
                write_message(&mut stdout.lock(), &make_response(
                    request_id.clone(),
                    serde_json::json!(syms),
                ))?;
            }
            "textDocument/signatureHelp" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_owned();
                let line = params["position"]["line"].as_u64().unwrap_or(0) as u32;
                let character = params["position"]["character"].as_u64().unwrap_or(0) as u32;
                let result = state.signature_help(&uri, line, character)
                    .map(|(label, param_labels, active)| serde_json::json!({
                        "signatures": [{
                            "label": label,
                            "parameters": param_labels.iter().map(|p| {
                                serde_json::json!({ "label": p })
                            }).collect::<Vec<_>>()
                        }],
                        "activeSignature": 0,
                        "activeParameter": active
                    }))
                    .unwrap_or(serde_json::Value::Null);
                write_message(&mut stdout.lock(), &make_response(request_id.clone(), result))?;
            }
            "textDocument/formatting" => {
                let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_owned();
                let source = state.documents.get(&uri).cloned().unwrap_or_default();
                let result = state.format(&uri)
                    .map(|formatted| {
                        let line_count = source.lines().count() as u32;
                        let last_line_len = source.lines().last().map(|l| l.len()).unwrap_or(0) as u32;
                        serde_json::json!([{
                            "range": {
                                "start": { "line": 0, "character": 0 },
                                "end":   { "line": line_count, "character": last_line_len }
                            },
                            "newText": formatted
                        }])
                    })
                    .unwrap_or(serde_json::json!([]));
                write_message(&mut stdout.lock(), &make_response(request_id.clone(), result))?;
            }
            "shutdown" => {
                write_message(&mut stdout.lock(), &make_response(request_id.clone(), serde_json::Value::Null))?;
            }
            "exit" => break,
            _ => {
                if request_id.is_some() {
                    write_message(&mut stdout.lock(), &make_response(request_id.clone(), serde_json::Value::Null))?;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn uri_to_module_name(uri: &str) -> String {
    uri.rsplit('/').next()
        .and_then(|f| f.split('.').next())
        .unwrap_or("module")
        .to_owned()
}

/// Convert a `file://` URI to a filesystem `PathBuf`.
/// Returns `None` for non-file URIs (e.g. `untitled:`).
fn uri_to_file_path(uri: &str) -> Option<std::path::PathBuf> {
    // file:///C%3A/Users/… or file:///home/…
    let stripped = uri.strip_prefix("file:///")?;
    // Percent-decode the path.
    let decoded: String = percent_decode(stripped);
    // On Windows the path looks like "C:/Users/…"; on Unix "/home/…".
    // std::path::PathBuf handles both forms.
    let path = std::path::PathBuf::from(&decoded);
    if path.exists() { Some(path) } else { None }
}

/// Minimal %-decode (covers the most common LSP URI escapes).
fn percent_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""),
                16,
            ) {
                out.push(byte as char);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn line_col_to_byte(source: &str, line: u32, character: u32) -> u32 {
    let mut cur_line = 0u32;
    let mut cur_col = 0u32;
    for (i, ch) in source.char_indices() {
        if cur_line == line && cur_col == character {
            return i as u32;
        }
        if ch == '\n' {
            cur_line += 1;
            cur_col = 0;
        } else {
            cur_col += 1;
        }
    }
    source.len() as u32
}

fn ident_at_byte(source: &str, byte: u32) -> Option<&str> {
    let src = source;
    let byte = byte as usize;
    if byte >= src.len() { return None; }
    let start = src[..byte].rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);
    let end = src[byte..].find(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| byte + i)
        .unwrap_or(src.len());
    if start < end { Some(&src[start..end]) } else { None }
}

fn make_response(id: Option<serde_json::Value>, result: serde_json::Value) -> String {
    serde_json::to_string(&serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })).unwrap_or_default()
}

fn make_diagnostics_notification(uri: &str, diags: &[LspDiagnostic]) -> String {
    let json_diags: Vec<serde_json::Value> = diags.iter().map(|d| serde_json::json!({
        "range": {
            "start": { "line": d.line, "character": d.character },
            "end":   { "line": d.end_line, "character": d.end_character },
        },
        "severity": d.severity,
        "message": d.message,
        "source": "iris",
    })).collect();
    serde_json::to_string(&serde_json::json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": json_diags },
    })).unwrap_or_default()
}

fn write_message(writer: &mut impl std::io::Write, body: &str) -> std::io::Result<()> {
    write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
    writer.flush()
}
