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

        // Find the identifier at (line, character) by scanning the source.
        let byte = line_col_to_byte(source, line, character);
        let ident = ident_at_byte(source, byte)?;

        // Look for a function with that name.
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

        // Add dynamic completions from the compiled module.
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

    // ------------------------------------------------------------------
    // Private
    // ------------------------------------------------------------------

    fn diagnose(&self, uri: &str) -> Vec<LspDiagnostic> {
        let Some(source) = self.documents.get(uri) else { return Vec::new() };
        let module_name = uri_to_module_name(uri);
        let mut diags = Vec::new();

        // Errors from compilation.
        if let Err(e) = crate::compile(source, &module_name, EmitKind::Ir) {
            let (line, character) = if let Some(byte) = error_byte_offset(&e) {
                let (l, c) = byte_to_line_col(source, byte);
                // Convert to 0-based.
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

        // Warnings from lint pass.
        if let Ok((_, warnings)) = crate::compile_with_warnings(source, &module_name, EmitKind::Ir) {
            for w in warnings {
                diags.push(LspDiagnostic {
                    line: 0, character: 0, end_line: 0, end_character: 1,
                    message: w.message,
                    severity: 2,
                });
            }
        }

        diags
    }
}

// ---------------------------------------------------------------------------
// LSP protocol server (JSON-RPC over stdin/stdout)
// ---------------------------------------------------------------------------

/// Runs the LSP server, reading JSON-RPC messages from stdin and writing
/// responses to stdout. Blocks until the client sends `exit`.
///
/// Use `iris lsp` to start this server; configure your editor to use
/// `iris lsp` as the language server command for `.iris` files.
pub fn run_lsp_server() -> std::io::Result<()> {
    use std::io::{Read, Write};
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut state = LspState::new();
    let mut request_id: Option<serde_json::Value> = None;

    loop {
        // Read Content-Length header.
        let mut header = String::new();
        let mut content_length: usize = 0;
        loop {
            let mut line = String::new();
            let mut byte = [0u8];
            let mut chars = String::new();
            // Read until \r\n\r\n
            chars.clear();
            loop {
                stdin.lock().read_exact(&mut byte)?;
                if byte[0] == b'\r' { continue; }
                if byte[0] == b'\n' { break; }
                chars.push(byte[0] as char);
            }
            if chars.is_empty() { break; } // blank line = end of headers
            if chars.to_lowercase().starts_with("content-length:") {
                let val = chars["content-length:".len()..].trim();
                content_length = val.parse().unwrap_or(0);
            }
            let _ = header;
            let _ = line;
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
                        "completionProvider": { "triggerCharacters": ["."] }
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
            "shutdown" => {
                write_message(&mut stdout.lock(), &make_response(request_id.clone(), serde_json::Value::Null))?;
            }
            "exit" => break,
            _ => {
                // Unknown request — send null response to avoid client hanging.
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
    // Extract stem from file:///path/to/foo.iris → "foo"
    uri.rsplit('/').next()
        .and_then(|f| f.split('.').next())
        .unwrap_or("module")
        .to_owned()
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
    // Walk left to start of identifier.
    let start = src[..byte].rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);
    // Walk right to end of identifier.
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
