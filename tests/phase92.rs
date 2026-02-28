//! Phase 92: LSP — language-server diagnostics, hover, and completions.

use iris::{LspDiagnostic, LspState};

const URI: &str = "file:///test/file.iris";

// ── 1. Parse error → LspDiagnostic with 0-based line ───────────────────────

#[test]
fn test_lsp_parse_error_diagnostic() {
    let mut lsp = LspState::new();
    let diags = lsp.open_document(URI, "def (((broken");
    assert!(!diags.is_empty(), "expected at least one diagnostic for broken source");
    assert_eq!(diags[0].severity, 1, "parse error should have severity=1 (Error)");
}

// ── 2. Lower error (undefined variable) → diagnostic with position ──────────

#[test]
fn test_lsp_lower_error_diagnostic() {
    let mut lsp = LspState::new();
    let src = "def f() -> i64 { undefined_var }";
    let diags = lsp.open_document(URI, src);
    assert!(!diags.is_empty(), "expected diagnostic for undefined variable");
    assert_eq!(diags[0].severity, 1);
}

// ── 3. Valid source → empty diagnostics ─────────────────────────────────────

#[test]
fn test_lsp_valid_source_no_diagnostics() {
    let mut lsp = LspState::new();
    let src = "def add(a: i64, b: i64) -> i64 { a + b }";
    let diags = lsp.open_document(URI, src);
    assert!(diags.is_empty(), "valid source should produce no diagnostics, got: {:?}", diags.iter().map(|d| &d.message).collect::<Vec<_>>());
}

// ── 4. Unknown function call → diagnostic ───────────────────────────────────

#[test]
fn test_lsp_unknown_function_call_diagnostic() {
    let mut lsp = LspState::new();
    let src = "def f() -> i64 { no_such_fn(1, 2) }";
    let diags = lsp.open_document(URI, src);
    assert!(!diags.is_empty(), "expected diagnostic for unknown function call");
}

// ── 5. Hover on defined function → returns signature string with "->" ───────

#[test]
fn test_lsp_hover_function_signature() {
    let mut lsp = LspState::new();
    let src = "def add(a: i64, b: i64) -> i64 { a + b }";
    lsp.open_document(URI, src);
    // "add" starts at byte 4, line 0, character 4 (0-based).
    let hover = lsp.hover(URI, 0, 4);
    let sig = hover.expect("expected hover result for 'add'");
    assert!(sig.contains("->"), "hover signature should contain '->': {}", sig);
    assert!(sig.contains("add"), "hover signature should contain function name: {}", sig);
}

// ── 6. Completions include static keywords ───────────────────────────────────

#[test]
fn test_lsp_completions_include_keywords() {
    let mut lsp = LspState::new();
    lsp.open_document(URI, "def f() -> i64 { 0 }");
    let completions = lsp.completions(URI);
    assert!(completions.contains(&"def".to_string()), "completions should include 'def'");
    assert!(completions.contains(&"val".to_string()), "completions should include 'val'");
    assert!(completions.contains(&"for".to_string()), "completions should include 'for'");
}

// ── 7. Completions include user-defined function name ───────────────────────

#[test]
fn test_lsp_completions_include_user_fn() {
    let mut lsp = LspState::new();
    let src = "def my_custom_fn(x: i64) -> i64 { x + 1 }";
    lsp.open_document(URI, src);
    let completions = lsp.completions(URI);
    assert!(
        completions.contains(&"my_custom_fn".to_string()),
        "completions should include user-defined function 'my_custom_fn'"
    );
}

// ── 8. update_document replaces diagnostics ──────────────────────────────────

#[test]
fn test_lsp_update_document_refreshes_diagnostics() {
    let mut lsp = LspState::new();
    // First: broken source → has errors.
    let bad_diags = lsp.open_document(URI, "def (broken");
    assert!(!bad_diags.is_empty(), "expected errors on broken source");
    // Then: fix the source → no errors.
    let good_diags = lsp.update_document(URI, "def f() -> i64 { 42 }");
    assert!(good_diags.is_empty(), "expected no errors after fixing source");
}
