//! Phase 92: LSP — language-server diagnostics, hover, and completions.

use iris::LspState;

const URI: &str = "file:///test/file.iris";

// ── 1. Parse error → LspDiagnostic with 0-based line ───────────────────────

#[test]
fn test_lsp_parse_error_diagnostic() {
    let mut lsp = LspState::new();
    let diags = lsp.open_document(URI, "def (((broken");
    assert!(
        !diags.is_empty(),
        "expected at least one diagnostic for broken source"
    );
    assert_eq!(
        diags[0].severity, 1,
        "parse error should have severity=1 (Error)"
    );
}

// ── 2. Lower error (undefined variable) → diagnostic with position ──────────

#[test]
fn test_lsp_lower_error_diagnostic() {
    let mut lsp = LspState::new();
    let src = "def f() -> i64 { undefined_var }";
    let diags = lsp.open_document(URI, src);
    assert!(
        !diags.is_empty(),
        "expected diagnostic for undefined variable"
    );
    assert_eq!(diags[0].severity, 1);
}

// ── 3. Valid source → empty diagnostics ─────────────────────────────────────

#[test]
fn test_lsp_valid_source_no_diagnostics() {
    let mut lsp = LspState::new();
    let src = "def add(a: i64, b: i64) -> i64 { a + b }";
    let diags = lsp.open_document(URI, src);
    assert!(
        diags.is_empty(),
        "valid source should produce no diagnostics, got: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

// ── 4. Unknown function call → diagnostic ───────────────────────────────────

#[test]
fn test_lsp_unknown_function_call_diagnostic() {
    let mut lsp = LspState::new();
    let src = "def f() -> i64 { no_such_fn(1, 2) }";
    let diags = lsp.open_document(URI, src);
    assert!(
        !diags.is_empty(),
        "expected diagnostic for unknown function call"
    );
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
    assert!(
        sig.contains("->"),
        "hover signature should contain '->': {}",
        sig
    );
    assert!(
        sig.contains("add"),
        "hover signature should contain function name: {}",
        sig
    );
}

// ── 6. Completions include static keywords ───────────────────────────────────

#[test]
fn test_lsp_completions_include_keywords() {
    let mut lsp = LspState::new();
    lsp.open_document(URI, "def f() -> i64 { 0 }");
    let completions = lsp.completions(URI);
    assert!(
        completions.contains(&"def".to_string()),
        "completions should include 'def'"
    );
    assert!(
        completions.contains(&"val".to_string()),
        "completions should include 'val'"
    );
    assert!(
        completions.contains(&"for".to_string()),
        "completions should include 'for'"
    );
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
    assert!(
        good_diags.is_empty(),
        "expected no errors after fixing source"
    );
}

// ── 9. Hover and Completions for Local val/var bindings with types ───────

#[test]
fn test_lsp_hover_and_completions_local_bindings() {
    let mut lsp = LspState::new();
    let src = "def test_fn() -> i64 {\n    val x = 10;\n    var y = 20.0;\n    val (a, b) = (30, true);\n    x\n}";
    lsp.open_document(URI, src);

    // Hover on 'x' at its usage (line 4, character 4)
    let hover_x = lsp.hover(URI, 4, 4).expect("expected hover on x");
    assert!(hover_x.contains("val x: i64"));
    assert!(hover_x.contains("Immutable local binding"));

    // Hover on 'y' at its definition (line 2, character 8)
    let hover_y = lsp.hover(URI, 2, 8).expect("expected hover on y");
    assert!(hover_y.contains("var y: f64"));
    assert!(hover_y.contains("Mutable local variable"));

    // Hover on destructured tuple element 'a' at definition (line 3, character 9)
    let hover_a = lsp.hover(URI, 3, 9).expect("expected hover on a");
    assert!(hover_a.contains("val a: i64"));
    assert!(hover_a.contains("Immutable local binding"));

    // Completions detail checks
    let completions = lsp.completion_items(URI);
    let x_comp = completions.iter().find(|c| c.label == "x").expect("expected completion item for x");
    assert_eq!(x_comp.detail.as_deref(), Some("val: i64"));

    let y_comp = completions.iter().find(|c| c.label == "y").expect("expected completion item for y");
    assert_eq!(y_comp.detail.as_deref(), Some("var: f64"));
}
