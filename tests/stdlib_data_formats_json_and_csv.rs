//! Phase 98 integration tests: stdlib data formats — JSON and CSV.

use iris::{compile_multi, EmitKind};

// ── 1. json_parse on {"key":"value"} — json_get returns the value ────────────
#[test]
fn test_json_parse_get() {
    let src = r#"
bring std.json
def f() -> str {
    val pairs = json_parse("{\"key\":\"value\"}")
    json_get(pairs, "key")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "value");
}

// ── 2. json_obj emits a valid JSON object string ─────────────────────────────
#[test]
fn test_json_obj_emit() {
    let src = r#"
bring std.json
def f() -> str {
    var keys = list()
    push(keys, "name");
    var vals = list()
    push(vals, json_str("alice"));
    json_obj(keys, vals)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), r#"{"name":"alice"}"#);
}

// ── 3. json_arr emits a valid JSON array string ───────────────────────────────
#[test]
fn test_json_arr_emit() {
    let src = r#"
bring std.json
def f() -> str {
    var items = list()
    push(items, "1");
    push(items, "2");
    push(items, "3");
    json_arr(items)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "[1,2,3]");
}

// ── 4. csv_row_count on "a,b\nc,d" returns 2 ────────────────────────────────
#[test]
fn test_csv_row_count() {
    let src = r#"
bring std.csv
def f() -> i64 {
    csv_row_count("a,b\nc,d")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 5. csv_col_count on "a,b\nc,d" returns 2 ────────────────────────────────
#[test]
fn test_csv_col_count() {
    let src = r#"
bring std.csv
def f() -> i64 {
    csv_col_count("a,b\nc,d")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 6. csv_get_row extracts the correct row ──────────────────────────────────
#[test]
fn test_csv_get_row() {
    let src = r#"
bring std.csv
def f() -> str {
    val row = csv_get_row("a,b\nc,d", 1)
    list_get(row, 0)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "c");
}

// ── 7. json_set then json_get round-trips ────────────────────────────────────
#[test]
fn test_json_set_get() {
    let src = r#"
bring std.json
def f() -> str {
    val pairs = json_new()
    val pairs2 = json_set(pairs, "x", "42")
    json_get(pairs2, "x")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 8. csv_emit_rows(csv_rows(text)) round-trips ─────────────────────────────
#[test]
fn test_csv_roundtrip() {
    let src = r#"
bring std.csv
def f() -> str {
    val text = "a,b\nc,d"
    val rows = csv_rows(text)
    csv_emit_rows(rows)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "a,b\nc,d");
}
