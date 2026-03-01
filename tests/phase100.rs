//! Phase 100 integration tests: stdlib KV store and in-memory table.

use iris::{compile_multi, EmitKind};

// ── 1. kv_set + kv_get round-trips a string value ───────────────────────────
#[test]
fn test_kv_set_get() {
    let src = r#"
bring std.kv
def f() -> str {
    val path = "/tmp/iris_test_kv1.txt"
    val _ = kv_set(path, "hello", "world")
    kv_get(path, "hello")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "world");
}

// ── 2. kv_delete removes a key ───────────────────────────────────────────────
#[test]
fn test_kv_delete() {
    let src = r#"
bring std.kv
def f() -> str {
    val path = "/tmp/iris_test_kv2.txt"
    val _ = kv_set(path, "foo", "bar")
    val _ = kv_delete(path, "foo")
    kv_get(path, "foo")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "");
}

// ── 3. kv_keys returns correct key count ────────────────────────────────────
#[test]
fn test_kv_keys() {
    let src = r#"
bring std.kv
def f() -> i64 {
    val path = "/tmp/iris_test_kv3.txt"
    val _ = kv_set(path, "a", "1")
    val _ = kv_set(path, "b", "2")
    list_len(kv_keys(path))
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 4. table_new creates empty table ────────────────────────────────────────
#[test]
fn test_table_new() {
    let src = r#"
bring std.table
def f() -> i64 {
    var cols = list()
    push(cols, "name");
    push(cols, "age");
    val t = table_new(cols)
    table_len(t)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ── 5. table_insert + table_len shows correct count ─────────────────────────
#[test]
fn test_table_insert_len() {
    let src = r#"
bring std.table
def f() -> i64 {
    var cols = list()
    push(cols, "name");
    push(cols, "score");
    val t0 = table_new(cols)
    var r1 = list()
    push(r1, "alice");
    push(r1, "90");
    val t1 = table_insert(t0, r1)
    var r2 = list()
    push(r2, "bob");
    push(r2, "85");
    val t2 = table_insert(t1, r2)
    table_len(t2)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 6. table_select extracts one column ─────────────────────────────────────
#[test]
fn test_table_select() {
    let src = r#"
bring std.table
def f() -> str {
    var cols = list()
    push(cols, "name");
    push(cols, "score");
    val t0 = table_new(cols)
    var r1 = list()
    push(r1, "alice");
    push(r1, "90");
    val t1 = table_insert(t0, r1)
    var r2 = list()
    push(r2, "bob");
    push(r2, "85");
    val t2 = table_insert(t1, r2)
    val names = table_select(t2, "name")
    list_get(names, 0)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "alice");
}

// ── 7. table_where filters rows by column value ──────────────────────────────
#[test]
fn test_table_where() {
    let src = r#"
bring std.table
def f() -> i64 {
    var cols = list()
    push(cols, "name");
    push(cols, "dept");
    val t0 = table_new(cols)
    var r1 = list()
    push(r1, "alice");
    push(r1, "eng");
    val t1 = table_insert(t0, r1)
    var r2 = list()
    push(r2, "bob");
    push(r2, "hr");
    val t2 = table_insert(t1, r2)
    var r3 = list()
    push(r3, "carol");
    push(r3, "eng");
    val t3 = table_insert(t2, r3)
    val eng = table_where(t3, "dept", "eng")
    table_len(eng)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 8. table_to_csv produces valid CSV output ────────────────────────────────
#[test]
fn test_table_to_csv() {
    let src = r#"
bring std.table
def f() -> str {
    var cols = list()
    push(cols, "x");
    push(cols, "y");
    val t0 = table_new(cols)
    var r1 = list()
    push(r1, "1");
    push(r1, "2");
    val t1 = table_insert(t0, r1)
    table_to_csv(t1)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    let trimmed = result.trim();
    assert!(trimmed.starts_with("x,y"), "expected CSV starting with 'x,y', got: {}", trimmed);
    assert!(trimmed.contains("1,2"), "expected row data '1,2' in CSV, got: {}", trimmed);
}
