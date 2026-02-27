//! Phase 56: File I/O builtins
//!
//! Tests for: file_read_all, file_write_all, file_exists, file_lines

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// file_exists: known-missing file returns false (bool 0)
// ---------------------------------------------------------------------------

#[test]
fn test_file_exists_false() {
    let src = r#"
def f() -> bool {
    file_exists("/nonexistent_iris_test_path_xyz123/missing.txt")
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "false");
}

// ---------------------------------------------------------------------------
// file_exists: IR text contains file_exists
// ---------------------------------------------------------------------------

#[test]
fn test_file_exists_ir() {
    let src = r#"
def f() -> bool {
    file_exists("/nonexistent.txt")
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("file_exists"), "expected file_exists in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// file_write_all + file_read_all + file_exists round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_file_write_read_roundtrip() {
    use std::env::temp_dir;
    let tmp = temp_dir().join("iris_test_phase56_rw.txt");
    let path = tmp.to_str().unwrap().replace('\\', "/");

    let src = format!(r#"
def f() -> i64 {{
    val wr = file_write_all("{path}", "hello iris")
    val rd = file_read_all("{path}")
    42
}}
"#);

    let result = compile(&src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");

    // Clean up
    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// file_read_all returns err for missing file
// ---------------------------------------------------------------------------

#[test]
fn test_file_read_missing() {
    let src = r#"
def f() -> bool {
    val r = file_read_all("/nonexistent_iris_xyz/missing.txt")
    is_ok(r)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "false");
}

// ---------------------------------------------------------------------------
// file_lines: IR text contains file_lines
// ---------------------------------------------------------------------------

#[test]
fn test_file_lines_ir() {
    let src = r#"
def f() -> i64 {
    val ls = file_lines("/nonexistent.txt")
    list_len(ls)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("file_lines"), "expected file_lines in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// file_lines: missing file returns empty list
// ---------------------------------------------------------------------------

#[test]
fn test_file_lines_missing() {
    let src = r#"
def f() -> i64 {
    val ls = file_lines("/nonexistent_iris_xyz/missing.txt")
    list_len(ls)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ---------------------------------------------------------------------------
// file_write_all + file_lines round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_file_lines_content() {
    use std::env::temp_dir;
    let tmp = temp_dir().join("iris_test_phase56_lines.txt");
    let path = tmp.to_str().unwrap().replace('\\', "/");

    // Write two lines then read them back
    let src = format!(r#"
def f() -> i64 {{
    val wr = file_write_all("{path}", "line1\nline2\nline3")
    val ls = file_lines("{path}")
    list_len(ls)
}}
"#);

    let result = compile(&src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// LLVM IR contains file I/O declare stubs
// ---------------------------------------------------------------------------

#[test]
fn test_file_io_llvm() {
    let src = r#"
def f() -> bool {
    file_exists("/tmp/iris_test.txt")
}
"#;
    let ll = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(ll.contains("iris_file_exists"), "expected iris_file_exists in LLVM:\n{}", ll);
}

// ---------------------------------------------------------------------------
// file_write_all + file_exists: file exists after writing
// ---------------------------------------------------------------------------

#[test]
fn test_file_exists_after_write() {
    use std::env::temp_dir;
    let tmp = temp_dir().join("iris_test_phase56_exists.txt");
    let path = tmp.to_str().unwrap().replace('\\', "/");

    let src = format!(r#"
def f() -> bool {{
    val wr = file_write_all("{path}", "data")
    file_exists("{path}")
}}
"#);

    let result = compile(&src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "true");

    let _ = std::fs::remove_file(&tmp);
}
