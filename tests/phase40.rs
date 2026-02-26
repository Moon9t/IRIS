//! Phase 40 integration tests: user input builtins — `read_line()`, `read_i64()`, `read_f64()`.
//!
//! These builtins read from stdin at runtime. Tests use `EmitKind::Ir` and
//! `EmitKind::Llvm` (structural checks) to avoid hanging on actual stdin reads.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. read_line() compiles and appears in IR
// ---------------------------------------------------------------------------
#[test]
fn test_read_line_ir() {
    let src = r#"
def f() -> str {
    read_line()
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("read_line"),
        "IR should contain read_line, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 2. read_i64() compiles and appears in IR
// ---------------------------------------------------------------------------
#[test]
fn test_read_i64_ir() {
    let src = r#"
def f() -> i64 {
    read_i64()
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("read_i64"),
        "IR should contain read_i64, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 3. read_f64() compiles and appears in IR
// ---------------------------------------------------------------------------
#[test]
fn test_read_f64_ir() {
    let src = r#"
def f() -> f64 {
    read_f64()
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("read_f64"),
        "IR should contain read_f64, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 4. read_line() in LLVM stub calls iris_read_line
// ---------------------------------------------------------------------------
#[test]
fn test_read_line_llvm() {
    let src = r#"
def f() -> str {
    read_line()
}
"#;
    let out = compile(src, "test", EmitKind::Llvm).expect("should emit LLVM stub");
    assert!(
        out.contains("iris_read_line"),
        "LLVM stub should call iris_read_line, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 5. read_i64() in LLVM stub calls iris_read_i64
// ---------------------------------------------------------------------------
#[test]
fn test_read_i64_llvm() {
    let src = r#"
def f() -> i64 {
    read_i64()
}
"#;
    let out = compile(src, "test", EmitKind::Llvm).expect("should emit LLVM stub");
    assert!(
        out.contains("iris_read_i64"),
        "LLVM stub should call iris_read_i64, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 6. read_f64() in LLVM stub calls iris_read_f64
// ---------------------------------------------------------------------------
#[test]
fn test_read_f64_llvm() {
    let src = r#"
def f() -> f64 {
    read_f64()
}
"#;
    let out = compile(src, "test", EmitKind::Llvm).expect("should emit LLVM stub");
    assert!(
        out.contains("iris_read_f64"),
        "LLVM stub should call iris_read_f64, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 7. read_i64() result can be used in arithmetic (IR check)
// ---------------------------------------------------------------------------
#[test]
fn test_read_i64_in_arithmetic_ir() {
    let src = r#"
def f() -> i64 {
    val n = read_i64()
    n * 2
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("read_i64") && out.contains("mul"),
        "IR should contain read_i64 and mul, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 8. read_line() result can be passed to print (IR check)
// ---------------------------------------------------------------------------
#[test]
fn test_read_line_to_print_ir() {
    let src = r#"
def f() -> i64 {
    val s = read_line()
    val _ = print(s)
    0
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("read_line") && out.contains("print"),
        "IR should contain read_line and print, got:\n{}",
        out
    );
}
