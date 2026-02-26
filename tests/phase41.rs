//! Phase 41 integration tests: `parse_i64(s)` and `parse_f64(s)` builtins.
//!
//! parse_i64(s) -> option<i64>: returns some(n) if s is a valid integer, none otherwise.
//! parse_f64(s) -> option<f64>: returns some(x) if s is a valid float, none otherwise.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. parse_i64 of a valid integer string returns some
// ---------------------------------------------------------------------------
#[test]
fn test_parse_i64_valid() {
    let src = r#"
def f() -> i64 {
    val r = parse_i64("42")
    when r {
        some(n) => n,
        none => -1,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "parse_i64(\"42\") should be some(42), got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 2. parse_i64 of an invalid string returns none
// ---------------------------------------------------------------------------
#[test]
fn test_parse_i64_invalid() {
    let src = r#"
def f() -> i64 {
    val r = parse_i64("abc")
    when r {
        some(n) => n,
        none => -1,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "-1", "parse_i64(\"abc\") should be none, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. parse_f64 of a valid float string returns some
// ---------------------------------------------------------------------------
#[test]
fn test_parse_f64_valid() {
    let src = r#"
def f() -> f64 {
    val r = parse_f64("3.14")
    when r {
        some(x) => x,
        none => -1.0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 3.14).abs() < 1e-9, "parse_f64(\"3.14\") should be ~3.14, got: {}", v);
}

// ---------------------------------------------------------------------------
// 4. parse_f64 of an invalid string returns none
// ---------------------------------------------------------------------------
#[test]
fn test_parse_f64_invalid() {
    let src = r#"
def f() -> f64 {
    val r = parse_f64("xyz")
    when r {
        some(x) => x,
        none => -1.0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - (-1.0)).abs() < 1e-9, "parse_f64(\"xyz\") should be none (-1.0), got: {}", v);
}

// ---------------------------------------------------------------------------
// 5. parse_i64 compiles to IR with parse_i64 instr
// ---------------------------------------------------------------------------
#[test]
fn test_parse_i64_ir() {
    let src = r#"
def f() -> i64 {
    val r = parse_i64("0")
    when r {
        some(n) => n,
        none => 0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should emit IR");
    assert!(
        out.contains("parse_i64"),
        "IR should contain parse_i64, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 6. parse_f64 compiles to IR with parse_f64 instr
// ---------------------------------------------------------------------------
#[test]
fn test_parse_f64_ir() {
    let src = r#"
def f() -> f64 {
    val r = parse_f64("0.0")
    when r {
        some(x) => x,
        none => 0.0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should emit IR");
    assert!(
        out.contains("parse_f64"),
        "IR should contain parse_f64, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 7. parse_i64 in LLVM stub calls iris_parse_i64
// ---------------------------------------------------------------------------
#[test]
fn test_parse_i64_llvm() {
    let src = r#"
def f() -> i64 {
    val r = parse_i64("1")
    when r {
        some(n) => n,
        none => 0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Llvm).expect("should emit LLVM stub");
    assert!(
        out.contains("iris_parse_i64"),
        "LLVM stub should call iris_parse_i64, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 8. parse_i64 of negative number string works
// ---------------------------------------------------------------------------
#[test]
fn test_parse_i64_negative() {
    let src = r#"
def f() -> i64 {
    val r = parse_i64("-99")
    when r {
        some(n) => n,
        none => 0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "-99", "parse_i64(\"-99\") should be -99, got: {}", out.trim());
}
