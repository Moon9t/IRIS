//! Phase 63 integration tests: extended numeric types (u32, u64, usize, i8, u8).

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. u32 arithmetic
// ---------------------------------------------------------------------------
#[test]
fn test_u32_arithmetic() {
    let src = r#"
def add_u32(a: u32, b: u32) -> u32 { a + b }
def f() -> i64 {
    add_u32(10, 20) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "30");
}

// ---------------------------------------------------------------------------
// 2. u64 arithmetic
// ---------------------------------------------------------------------------
#[test]
fn test_u64_arithmetic() {
    let src = r#"
def add_u64(a: u64, b: u64) -> u64 { a + b }
def f() -> i64 {
    add_u64(100, 200) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "300");
}

// ---------------------------------------------------------------------------
// 3. usize arithmetic
// ---------------------------------------------------------------------------
#[test]
fn test_usize_arithmetic() {
    let src = r#"
def add_sz(a: usize, b: usize) -> usize { a + b }
def f() -> i64 {
    add_sz(7, 8) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15");
}

// ---------------------------------------------------------------------------
// 4. i8 arithmetic
// ---------------------------------------------------------------------------
#[test]
fn test_i8_arithmetic() {
    let src = r#"
def add_i8(a: i8, b: i8) -> i8 { a + b }
def f() -> i64 {
    add_i8(5, 3) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "8");
}

// ---------------------------------------------------------------------------
// 5. u8 arithmetic
// ---------------------------------------------------------------------------
#[test]
fn test_u8_arithmetic() {
    let src = r#"
def add_u8(a: u8, b: u8) -> u8 { a + b }
def f() -> i64 {
    add_u8(100, 50) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "150");
}

// ---------------------------------------------------------------------------
// 6. u32 comparison
// ---------------------------------------------------------------------------
#[test]
fn test_u32_comparison() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def gt_u32(a: u32, b: u32) -> bool { a > b }
def f() -> i64 {
    bool_to_i64(gt_u32(50, 30)) + bool_to_i64(gt_u32(10, 20))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 7. u32 multiplication
// ---------------------------------------------------------------------------
#[test]
fn test_u32_multiply() {
    let src = r#"
def mul_u32(a: u32, b: u32) -> u32 { a * b }
def f() -> i64 {
    mul_u32(6, 7) to i64
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// 8. Type in IR text contains new type names
// ---------------------------------------------------------------------------
#[test]
fn test_numeric_types_in_ir() {
    let src = r#"
def add_u32(a: u32, b: u32) -> u32 { a + b }
def add_usize(a: usize, b: usize) -> usize { a + b }
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("u32") || ir.contains("usize"),
        "expected new type names in IR: {}", ir);
}
