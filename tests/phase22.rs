//! Phase 22 integration tests: Array types `[T; N]`, array literals, indexing.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Array literal IR emission
// ---------------------------------------------------------------------------
#[test]
fn test_array_lit_ir() {
    let src = r#"
def f() -> i64 {
    val arr = [1, 2, 3, 4, 5]
    arr[0]
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(ir.contains("alloc_array"), "IR should contain alloc_array, got:\n{}", ir);
}

// ---------------------------------------------------------------------------
// 2. Array literal element access eval
// ---------------------------------------------------------------------------
#[test]
fn test_array_index_eval() {
    let src = r#"
def f() -> i64 {
    val arr = [10, 20, 30]
    arr[1]
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "20", "arr[1] should be 20, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. Array first element
// ---------------------------------------------------------------------------
#[test]
fn test_array_first_element() {
    let src = r#"
def f() -> i64 {
    val arr = [42, 99, 0]
    arr[0]
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "arr[0] should be 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. Array last element
// ---------------------------------------------------------------------------
#[test]
fn test_array_last_element() {
    let src = r#"
def f() -> i64 {
    val arr = [1, 2, 3, 4, 5]
    arr[4]
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "5", "arr[4] should be 5, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. Array store (mutation via reassign)
// ---------------------------------------------------------------------------
#[test]
fn test_array_store_eval() {
    let src = r#"
def f() -> i64 {
    var arr = [1, 2, 3]
    arr[1] = 99
    arr[1]
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "99", "arr[1] after store should be 99, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. Array in IR contains array_load
// ---------------------------------------------------------------------------
#[test]
fn test_array_load_ir() {
    let src = r#"
def f() -> i64 {
    val arr = [7, 8, 9]
    arr[2]
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(ir.contains("array_load"), "IR should contain array_load, got:\n{}", ir);
}

// ---------------------------------------------------------------------------
// 7. Array of floats
// ---------------------------------------------------------------------------
#[test]
fn test_array_of_floats() {
    let src = r#"
def f() -> f32 {
    val arr = [1.0, 2.5, 3.14]
    arr[1]
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // The value should be approximately 2.5
    let v: f32 = out.trim().parse().expect("should parse as float");
    assert!((v - 2.5f32).abs() < 1e-5, "arr[1] should be 2.5, got: {}", v);
}

// ---------------------------------------------------------------------------
// 8. Sum of array elements using for loop
// ---------------------------------------------------------------------------
#[test]
fn test_array_sum_loop() {
    let src = r#"
def f() -> i64 {
    val arr = [1, 2, 3, 4, 5]
    var sum = 0
    for i in 0..5 {
        sum = sum + arr[i]
    }
    sum
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "15", "sum of [1,2,3,4,5] should be 15, got: {}", out.trim());
}
