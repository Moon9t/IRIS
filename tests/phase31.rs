//! Phase 31 integration tests: `sparse<T>` — sparse tensor representation.
//!
//! sparsify(arr) converts a dense array to a sparse representation that stores
//! only non-zero (index, value) pairs.
//! densify(sparse) converts the sparse representation back; in the interpreter
//! it returns the number of non-zero elements (nnz) as an i64.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. sparsify produces a Sparsify instruction in IR
// ---------------------------------------------------------------------------
#[test]
fn test_sparsify_ir() {
    let src = r#"
def f() -> i64 {
    val arr = [1, 0, 3, 0, 5]
    val s = sparsify(arr)
    densify(s)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(
        ir.contains("sparsify"),
        "IR should contain 'sparsify', got:\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 2. densify produces a Densify instruction in IR
// ---------------------------------------------------------------------------
#[test]
fn test_densify_ir() {
    let src = r#"
def f() -> i64 {
    val arr = [1, 0, 3]
    val s = sparsify(arr)
    densify(s)
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(
        ir.contains("densify"),
        "IR should contain 'densify', got:\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 3. densify(sparsify([1,0,3,0,5])) == 3  (three non-zeros)
// ---------------------------------------------------------------------------
#[test]
fn test_sparse_nnz_three() {
    let src = r#"
def f() -> i64 {
    val arr = [1, 0, 3, 0, 5]
    val s = sparsify(arr)
    densify(s)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "3", "nnz should be 3, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. densify(sparsify([0,0,0])) == 0  (all zeros)
// ---------------------------------------------------------------------------
#[test]
fn test_sparse_all_zeros() {
    let src = r#"
def f() -> i64 {
    val arr = [0, 0, 0]
    val s = sparsify(arr)
    densify(s)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "0", "all-zero array nnz should be 0, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. densify(sparsify([1,2,3])) == 3  (all non-zero)
// ---------------------------------------------------------------------------
#[test]
fn test_sparse_all_nonzero() {
    let src = r#"
def f() -> i64 {
    val arr = [1, 2, 3]
    val s = sparsify(arr)
    densify(s)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "3", "all-nonzero array nnz should be 3, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. densify(sparsify([0,7,0,0,9,0])) == 2
// ---------------------------------------------------------------------------
#[test]
fn test_sparse_two_nonzero() {
    let src = r#"
def f() -> i64 {
    val arr = [0, 7, 0, 0, 9, 0]
    val s = sparsify(arr)
    densify(s)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "2", "nnz should be 2, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. Single-element sparse: [42] → nnz == 1
// ---------------------------------------------------------------------------
#[test]
fn test_sparse_single_nonzero() {
    let src = r#"
def f() -> i64 {
    val arr = [42]
    val s = sparsify(arr)
    densify(s)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "1", "single non-zero nnz should be 1, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. sparse<T> type annotation compiles
// ---------------------------------------------------------------------------
#[test]
fn test_sparse_type_annotation_compiles() {
    let src = r#"
def f() -> i64 {
    val arr = [10, 0, 20, 0, 30]
    val s: sparse<[i64; 5]> = sparsify(arr)
    densify(s)
}
"#;
    // Just verify it compiles without error.
    compile(src, "test", EmitKind::Ir).expect("sparse<T> type annotation should compile");
}
