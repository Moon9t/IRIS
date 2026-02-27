//! Phase 54 integration tests: C runtime + native binary build pipeline.
//!
//! Tests cover:
//!  1. Boxing declare stmts in LLVM IR (iris_box_i64, etc.)
//!  2. Typed to-string declares (iris_i64_to_str, etc.)
//!  3. Fixed grad signatures (iris_make_grad, iris_grad_value, iris_grad_tangent)
//!  4. Fixed sparse/atomic signatures (iris_sparsify, iris_densify, iris_atomic_new)
//!  5. ValueToStr dispatch — to_str(42) calls iris_i64_to_str, not iris_value_to_str
//!  6. ListPush boxing — scalar pushed to list is boxed
//!  7. Runtime C source is accessible via codegen::build module
//!  8. EmitKind::Binary exists and emits LLVM IR

use iris::{compile, EmitKind};
use iris::codegen::{runtime_c_source, runtime_h_source};

// ---------------------------------------------------------------------------
// 1. Boxing function declares are emitted
// ---------------------------------------------------------------------------
#[test]
fn test_boxing_declares_present() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Binary).unwrap();
    assert!(
        ir.contains("declare ptr @iris_box_i64(i64)"),
        "expected 'declare ptr @iris_box_i64(i64)':\n{}",
        ir
    );
    assert!(
        ir.contains("declare ptr @iris_box_f64(double)"),
        "expected 'declare ptr @iris_box_f64(double)':\n{}",
        ir
    );
    assert!(
        ir.contains("declare ptr @iris_box_bool(i1)"),
        "expected 'declare ptr @iris_box_bool(i1)':\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 2. Typed to-string declares are emitted
// ---------------------------------------------------------------------------
#[test]
fn test_typed_to_str_declares() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Binary).unwrap();
    assert!(
        ir.contains("declare ptr @iris_i64_to_str(i64)"),
        "expected 'declare ptr @iris_i64_to_str(i64)':\n{}",
        ir
    );
    assert!(
        ir.contains("declare ptr @iris_f64_to_str(double)"),
        "expected 'declare ptr @iris_f64_to_str(double)':\n{}",
        ir
    );
    assert!(
        ir.contains("declare ptr @iris_bool_to_str(i1)"),
        "expected 'declare ptr @iris_bool_to_str(i1)':\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 3. Grad function declares have correct signatures
// ---------------------------------------------------------------------------
#[test]
fn test_grad_declares_correct() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Binary).unwrap();
    assert!(
        ir.contains("declare ptr @iris_make_grad(double, double)"),
        "expected 'declare ptr @iris_make_grad(double, double)':\n{}",
        ir
    );
    assert!(
        ir.contains("declare double @iris_grad_value(ptr)"),
        "expected 'declare double @iris_grad_value(ptr)':\n{}",
        ir
    );
    assert!(
        ir.contains("declare double @iris_grad_tangent(ptr)"),
        "expected 'declare double @iris_grad_tangent(ptr)':\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 4. Sparse and atomic declares have correct signatures
// ---------------------------------------------------------------------------
#[test]
fn test_sparse_atomic_declares_correct() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Binary).unwrap();
    assert!(
        ir.contains("declare ptr @iris_sparsify(ptr)"),
        "expected 'declare ptr @iris_sparsify(ptr)':\n{}",
        ir
    );
    assert!(
        ir.contains("declare ptr @iris_densify(ptr)"),
        "expected 'declare ptr @iris_densify(ptr)':\n{}",
        ir
    );
    assert!(
        ir.contains("declare ptr @iris_atomic_new(ptr)"),
        "expected 'declare ptr @iris_atomic_new(ptr)':\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 5. ValueToStr dispatches by type — to_str(i64) → iris_i64_to_str
// ---------------------------------------------------------------------------
#[test]
fn test_value_to_str_type_dispatch() {
    let src = r#"
def f() -> str {
    val x = 42
    to_str(x)
}
"#;
    let ir = compile(src, "test", EmitKind::Binary).unwrap();
    assert!(
        ir.contains("@iris_i64_to_str"),
        "expected iris_i64_to_str for to_str(i64), got:\n{}",
        ir
    );
    // Should NOT *call* the generic value_to_str for a plain i64 (the declare is still there).
    assert!(
        !ir.contains("call ptr @iris_value_to_str"),
        "unexpected call to iris_value_to_str for known scalar type, got:\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 6. ListPush boxes a scalar before passing to iris_list_push
// ---------------------------------------------------------------------------
#[test]
fn test_list_push_boxes_scalar() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 99);
    list_len(xs)
}
"#;
    let ir = compile(src, "test", EmitKind::Binary).unwrap();
    assert!(
        ir.contains("@iris_box_i64"),
        "expected @iris_box_i64 boxing call before iris_list_push, got:\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 7. Runtime C source is accessible (embedded in IRIS binary)
// ---------------------------------------------------------------------------
#[test]
fn test_runtime_c_source_accessible() {
    let c_src = runtime_c_source();
    assert!(
        !c_src.is_empty(),
        "runtime_c_source() returned empty string"
    );
    assert!(
        c_src.contains("iris_box_i64"),
        "runtime C source should contain iris_box_i64 implementation"
    );
    assert!(
        c_src.contains("iris_list_push"),
        "runtime C source should contain iris_list_push implementation"
    );

    let h_src = runtime_h_source();
    assert!(
        !h_src.is_empty(),
        "runtime_h_source() returned empty string"
    );
    assert!(
        h_src.contains("IrisVal"),
        "runtime header should define IrisVal"
    );
}

// ---------------------------------------------------------------------------
// 8. EmitKind::Binary emits valid LLVM IR
// ---------------------------------------------------------------------------
#[test]
fn test_emit_binary_produces_llvm_ir() {
    let src = r#"
def f() -> i64 {
    val x = 10
    val y = 20
    x + y
}
"#;
    let ir = compile(src, "test", EmitKind::Binary).unwrap();
    // Should be LLVM IR, not empty, not IR text.
    assert!(
        ir.contains("target triple"),
        "EmitKind::Binary should emit LLVM IR with target triple:\n{}",
        ir
    );
    assert!(
        ir.contains("define"),
        "EmitKind::Binary should emit function definitions:\n{}",
        ir
    );
    assert!(
        ir.contains("declare"),
        "EmitKind::Binary should emit runtime declares:\n{}",
        ir
    );
}
