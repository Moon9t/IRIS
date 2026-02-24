//! Phase 10 integration tests: tensor indexing, modulo, and casts.

use iris::{compile, EmitKind};

#[test]
fn test_tensor_load_ir() {
    let src = "def get(t: tensor<f32,[8]>, i: i64) -> f32 { t[i] }";
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(ir.contains("load"), "IR should contain 'load': {}", ir);
}

#[test]
fn test_tensor_store_ir() {
    // A function that stores into a tensor via index assignment.
    let src = r#"
        def set(t: tensor<f32,[8]>, v: f32) -> f32 {
            t[0] = v
            v
        }
    "#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "should compile: {:?}", result.err());
    let ir = result.unwrap();
    assert!(ir.contains("store"), "IR should contain 'store': {}", ir);
}

#[test]
fn test_modulo_ir() {
    let src = "def rem(a: i64, b: i64) -> i64 { a % b }";
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(ir.contains("mod"), "IR should contain 'mod': {}", ir);
}

#[test]
fn test_modulo_const_fold() {
    let src = "def rem() -> i64 { 10 % 3 }";
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    // After const folding, the result should be 1 (10 % 3 = 1).
    assert!(ir.contains("1"), "IR should contain folded result '1': {}", ir);
}

#[test]
fn test_cast_f32_to_i64() {
    let src = "def conv(x: f32) -> i64 { x to i64 }";
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(ir.contains("cast"), "IR should contain 'cast': {}", ir);
}

#[test]
fn test_cast_llvm_fptosi() {
    let src = "def conv(x: f32) -> i64 { x to i64 }";
    let llvm = compile(src, "test", EmitKind::Llvm).expect("should compile to LLVM");
    assert!(llvm.contains("fptosi"), "LLVM IR should contain 'fptosi': {}", llvm);
}

#[test]
fn test_cast_llvm_sitofp() {
    let src = "def conv(x: i64) -> f32 { x to f32 }";
    let llvm = compile(src, "test", EmitKind::Llvm).expect("should compile to LLVM");
    assert!(llvm.contains("sitofp"), "LLVM IR should contain 'sitofp': {}", llvm);
}

#[test]
fn test_modulo_llvm() {
    let src = "def rem(a: i64, b: i64) -> i64 { a % b }";
    let llvm = compile(src, "test", EmitKind::Llvm).expect("should compile to LLVM");
    assert!(llvm.contains("srem"), "LLVM IR should contain 'srem': {}", llvm);
}
