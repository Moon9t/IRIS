//! End-to-end tests: source string → compile() → assert output.

use iris::{compile, EmitKind};

const ADD_SRC: &str = r#"
def add(x: f32, y: f32) -> f32 {
    x + y
}
"#;

const ID_SRC: &str = r#"
def identity(x: i64) -> i64 {
    x
}
"#;

const MULTI_FN_SRC: &str = r#"
def square(x: f32) -> f32 {
    x * x
}

def negate(x: f32) -> f32 {
    x * x
}
"#;

const MATMUL_SRC: &str = r#"
def matmul(A: tensor<f32, [M, K]>, B: tensor<f32, [K, N]>) -> tensor<f32, [M, N]> {
    val C = einsum("mk,kn->mn", A, B)
    C
}
"#;

const CONST_SRC: &str = r#"
def always_one() -> f32 {
    1.0
}
"#;

// ---------------------------------------------------------------------------
// IR emit tests
// ---------------------------------------------------------------------------

#[test]
fn test_compile_add_emit_ir() {
    let output = compile(ADD_SRC, "add_module", EmitKind::Ir).expect("compile should succeed");
    assert!(output.contains("def add"), "IR should contain 'def add'");
    assert!(output.contains("return"), "IR should contain 'return'");
    // The add binop appears in the IR text
    assert!(output.contains("add"), "IR should contain an add instruction");
}

#[test]
fn test_compile_identity_emit_ir() {
    let output = compile(ID_SRC, "id_module", EmitKind::Ir).expect("compile should succeed");
    assert!(output.contains("def identity"));
    assert!(output.contains("return"));
}

#[test]
fn test_compile_constant_emit_ir() {
    let output = compile(CONST_SRC, "const_module", EmitKind::Ir).expect("compile should succeed");
    assert!(output.contains("def always_one"));
    assert!(output.contains("const.f"));
    assert!(output.contains("return"));
}

#[test]
fn test_compile_matmul_emit_ir() {
    let output = compile(MATMUL_SRC, "matmul_module", EmitKind::Ir)
        .expect("matmul compile should succeed");
    assert!(output.contains("def matmul"));
    assert!(output.contains("einsum"));
    assert!(output.contains("return"));
}

#[test]
fn test_compile_multi_function_emit_ir() {
    let output =
        compile(MULTI_FN_SRC, "multi_module", EmitKind::Ir).expect("multi fn compile should succeed");
    assert!(output.contains("def square"));
    assert!(output.contains("def negate"));
}

// ---------------------------------------------------------------------------
// LLVM stub emit tests
// ---------------------------------------------------------------------------

#[test]
fn test_compile_add_emit_llvm() {
    let output =
        compile(ADD_SRC, "add_module", EmitKind::Llvm).expect("llvm stub should succeed");
    assert!(output.contains("define"));
    assert!(output.contains("@add"));
    assert!(output.contains("float %x"));
    assert!(output.contains("float %y"));
    assert!(output.contains("ret float"));
}

#[test]
fn test_compile_identity_emit_llvm() {
    let output = compile(ID_SRC, "id_module", EmitKind::Llvm).expect("llvm stub should succeed");
    assert!(output.contains("define"));
    assert!(output.contains("@identity"));
    assert!(output.contains("i64 %x"));
}

#[test]
fn test_compile_matmul_emit_llvm() {
    let output = compile(MATMUL_SRC, "matmul_module", EmitKind::Llvm)
        .expect("llvm stub should succeed");
    assert!(output.contains("define"));
    assert!(output.contains("@matmul"));
    // tensor args lower to ptr
    assert!(output.contains("ptr"));
}

// ---------------------------------------------------------------------------
// Error recovery tests
// ---------------------------------------------------------------------------

#[test]
fn test_parse_error_on_unexpected_token() {
    let src = "def $bad() -> f32 {}";
    let result = compile(src, "bad", EmitKind::Ir);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("parse error") || msg.contains("unexpected"),
        "error message was: {}",
        msg
    );
}

#[test]
fn test_parse_error_on_missing_return_type() {
    let src = "def no_ret(x: f32) { x }";
    let result = compile(src, "bad", EmitKind::Ir);
    assert!(result.is_err());
}

#[test]
fn test_lower_error_on_undefined_variable() {
    let src = "def bad(x: f32) -> f32 { y }";
    let result = compile(src, "bad", EmitKind::Ir);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("lowering") || msg.contains("undefined") || msg.contains("cannot find"),
        "error message was: {}",
        msg
    );
}

#[test]
fn test_type_mismatch_binop() {
    // f32 + i64 should fail in the lowerer (type mismatch).
    let src = r#"
def bad(x: f32, y: i64) -> f32 {
    x + y
}
"#;
    let result = compile(src, "bad", EmitKind::Ir);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Module naming
// ---------------------------------------------------------------------------

#[test]
fn test_module_name_in_ir_output() {
    let output = compile(ADD_SRC, "my_custom_module", EmitKind::Ir).unwrap();
    assert!(output.contains("my_custom_module"));
}
