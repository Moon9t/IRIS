//! Integration tests for Phase 8 — stub completion:
//! - CmpNe / CmpGt / CmpGe produce correct IR ops (were wrong: Ne→Eq, Gt→Lt, Ge→Le)
//! - LLVM emitter uses real operands for Load/Store (was `undef`)
//! - TypeInferPass rejects `neg` on Bool and `not` on numeric types
//! - ShapeCheckPass validates Einsum notation (was skipped/empty)

use iris::ir::function::Param;
use iris::ir::instr::{IrInstr, ScalarUnaryOp, TensorOp};
use iris::ir::module::{IrFunctionBuilder, IrModule};
use iris::ir::types::{DType, Dim, IrType, Shape};
use iris::pass::type_infer::TypeInferPass;
use iris::pass::validate::ValidatePass;
use iris::pass::{ConstFoldPass, CsePass, DcePass, OpExpandPass, PassManager, ShapeCheckPass};
use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. CmpNe emits `cmpne` in IR (was wrongly mapped to `cmpeq`)
// ---------------------------------------------------------------------------

#[test]
fn test_cmpne_correct_ir() {
    let src = r#"
def neq(a: f32, b: f32) -> bool {
    a != b
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("cmpne"),
        "IR must contain 'cmpne' for !=\n{}",
        output
    );
    assert!(
        !output.contains("cmpeq"),
        "IR must NOT contain 'cmpeq' for !=\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 2. CmpGt emits `cmpgt` in IR (was wrongly mapped to `cmplt`)
// ---------------------------------------------------------------------------

#[test]
fn test_cmpgt_correct_ir() {
    let src = r#"
def gt(a: f32, b: f32) -> bool {
    a > b
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("cmpgt"),
        "IR must contain 'cmpgt' for >\n{}",
        output
    );
    assert!(
        !output.contains("cmplt"),
        "IR must NOT contain 'cmplt' for >\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 3. CmpGe emits `cmpge` in IR (was wrongly mapped to `cmple`)
// ---------------------------------------------------------------------------

#[test]
fn test_cmpge_correct_ir() {
    let src = r#"
def ge(a: f32, b: f32) -> bool {
    a >= b
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("cmpge"),
        "IR must contain 'cmpge' for >=\n{}",
        output
    );
    assert!(
        !output.contains("cmple"),
        "IR must NOT contain 'cmple' for >=\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 4. LLVM emitter emits `fcmp one` for f32 !=  (not `fcmp oeq`)
// ---------------------------------------------------------------------------

#[test]
fn test_llvm_cmpne_float() {
    let src = r#"
def neq(a: f32, b: f32) -> bool {
    a != b
}
"#;
    let output = compile(src, "test", EmitKind::Llvm).expect("compile");
    assert!(
        output.contains("fcmp one"),
        "LLVM must contain 'fcmp one' for f32 !=\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 5. LLVM emitter emits `fcmp ogt` for f32 >
// ---------------------------------------------------------------------------

#[test]
fn test_llvm_cmpgt_float() {
    let src = r#"
def gt(a: f32, b: f32) -> bool {
    a > b
}
"#;
    let output = compile(src, "test", EmitKind::Llvm).expect("compile");
    assert!(
        output.contains("fcmp ogt"),
        "LLVM must contain 'fcmp ogt' for f32 >\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 6. TypeInferPass rejects `neg` applied to a bool operand
// ---------------------------------------------------------------------------

#[test]
fn test_type_infer_neg_on_bool_rejected() {
    let params = vec![Param { name: "b".into(), ty: IrType::Scalar(DType::Bool) }];
    let mut builder = IrFunctionBuilder::new("bad", params, IrType::Scalar(DType::Bool));
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);
    let b_val = builder.add_block_param(entry, Some("b"), IrType::Scalar(DType::Bool));
    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::UnaryOp {
            result,
            op: ScalarUnaryOp::Neg,
            operand: b_val,
            ty: IrType::Scalar(DType::Bool),
        },
        Some(IrType::Scalar(DType::Bool)),
    );
    builder.push_instr(IrInstr::Return { values: vec![result] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    assert!(pm.run(&mut module).is_err(), "TypeInferPass must reject neg on bool");
}

// ---------------------------------------------------------------------------
// 7. TypeInferPass rejects `not` applied to a float operand
// ---------------------------------------------------------------------------

#[test]
fn test_type_infer_not_on_float_rejected() {
    let params = vec![Param { name: "x".into(), ty: IrType::Scalar(DType::F32) }];
    let mut builder = IrFunctionBuilder::new("bad", params, IrType::Scalar(DType::Bool));
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);
    let x_val = builder.add_block_param(entry, Some("x"), IrType::Scalar(DType::F32));
    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::UnaryOp {
            result,
            op: ScalarUnaryOp::Not,
            operand: x_val,
            ty: IrType::Scalar(DType::Bool),
        },
        Some(IrType::Scalar(DType::Bool)),
    );
    builder.push_instr(IrInstr::Return { values: vec![result] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    assert!(pm.run(&mut module).is_err(), "TypeInferPass must reject not on float");
}

// ---------------------------------------------------------------------------
// 8. ShapeCheckPass rejects einsum notation missing "->"
// ---------------------------------------------------------------------------

#[test]
fn test_einsum_shape_check_missing_arrow() {
    let tensor_ty = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(4), Dim::Literal(4)]),
    };
    let params = vec![Param { name: "x".into(), ty: tensor_ty.clone() }];
    let mut builder = IrFunctionBuilder::new("bad", params, tensor_ty.clone());
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);
    let x_val = builder.add_block_param(entry, Some("x"), tensor_ty.clone());
    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::TensorOp {
            result,
            op: TensorOp::Einsum { notation: "ij,jk".into() }, // missing "->"
            inputs: vec![x_val],
            result_ty: tensor_ty.clone(),
        },
        Some(tensor_ty),
    );
    builder.push_instr(IrInstr::Return { values: vec![result] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    pm.add_pass(ConstFoldPass);
    pm.add_pass(OpExpandPass);
    pm.add_pass(DcePass);
    pm.add_pass(CsePass);
    pm.add_pass(ShapeCheckPass);
    assert!(pm.run(&mut module).is_err(), "ShapeCheckPass must reject einsum without '->'");
}

// ---------------------------------------------------------------------------
// 9. ShapeCheckPass rejects einsum with wrong input count
// ---------------------------------------------------------------------------

#[test]
fn test_einsum_shape_check_wrong_input_count() {
    let tensor_ty = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(4), Dim::Literal(4)]),
    };
    let params = vec![Param { name: "x".into(), ty: tensor_ty.clone() }];
    let mut builder = IrFunctionBuilder::new("bad", params, tensor_ty.clone());
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);
    let x_val = builder.add_block_param(entry, Some("x"), tensor_ty.clone());
    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::TensorOp {
            result,
            op: TensorOp::Einsum { notation: "ij,jk->ik".into() }, // 2 specs, 1 input
            inputs: vec![x_val],
            result_ty: tensor_ty.clone(),
        },
        Some(tensor_ty),
    );
    builder.push_instr(IrInstr::Return { values: vec![result] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    pm.add_pass(ConstFoldPass);
    pm.add_pass(OpExpandPass);
    pm.add_pass(DcePass);
    pm.add_pass(CsePass);
    pm.add_pass(ShapeCheckPass);
    assert!(pm.run(&mut module).is_err(), "ShapeCheckPass must reject einsum with wrong input count");
}

// ---------------------------------------------------------------------------
// 10. LLVM Load with single index emits getelementptr + load (not undef)
// ---------------------------------------------------------------------------

#[test]
fn test_llvm_load_no_undef() {
    use iris::codegen::llvm_stub::emit_llvm_stub;

    let tensor_ty = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(8)]),
    };
    let params = vec![
        Param { name: "t".into(), ty: tensor_ty.clone() },
        Param { name: "i".into(), ty: IrType::Scalar(DType::I64) },
    ];
    let mut builder =
        IrFunctionBuilder::new("loadtest", params, IrType::Scalar(DType::F32));
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);
    let t_val = builder.add_block_param(entry, Some("t"), tensor_ty);
    let i_val = builder.add_block_param(entry, Some("i"), IrType::Scalar(DType::I64));
    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::Load {
            result,
            tensor: t_val,
            indices: vec![i_val],
            result_ty: IrType::Scalar(DType::F32),
        },
        Some(IrType::Scalar(DType::F32)),
    );
    builder.push_instr(IrInstr::Return { values: vec![result] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let output = emit_llvm_stub(&module).expect("emit");
    assert!(
        output.contains("getelementptr"),
        "LLVM load with index must emit getelementptr\n{}",
        output
    );
    assert!(
        !output.contains("undef"),
        "LLVM output must not contain 'undef'\n{}",
        output
    );
}
