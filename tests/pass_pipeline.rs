//! Integration tests for the pass pipeline.
//! Builds modules via the IR builder API and runs passes directly.

use iris::ir::function::Param;
use iris::ir::instr::{BinOp, IrInstr, TensorOp};
use iris::ir::module::{IrFunctionBuilder, IrModule};
use iris::ir::types::{DType, Dim, IrType, Shape};
use iris::pass::validate::ValidatePass;
use iris::pass::type_infer::TypeInferPass;
use iris::pass::PassManager;

fn build_add_module() -> IrModule {
    let mut module = IrModule::new("add_module");
    let f32_ty = IrType::Scalar(DType::F32);
    let params = vec![
        Param {
            name: "x".into(),
            ty: f32_ty.clone(),
        },
        Param {
            name: "y".into(),
            ty: f32_ty.clone(),
        },
    ];
    let mut builder = IrFunctionBuilder::new("add", params, f32_ty.clone());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f32_ty.clone());
    let y = builder.add_block_param(entry, Some("y"), f32_ty.clone());
    builder.set_current_block(entry);
    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result,
            op: BinOp::Add,
            lhs: x,
            rhs: y,
            ty: f32_ty.clone(),
        },
        Some(f32_ty),
    );
    builder.push_instr(IrInstr::Return { values: vec![result] }, None);
    module.add_function(builder.build()).unwrap();
    module
}

fn build_matmul_module() -> IrModule {
    let mut module = IrModule::new("matmul_module");
    let ty_a = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Symbolic("M".into()), Dim::Symbolic("K".into())]),
    };
    let ty_b = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Symbolic("K".into()), Dim::Symbolic("N".into())]),
    };
    let ty_c = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Symbolic("M".into()), Dim::Symbolic("N".into())]),
    };
    let params = vec![
        Param {
            name: "A".into(),
            ty: ty_a.clone(),
        },
        Param {
            name: "B".into(),
            ty: ty_b.clone(),
        },
    ];
    let mut builder = IrFunctionBuilder::new("matmul", params, ty_c.clone());
    let entry = builder.create_block(Some("entry"));
    let a = builder.add_block_param(entry, Some("A"), ty_a);
    let b = builder.add_block_param(entry, Some("B"), ty_b);
    builder.set_current_block(entry);
    let c = builder.fresh_value();
    builder.push_instr(
        IrInstr::TensorOp {
            result: c,
            op: TensorOp::Einsum {
                notation: "mk,kn->mn".into(),
            },
            inputs: vec![a, b],
            result_ty: ty_c.clone(),
        },
        Some(ty_c),
    );
    builder.push_instr(IrInstr::Return { values: vec![c] }, None);
    module.add_function(builder.build()).unwrap();
    module
}

#[test]
fn test_validate_pass_on_scalar_add() {
    let mut module = build_add_module();
    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    assert!(pm.run(&mut module).is_ok());
}

#[test]
fn test_full_pipeline_on_scalar_add() {
    let mut module = build_add_module();
    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    assert!(pm.run(&mut module).is_ok());
}

#[test]
fn test_full_pipeline_on_matmul() {
    let mut module = build_matmul_module();
    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    assert!(pm.run(&mut module).is_ok());
}

#[test]
fn test_ir_printer_contains_expected_tokens() {
    let mut module = build_add_module();
    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.run(&mut module).unwrap();

    let output = iris::codegen::emit_ir_text(&module).unwrap();
    assert!(output.contains("def add"), "output should contain 'def add'");
    assert!(output.contains("return"), "output should contain 'return'");
    assert!(output.contains("add"), "output should contain the add binop");
}

#[test]
fn test_llvm_stub_contains_expected_tokens() {
    let mut module = build_add_module();
    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.run(&mut module).unwrap();

    let output = iris::codegen::emit_llvm_stub(&module).unwrap();
    assert!(output.contains("define"), "LLVM stub should contain 'define'");
    assert!(output.contains("@add"), "LLVM stub should contain '@add'");
    assert!(output.contains("float"), "LLVM stub should contain 'float' type");
}

#[test]
fn test_pass_manager_names() {
    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    let names = pm.pass_names();
    assert_eq!(names, vec!["validate", "type-infer"]);
}

#[test]
fn test_validate_rejects_infer_type() {
    // Build a module where a value has IrType::Infer — should fail ValidatePass.
    let mut module = IrModule::new("infer_test");
    let f32_ty = IrType::Scalar(DType::F32);
    let params = vec![Param {
        name: "x".into(),
        ty: f32_ty.clone(),
    }];
    let mut builder = IrFunctionBuilder::new("bad", params, f32_ty.clone());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f32_ty.clone());
    builder.set_current_block(entry);
    // Push a constant with Infer type to simulate unresolved inference.
    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result,
            value: 1.0,
            ty: IrType::Infer,
        },
        Some(IrType::Infer), // deliberately broken
    );
    builder.push_instr(IrInstr::Return { values: vec![x] }, None);
    module.add_function(builder.build()).unwrap();

    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    let err = pm.run(&mut module);
    assert!(
        err.is_err(),
        "ValidatePass must reject modules with Infer types"
    );
    let (pass_name, _) = err.unwrap_err();
    assert_eq!(pass_name, "validate");
}
