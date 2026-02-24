//! Tests that construct IR using the builder API directly, without parsing.
//! These verify IR invariants at the type and structure level.

use iris::ir::function::Param;
use iris::ir::instr::{BinOp, IrInstr, TensorOp};
use iris::ir::module::{IrFunctionBuilder, IrModule};
use iris::ir::types::{DType, Dim, IrType, Shape};

fn tensor_ty(dtype: DType, dims: &[&str]) -> IrType {
    IrType::Tensor {
        dtype,
        shape: Shape(
            dims.iter()
                .map(|s| Dim::Symbolic(s.to_string()))
                .collect(),
        ),
    }
}

#[test]
fn test_build_scalar_add() {
    let mut module = IrModule::new("test_scalar");

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

    let func = builder.build();
    assert_eq!(func.blocks().len(), 1);
    assert_eq!(func.entry_block().params.len(), 2);
    assert_eq!(func.entry_block().instrs.len(), 2); // BinOp + Return

    module.add_function(func).expect("should add function");
    assert!(module.function_by_name("add").is_some());
}

#[test]
fn test_build_matmul_einsum() {
    let mut module = IrModule::new("test_matmul");

    let ty_a = tensor_ty(DType::F32, &["M", "K"]);
    let ty_b = tensor_ty(DType::F32, &["K", "N"]);
    let ty_c = tensor_ty(DType::F32, &["M", "N"]);

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

    let func = builder.build();
    assert!(func.entry_block().is_sealed());
    assert_eq!(func.entry_block().instrs.len(), 2);

    module.add_function(func).unwrap();
    let f = module.function_by_name("matmul").unwrap();
    assert_eq!(f.blocks().len(), 1);
}

#[test]
fn test_value_types_recorded() {
    let i64_ty = IrType::Scalar(DType::I64);
    let params = vec![Param {
        name: "a".into(),
        ty: i64_ty.clone(),
    }];
    let mut builder = IrFunctionBuilder::new("id", params, i64_ty.clone());
    let entry = builder.create_block(Some("entry"));
    let a = builder.add_block_param(entry, Some("a"), i64_ty.clone());
    builder.set_current_block(entry);
    builder.push_instr(IrInstr::Return { values: vec![a] }, None);
    let func = builder.build();

    assert_eq!(func.value_type(a), Some(&i64_ty));
}

#[test]
fn test_duplicate_function_name_rejected() {
    let mut module = IrModule::new("dup_test");

    let add_func = |name: &str| {
        let f32_ty = IrType::Scalar(DType::F32);
        let params = vec![Param {
            name: "x".into(),
            ty: f32_ty.clone(),
        }];
        let mut b = IrFunctionBuilder::new(name, params, f32_ty.clone());
        let entry = b.create_block(Some("entry"));
        let x = b.add_block_param(entry, Some("x"), f32_ty);
        b.set_current_block(entry);
        b.push_instr(IrInstr::Return { values: vec![x] }, None);
        b.build()
    };

    module.add_function(add_func("foo")).unwrap();
    let result = module.add_function(add_func("foo"));
    assert!(result.is_err(), "duplicate function name must be rejected");
}

#[test]
fn test_multi_block_function() {
    let mut module = IrModule::new("multi_block");
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
    let mut builder = IrFunctionBuilder::new("two_blocks", params, f32_ty.clone());

    let entry = builder.create_block(Some("entry"));
    let merge = builder.create_block(Some("merge"));

    let x = builder.add_block_param(entry, Some("x"), f32_ty.clone());
    let y = builder.add_block_param(entry, Some("y"), f32_ty.clone());
    // merge block takes one param (the result)
    let merged_val = builder.add_block_param(merge, Some("v"), f32_ty.clone());

    builder.set_current_block(entry);
    let sum = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: sum,
            op: BinOp::Add,
            lhs: x,
            rhs: y,
            ty: f32_ty.clone(),
        },
        Some(f32_ty.clone()),
    );
    builder.push_instr(
        IrInstr::Br {
            target: merge,
            args: vec![sum],
        },
        None,
    );

    builder.set_current_block(merge);
    builder.push_instr(IrInstr::Return { values: vec![merged_val] }, None);

    let func = builder.build();
    assert_eq!(func.blocks().len(), 2);
    assert!(func.blocks()[0].is_sealed());
    assert!(func.blocks()[1].is_sealed());

    module.add_function(func).unwrap();
}
