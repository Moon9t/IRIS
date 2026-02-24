//! Phase 11 integration tests: tree-walking IR interpreter.

use iris::interp::{eval_function, IrValue};
use iris::ir::function::Param;
use iris::ir::instr::{BinOp, IrInstr, ScalarUnaryOp};
use iris::ir::module::IrFunctionBuilder;
use iris::ir::types::{DType, Dim, IrType, Shape};
use iris::{compile, EmitKind};

fn f32_ty() -> IrType {
    IrType::Scalar(DType::F32)
}
fn i64_ty() -> IrType {
    IrType::Scalar(DType::I64)
}
fn bool_ty() -> IrType {
    IrType::Scalar(DType::Bool)
}

// ---------------------------------------------------------------------------
// 1. Constant return
// ---------------------------------------------------------------------------
#[test]
fn test_eval_const_return() {
    let ty = i64_ty();
    let mut b = IrFunctionBuilder::new("answer", vec![], ty.clone());
    let entry = b.create_block(Some("entry"));
    b.set_current_block(entry);
    let c = b.fresh_value();
    b.push_instr(IrInstr::ConstInt { result: c, value: 42, ty: ty.clone() }, Some(ty));
    b.push_instr(IrInstr::Return { values: vec![c] }, None);
    let func = b.build();

    let result = eval_function(&func, &[]).expect("should evaluate");
    assert_eq!(result, vec![IrValue::I64(42)]);
}

// ---------------------------------------------------------------------------
// 2. Arithmetic
// ---------------------------------------------------------------------------
#[test]
fn test_eval_arithmetic() {
    let ty = f32_ty();
    let params = vec![
        Param { name: "a".into(), ty: ty.clone() },
        Param { name: "b".into(), ty: ty.clone() },
    ];
    let mut b = IrFunctionBuilder::new("add", params, ty.clone());
    let entry = b.create_block(Some("entry"));
    let a = b.add_block_param(entry, Some("a"), ty.clone());
    let bv = b.add_block_param(entry, Some("b"), ty.clone());
    b.set_current_block(entry);
    let sum = b.fresh_value();
    b.push_instr(
        IrInstr::BinOp { result: sum, op: BinOp::Add, lhs: a, rhs: bv, ty: ty.clone() },
        Some(ty),
    );
    b.push_instr(IrInstr::Return { values: vec![sum] }, None);
    let func = b.build();

    let result = eval_function(&func, &[IrValue::F32(1.0), IrValue::F32(2.0)]).expect("should eval");
    assert_eq!(result, vec![IrValue::F32(3.0)]);
}

// ---------------------------------------------------------------------------
// 3. If-else: branch taken (a > b)
// ---------------------------------------------------------------------------
#[test]
fn test_eval_if_true() {
    let ty = f32_ty();
    let bty = bool_ty();
    let params = vec![
        Param { name: "a".into(), ty: ty.clone() },
        Param { name: "b".into(), ty: ty.clone() },
    ];
    let mut builder = IrFunctionBuilder::new("max", params, ty.clone());
    let entry = builder.create_block(Some("entry"));
    let then_bb = builder.create_block(Some("then"));
    let else_bb = builder.create_block(Some("else_"));

    let a = builder.add_block_param(entry, Some("a"), ty.clone());
    let bv = builder.add_block_param(entry, Some("b"), ty.clone());
    let then_val = builder.add_block_param(then_bb, Some("x"), ty.clone());
    let else_val = builder.add_block_param(else_bb, Some("y"), ty.clone());

    builder.set_current_block(entry);
    let cond = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp { result: cond, op: BinOp::CmpGt, lhs: a, rhs: bv, ty: bty.clone() },
        Some(bty),
    );
    builder.push_instr(
        IrInstr::CondBr {
            cond,
            then_block: then_bb,
            then_args: vec![a],
            else_block: else_bb,
            else_args: vec![bv],
        },
        None,
    );

    builder.set_current_block(then_bb);
    builder.push_instr(IrInstr::Return { values: vec![then_val] }, None);

    builder.set_current_block(else_bb);
    builder.push_instr(IrInstr::Return { values: vec![else_val] }, None);

    let func = builder.build();

    // a=5.0 > b=3.0 → takes then branch → returns 5.0
    let result =
        eval_function(&func, &[IrValue::F32(5.0), IrValue::F32(3.0)]).expect("should eval");
    assert_eq!(result, vec![IrValue::F32(5.0)]);
}

// ---------------------------------------------------------------------------
// 4. If-else: else branch taken (b > a)
// ---------------------------------------------------------------------------
#[test]
fn test_eval_if_false() {
    // Same max function; a < b → else branch → returns b
    let ty = f32_ty();
    let bty = bool_ty();
    let params = vec![
        Param { name: "a".into(), ty: ty.clone() },
        Param { name: "b".into(), ty: ty.clone() },
    ];
    let mut builder = IrFunctionBuilder::new("max2", params, ty.clone());
    let entry = builder.create_block(Some("entry"));
    let then_bb = builder.create_block(Some("then"));
    let else_bb = builder.create_block(Some("else_"));

    let a = builder.add_block_param(entry, Some("a"), ty.clone());
    let bv = builder.add_block_param(entry, Some("b"), ty.clone());
    let then_val = builder.add_block_param(then_bb, Some("x"), ty.clone());
    let else_val = builder.add_block_param(else_bb, Some("y"), ty.clone());

    builder.set_current_block(entry);
    let cond = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp { result: cond, op: BinOp::CmpGt, lhs: a, rhs: bv, ty: bty.clone() },
        Some(bty),
    );
    builder.push_instr(
        IrInstr::CondBr {
            cond,
            then_block: then_bb,
            then_args: vec![a],
            else_block: else_bb,
            else_args: vec![bv],
        },
        None,
    );

    builder.set_current_block(then_bb);
    builder.push_instr(IrInstr::Return { values: vec![then_val] }, None);

    builder.set_current_block(else_bb);
    builder.push_instr(IrInstr::Return { values: vec![else_val] }, None);

    let func = builder.build();

    // a=2.0 < b=7.0 → takes else branch → returns 7.0
    let result =
        eval_function(&func, &[IrValue::F32(2.0), IrValue::F32(7.0)]).expect("should eval");
    assert_eq!(result, vec![IrValue::F32(7.0)]);
}

// ---------------------------------------------------------------------------
// 5. Unary negation
// ---------------------------------------------------------------------------
#[test]
fn test_eval_neg() {
    let ty = f32_ty();
    let params = vec![Param { name: "x".into(), ty: ty.clone() }];
    let mut b = IrFunctionBuilder::new("neg", params, ty.clone());
    let entry = b.create_block(Some("entry"));
    let x = b.add_block_param(entry, Some("x"), ty.clone());
    b.set_current_block(entry);
    let r = b.fresh_value();
    b.push_instr(
        IrInstr::UnaryOp { result: r, op: ScalarUnaryOp::Neg, operand: x, ty: ty.clone() },
        Some(ty),
    );
    b.push_instr(IrInstr::Return { values: vec![r] }, None);
    let func = b.build();

    let result = eval_function(&func, &[IrValue::F32(3.0)]).expect("should eval");
    assert_eq!(result, vec![IrValue::F32(-3.0)]);
}

// ---------------------------------------------------------------------------
// 6. Boolean NOT
// ---------------------------------------------------------------------------
#[test]
fn test_eval_not() {
    let bty = bool_ty();
    let params = vec![Param { name: "b".into(), ty: bty.clone() }];
    let mut b = IrFunctionBuilder::new("inv", params, bty.clone());
    let entry = b.create_block(Some("entry"));
    let bv = b.add_block_param(entry, Some("b"), bty.clone());
    b.set_current_block(entry);
    let r = b.fresh_value();
    b.push_instr(
        IrInstr::UnaryOp { result: r, op: ScalarUnaryOp::Not, operand: bv, ty: bty.clone() },
        Some(bty),
    );
    b.push_instr(IrInstr::Return { values: vec![r] }, None);
    let func = b.build();

    let result = eval_function(&func, &[IrValue::Bool(true)]).expect("should eval");
    assert_eq!(result, vec![IrValue::Bool(false)]);
}

// ---------------------------------------------------------------------------
// 7. Tensor load
// ---------------------------------------------------------------------------
#[test]
fn test_eval_tensor_load() {
    let tensor_ty = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(4)]),
    };
    let idx_ty = i64_ty();
    let elem_ty = f32_ty();

    let params = vec![
        Param { name: "t".into(), ty: tensor_ty.clone() },
        Param { name: "i".into(), ty: idx_ty.clone() },
    ];
    let mut b = IrFunctionBuilder::new("get", params, elem_ty.clone());
    let entry = b.create_block(Some("entry"));
    let t = b.add_block_param(entry, Some("t"), tensor_ty);
    let idx = b.add_block_param(entry, Some("i"), idx_ty);
    b.set_current_block(entry);
    let r = b.fresh_value();
    b.push_instr(
        IrInstr::Load { result: r, tensor: t, indices: vec![idx], result_ty: elem_ty.clone() },
        Some(elem_ty),
    );
    b.push_instr(IrInstr::Return { values: vec![r] }, None);
    let func = b.build();

    // Tensor [10.0, 20.0, 30.0, 40.0], index 2 → 30.0
    let tensor = IrValue::Tensor(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let result = eval_function(&func, &[tensor, IrValue::I64(2)]).expect("should eval");
    assert_eq!(result, vec![IrValue::F32(30.0)]);
}

// ---------------------------------------------------------------------------
// 8. EmitKind::Eval via compile()
// ---------------------------------------------------------------------------
#[test]
fn test_eval_emit_kind() {
    let src = "def answer() -> i64 { 42 }";
    let out = compile(src, "test", EmitKind::Eval).expect("should compile and eval");
    assert_eq!(out, "42\n");
}
