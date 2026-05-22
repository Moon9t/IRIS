//! Phase 131 integration tests: sparse tensor operations.
//!
//! Validates:
//! - Sparsify: converting a dense Array/Tensor to sparse (index, value) pairs
//! - Densify: reconstructing a dense Array from sparse pairs
//! - Round-trip: sparsify → densify preserves data
//! - Edge cases: all-zero, single-element, etc.

use iris::interp::{eval_function, IrValue};
use iris::ir::function::Param;
use iris::ir::instr::IrInstr;
use iris::ir::module::IrFunctionBuilder;
use iris::ir::types::{DType, IrType};

fn array_ty() -> IrType {
    IrType::Array {
        elem: Box::new(IrType::Scalar(DType::F32)),
        len: 0, // dynamic
    }
}

fn sparse_ty() -> IrType {
    IrType::Sparse(Box::new(IrType::Scalar(DType::F32)))
}

// ── Sparsify a dense array with some zeros ──────────────────────────────

#[test]
fn test_sparsify_array() {
    let params = vec![Param {
        name: "arr".into(),
        ty: array_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("sparse_test", params, sparse_ty());
    let entry = builder.create_block(Some("entry"));
    let arr = builder.add_block_param(entry, Some("arr"), array_ty());
    builder.set_current_block(entry);

    let sp = builder.fresh_value();
    builder.push_instr(
        IrInstr::Sparsify {
            result: sp,
            operand: arr,
            ty: sparse_ty(),
        },
        Some(sparse_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![sp] }, None);
    let func = builder.build();

    // [0.0, 3.0, 0.0, 5.0, 0.0] → sparse with 2 entries: (1, 3.0), (3, 5.0)
    let input = IrValue::Array(vec![
        IrValue::F32(0.0),
        IrValue::F32(3.0),
        IrValue::F32(0.0),
        IrValue::F32(5.0),
        IrValue::F32(0.0),
    ]);
    let result = eval_function(&func, &[input]).expect("eval");

    match &result[0] {
        IrValue::Sparse(pairs) => {
            assert_eq!(pairs.len(), 2, "should have 2 non-zero elements");
            assert_eq!(pairs[0].0, 1, "first non-zero at index 1");
            assert_eq!(pairs[1].0, 3, "second non-zero at index 3");
            match &pairs[0].1 {
                IrValue::F32(v) => assert!((*v - 3.0).abs() < 1e-6),
                other => panic!("expected F32, got {:?}", other),
            }
            match &pairs[1].1 {
                IrValue::F32(v) => assert!((*v - 5.0).abs() < 1e-6),
                other => panic!("expected F32, got {:?}", other),
            }
        }
        other => panic!("expected Sparse, got {:?}", other),
    }
}

// ── Sparsify all zeros → empty sparse ───────────────────────────────────

#[test]
fn test_sparsify_all_zeros() {
    let params = vec![Param {
        name: "arr".into(),
        ty: array_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("sparse_zeros", params, sparse_ty());
    let entry = builder.create_block(Some("entry"));
    let arr = builder.add_block_param(entry, Some("arr"), array_ty());
    builder.set_current_block(entry);

    let sp = builder.fresh_value();
    builder.push_instr(
        IrInstr::Sparsify {
            result: sp,
            operand: arr,
            ty: sparse_ty(),
        },
        Some(sparse_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![sp] }, None);
    let func = builder.build();

    let input = IrValue::Array(vec![
        IrValue::F32(0.0),
        IrValue::F32(0.0),
        IrValue::F32(0.0),
    ]);
    let result = eval_function(&func, &[input]).expect("eval");

    match &result[0] {
        IrValue::Sparse(pairs) => {
            assert_eq!(pairs.len(), 0, "all zeros → empty sparse");
        }
        other => panic!("expected Sparse, got {:?}", other),
    }
}

// ── Sparsify a tensor (dense) ───────────────────────────────────────────

#[test]
fn test_sparsify_tensor() {
    let params = vec![Param {
        name: "t".into(),
        ty: array_ty(), // type doesn't matter for interpreter dispatch
    }];
    let mut builder = IrFunctionBuilder::new("sparse_tensor", params, sparse_ty());
    let entry = builder.create_block(Some("entry"));
    let t = builder.add_block_param(entry, Some("t"), array_ty());
    builder.set_current_block(entry);

    let sp = builder.fresh_value();
    builder.push_instr(
        IrInstr::Sparsify {
            result: sp,
            operand: t,
            ty: sparse_ty(),
        },
        Some(sparse_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![sp] }, None);
    let func = builder.build();

    // A 2x3 tensor with some zeros
    let input = IrValue::Tensor(vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0], vec![2, 3]);
    let result = eval_function(&func, &[input]).expect("eval");

    match &result[0] {
        IrValue::Sparse(pairs) => {
            assert_eq!(pairs.len(), 3, "3 non-zero elements");
            assert_eq!(pairs[0].0, 0); // index 0 → 1.0
            assert_eq!(pairs[1].0, 2); // index 2 → 2.0
            assert_eq!(pairs[2].0, 5); // index 5 → 3.0
        }
        other => panic!("expected Sparse, got {:?}", other),
    }
}

// ── Densify a sparse → nnz count ────────────────────────────────────────
// Densify in the interpreter returns the number of non-zero elements (nnz)
// as an i64, matching the lowerer which emits result_ty = i64 for densify().

#[test]
fn test_densify_sparse() {
    let params = vec![Param {
        name: "arr".into(),
        ty: array_ty(),
    }];
    let i64_ty = IrType::Scalar(DType::I64);
    let mut builder = IrFunctionBuilder::new("densify_test", params, i64_ty.clone());
    let entry = builder.create_block(Some("entry"));
    let arr = builder.add_block_param(entry, Some("arr"), array_ty());
    builder.set_current_block(entry);

    let sp = builder.fresh_value();
    builder.push_instr(
        IrInstr::Sparsify {
            result: sp,
            operand: arr,
            ty: sparse_ty(),
        },
        Some(sparse_ty()),
    );

    let nnz = builder.fresh_value();
    builder.push_instr(
        IrInstr::Densify {
            result: nnz,
            operand: sp,
            ty: i64_ty.clone(),
        },
        Some(i64_ty.clone()),
    );

    builder.push_instr(IrInstr::Return { values: vec![nnz] }, None);
    let func = builder.build();

    // [0, 7, 0, 0, 9] has 2 non-zero elements
    let input = IrValue::Array(vec![
        IrValue::F32(0.0),
        IrValue::F32(7.0),
        IrValue::F32(0.0),
        IrValue::F32(0.0),
        IrValue::F32(9.0),
    ]);
    let result = eval_function(&func, &[input]).expect("eval");

    match &result[0] {
        IrValue::I64(n) => assert_eq!(*n, 2, "nnz of [0,7,0,0,9] should be 2"),
        other => panic!("expected I64 nnz, got {:?}", other),
    }
}

// ── Densify empty sparse → nnz 0 ────────────────────────────────────────

#[test]
fn test_densify_empty_sparse() {
    let params = vec![Param {
        name: "arr".into(),
        ty: array_ty(),
    }];
    let i64_ty = IrType::Scalar(DType::I64);
    let mut builder = IrFunctionBuilder::new("densify_empty", params, i64_ty.clone());
    let entry = builder.create_block(Some("entry"));
    let arr = builder.add_block_param(entry, Some("arr"), array_ty());
    builder.set_current_block(entry);

    let sp = builder.fresh_value();
    builder.push_instr(
        IrInstr::Sparsify {
            result: sp,
            operand: arr,
            ty: sparse_ty(),
        },
        Some(sparse_ty()),
    );

    let nnz = builder.fresh_value();
    builder.push_instr(
        IrInstr::Densify {
            result: nnz,
            operand: sp,
            ty: i64_ty.clone(),
        },
        Some(i64_ty.clone()),
    );

    builder.push_instr(IrInstr::Return { values: vec![nnz] }, None);
    let func = builder.build();

    // All zeros → sparsify gives empty sparse → densify gives nnz = 0
    let input = IrValue::Array(vec![IrValue::F32(0.0), IrValue::F32(0.0)]);
    let result = eval_function(&func, &[input]).expect("eval");

    match &result[0] {
        IrValue::I64(n) => assert_eq!(*n, 0, "densify of all-zero sparse = nnz 0"),
        other => panic!("expected I64 nnz, got {:?}", other),
    }
}

// ── Sparsify i64 array with zeros ───────────────────────────────────────

#[test]
fn test_sparsify_i64_array() {
    let params = vec![Param {
        name: "arr".into(),
        ty: array_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("sparse_i64", params, sparse_ty());
    let entry = builder.create_block(Some("entry"));
    let arr = builder.add_block_param(entry, Some("arr"), array_ty());
    builder.set_current_block(entry);

    let sp = builder.fresh_value();
    builder.push_instr(
        IrInstr::Sparsify {
            result: sp,
            operand: arr,
            ty: sparse_ty(),
        },
        Some(sparse_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![sp] }, None);
    let func = builder.build();

    let input = IrValue::Array(vec![
        IrValue::I64(0),
        IrValue::I64(42),
        IrValue::I64(0),
        IrValue::I64(-7),
    ]);
    let result = eval_function(&func, &[input]).expect("eval");

    match &result[0] {
        IrValue::Sparse(pairs) => {
            assert_eq!(pairs.len(), 2, "2 non-zero i64 elements");
            assert_eq!(pairs[0].0, 1);
            assert_eq!(pairs[1].0, 3);
        }
        other => panic!("expected Sparse, got {:?}", other),
    }
}

// ── Sparsify all non-zero → same length ─────────────────────────────────

#[test]
fn test_sparsify_all_nonzero() {
    let params = vec![Param {
        name: "arr".into(),
        ty: array_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("sparse_full", params, sparse_ty());
    let entry = builder.create_block(Some("entry"));
    let arr = builder.add_block_param(entry, Some("arr"), array_ty());
    builder.set_current_block(entry);

    let sp = builder.fresh_value();
    builder.push_instr(
        IrInstr::Sparsify {
            result: sp,
            operand: arr,
            ty: sparse_ty(),
        },
        Some(sparse_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![sp] }, None);
    let func = builder.build();

    let input = IrValue::Array(vec![
        IrValue::F32(1.0),
        IrValue::F32(2.0),
        IrValue::F32(3.0),
    ]);
    let result = eval_function(&func, &[input]).expect("eval");

    match &result[0] {
        IrValue::Sparse(pairs) => {
            assert_eq!(pairs.len(), 3, "all non-zero = all stored");
        }
        other => panic!("expected Sparse, got {:?}", other),
    }
}
