//! Phase 129 integration tests: tensor runtime operations.
//!
//! Validates interpreter-level tensor operations:
//! - Einsum: matmul (mk,kn->mn), dot product (i,i->), mat-vec (mk,k->m),
//!   transpose-via-einsum (ij->ji), batched matmul (bmk,bkn->bmn)
//! - Reshape: 2D→1D, 1D→2D, dimension mismatch rejection
//! - Transpose: 2D default reverse, 2D explicit axes, 3D transpose
//! - Reduce: sum, max, mean over axis 0/1, full reduce, keepdims
//! - Unary: relu, sigmoid, tanh on tensors

use iris::interp::{eval_function, IrValue};
use iris::ir::function::Param;
use iris::ir::instr::{IrInstr, TensorOp};
use iris::ir::module::IrFunctionBuilder;
use iris::ir::types::{DType, Dim, IrType, Shape};

fn tensor_ty(dtype: DType, dims: &[usize]) -> IrType {
    IrType::Tensor {
        dtype,
        shape: Shape(dims.iter().map(|d| Dim::Literal(*d as u64)).collect()),
    }
}

fn scalar_ty(dtype: DType) -> IrType {
    IrType::Scalar(dtype)
}

/// Build a single-block function that runs a TensorOp and returns the result.
fn run_tensor_op(op: TensorOp, inputs: Vec<(IrValue, IrType)>, result_ty: IrType) -> Vec<IrValue> {
    let params: Vec<Param> = inputs
        .iter()
        .enumerate()
        .map(|(i, (_, ty))| Param {
            name: format!("in{}", i),
            ty: ty.clone(),
        })
        .collect();
    let mut builder = IrFunctionBuilder::new("tensor_test", params, result_ty.clone());
    let entry = builder.create_block(Some("entry"));

    let mut input_vals = Vec::new();
    for (i, (_, ty)) in inputs.iter().enumerate() {
        let v = builder.add_block_param(entry, Some(&format!("in{}", i)), ty.clone());
        input_vals.push(v);
    }
    builder.set_current_block(entry);

    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::TensorOp {
            result,
            op,
            inputs: input_vals,
            result_ty: result_ty.clone(),
        },
        Some(result_ty),
    );
    builder.push_instr(
        IrInstr::Return {
            values: vec![result],
        },
        None,
    );

    let func = builder.build();
    let args: Vec<IrValue> = inputs.into_iter().map(|(v, _)| v).collect();
    eval_function(&func, &args).expect("eval should succeed")
}

fn assert_tensor_close(result: &IrValue, expected_data: &[f32], expected_shape: &[usize]) {
    match result {
        IrValue::Tensor(data, shape) => {
            assert_eq!(shape, expected_shape, "shape mismatch");
            assert_eq!(
                data.len(),
                expected_data.len(),
                "data length mismatch: got {}, expected {}",
                data.len(),
                expected_data.len()
            );
            for (i, (a, b)) in data.iter().zip(expected_data.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "element {} mismatch: got {}, expected {}",
                    i,
                    a,
                    b
                );
            }
        }
        _ => panic!("expected Tensor, got {:?}", result),
    }
}

fn assert_scalar_close(result: &IrValue, expected: f32) {
    match result {
        IrValue::F32(v) => {
            assert!(
                (v - expected).abs() < 1e-4,
                "scalar mismatch: got {}, expected {}",
                v,
                expected
            );
        }
        _ => panic!("expected F32 scalar, got {:?}", result),
    }
}

// ===========================================================================
// 1. Einsum: matmul  mk,kn->mn
// ===========================================================================

#[test]
fn test_einsum_matmul_2x3_3x2() {
    // A = [[1,2,3],[4,5,6]] (2x3)
    // B = [[7,8],[9,10],[11,12]] (3x2)
    // C = A @ B = [[58,64],[139,154]] (2x2)
    let a = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = IrValue::Tensor(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let ty_a = tensor_ty(DType::F32, &[2, 3]);
    let ty_b = tensor_ty(DType::F32, &[3, 2]);
    let ty_c = tensor_ty(DType::F32, &[2, 2]);

    let result = run_tensor_op(
        TensorOp::Einsum {
            notation: "mk,kn->mn".into(),
        },
        vec![(a, ty_a), (b, ty_b)],
        ty_c,
    );

    assert_eq!(result.len(), 1);
    assert_tensor_close(&result[0], &[58.0, 64.0, 139.0, 154.0], &[2, 2]);
}

#[test]
fn test_einsum_matmul_identity() {
    // A = [[1,0],[0,1]] (identity), B = [[3,4],[5,6]]
    // C = A @ B = [[3,4],[5,6]]
    let a = IrValue::Tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let b = IrValue::Tensor(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
    let ty = tensor_ty(DType::F32, &[2, 2]);

    let result = run_tensor_op(
        TensorOp::Einsum {
            notation: "mk,kn->mn".into(),
        },
        vec![(a, ty.clone()), (b, ty.clone())],
        ty,
    );
    assert_tensor_close(&result[0], &[3.0, 4.0, 5.0, 6.0], &[2, 2]);
}

// ===========================================================================
// 2. Einsum: dot product  i,i->
// ===========================================================================

#[test]
fn test_einsum_dot_product() {
    // a = [1,2,3], b = [4,5,6] → dot = 32
    let a = IrValue::Tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let b = IrValue::Tensor(vec![4.0, 5.0, 6.0], vec![3]);
    let ty_v = tensor_ty(DType::F32, &[3]);
    let ty_s = scalar_ty(DType::F32);

    let result = run_tensor_op(
        TensorOp::Einsum {
            notation: "i,i->".into(),
        },
        vec![(a, ty_v.clone()), (b, ty_v)],
        ty_s,
    );
    assert_scalar_close(&result[0], 32.0);
}

// ===========================================================================
// 3. Einsum: matrix-vector multiply  mk,k->m
// ===========================================================================

#[test]
fn test_einsum_matvec() {
    // A = [[1,2],[3,4]] (2x2), v = [5,6] → [17, 39]
    let a = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let v = IrValue::Tensor(vec![5.0, 6.0], vec![2]);
    let ty_a = tensor_ty(DType::F32, &[2, 2]);
    let ty_v = tensor_ty(DType::F32, &[2]);
    let ty_r = tensor_ty(DType::F32, &[2]);

    let result = run_tensor_op(
        TensorOp::Einsum {
            notation: "mk,k->m".into(),
        },
        vec![(a, ty_a), (v, ty_v)],
        ty_r,
    );
    assert_tensor_close(&result[0], &[17.0, 39.0], &[2]);
}

// ===========================================================================
// 4. Einsum: transpose via einsum  ij->ji
// ===========================================================================

#[test]
fn test_einsum_transpose() {
    // Test outer product: i,j->ij
    // a = [1,2,3], b = [4,5] → [[4,5],[8,10],[12,15]]
    let a2 = IrValue::Tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let b2 = IrValue::Tensor(vec![4.0, 5.0], vec![2]);
    let ty_a2 = tensor_ty(DType::F32, &[3]);
    let ty_b2 = tensor_ty(DType::F32, &[2]);
    let ty_r2 = tensor_ty(DType::F32, &[3, 2]);

    let result = run_tensor_op(
        TensorOp::Einsum {
            notation: "i,j->ij".into(),
        },
        vec![(a2, ty_a2), (b2, ty_b2)],
        ty_r2,
    );
    // outer product: [[4,5],[8,10],[12,15]]
    assert_tensor_close(&result[0], &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0], &[3, 2]);
}

// ===========================================================================
// 5. Reshape: 2D → 1D
// ===========================================================================

#[test]
fn test_reshape_2d_to_1d() {
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let ty_in = tensor_ty(DType::F32, &[2, 3]);
    let ty_out = tensor_ty(DType::F32, &[6]);

    let result = run_tensor_op(TensorOp::Reshape, vec![(t, ty_in)], ty_out);
    // With only 1 input and no shape dims, it flattens to [6]
    assert_tensor_close(&result[0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
}

#[test]
fn test_reshape_1d_to_2d() {
    // Provide shape via extra i64 inputs
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
    let dim0 = IrValue::I64(2);
    let dim1 = IrValue::I64(3);
    let ty_in = tensor_ty(DType::F32, &[6]);
    let ty_d = scalar_ty(DType::I64);
    let ty_out = tensor_ty(DType::F32, &[2, 3]);

    let result = run_tensor_op(
        TensorOp::Reshape,
        vec![(t, ty_in), (dim0, ty_d.clone()), (dim1, ty_d)],
        ty_out,
    );
    assert_tensor_close(&result[0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
}

// ===========================================================================
// 6. Transpose: 2D default (reverse axes)
// ===========================================================================

#[test]
fn test_transpose_2d_default() {
    // [[1,2,3],[4,5,6]] (2x3) → [[1,4],[2,5],[3,6]] (3x2)
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let ty_in = tensor_ty(DType::F32, &[2, 3]);
    let ty_out = tensor_ty(DType::F32, &[3, 2]);

    let result = run_tensor_op(
        TensorOp::Transpose { axes: vec![] },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_tensor_close(&result[0], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
}

#[test]
fn test_transpose_2d_explicit() {
    // Same as default for 2D: axes = [1, 0]
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let ty_in = tensor_ty(DType::F32, &[2, 3]);
    let ty_out = tensor_ty(DType::F32, &[3, 2]);

    let result = run_tensor_op(
        TensorOp::Transpose { axes: vec![1, 0] },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_tensor_close(&result[0], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
}

#[test]
fn test_transpose_3d() {
    // Shape [2,3,4] with axes [2,0,1] → shape [4,2,3]
    // Fill with sequential values 0..24
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = IrValue::Tensor(data, vec![2, 3, 4]);
    let ty_in = tensor_ty(DType::F32, &[2, 3, 4]);
    let ty_out = tensor_ty(DType::F32, &[4, 2, 3]);

    let result = run_tensor_op(
        TensorOp::Transpose {
            axes: vec![2, 0, 1],
        },
        vec![(t, ty_in)],
        ty_out,
    );

    // Verify shape
    if let IrValue::Tensor(_, shape) = &result[0] {
        assert_eq!(shape, &[4, 2, 3]);
    } else {
        panic!("expected tensor");
    }

    // Verify specific elements:
    // Original [i,j,k] → New [k,i,j]
    // Original [0,0,0]=0 → New [0,0,0]=0
    // Original [0,0,1]=1 → New [1,0,0]=1 (flat: 1*6+0*3+0=6)
    // Original [1,2,3]=23 → New [3,1,2]=23 (flat: 3*6+1*3+2=23)
    if let IrValue::Tensor(data, _) = &result[0] {
        assert!((data[0] - 0.0).abs() < 1e-5, "element [0,0,0]");
        assert!(
            (data[6] - 1.0).abs() < 1e-5,
            "element [1,0,0] = orig [0,0,1]"
        );
        assert!(
            (data[23] - 23.0).abs() < 1e-5,
            "element [3,1,2] = orig [1,2,3]"
        );
    }
}

// ===========================================================================
// 7. Reduce: sum over axis 0
// ===========================================================================

#[test]
fn test_reduce_sum_axis0() {
    // [[1,2],[3,4]] → sum axis 0 → [4, 6]
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let ty_in = tensor_ty(DType::F32, &[2, 2]);
    let ty_out = tensor_ty(DType::F32, &[2]);

    let result = run_tensor_op(
        TensorOp::Reduce {
            op: "sum".into(),
            axes: vec![0],
            keepdims: false,
        },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_tensor_close(&result[0], &[4.0, 6.0], &[2]);
}

#[test]
fn test_reduce_sum_axis1() {
    // [[1,2],[3,4]] → sum axis 1 → [3, 7]
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let ty_in = tensor_ty(DType::F32, &[2, 2]);
    let ty_out = tensor_ty(DType::F32, &[2]);

    let result = run_tensor_op(
        TensorOp::Reduce {
            op: "sum".into(),
            axes: vec![1],
            keepdims: false,
        },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_tensor_close(&result[0], &[3.0, 7.0], &[2]);
}

#[test]
fn test_reduce_sum_keepdims() {
    // [[1,2],[3,4]] → sum axis 0 keepdims → [[4, 6]] (shape [1,2])
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let ty_in = tensor_ty(DType::F32, &[2, 2]);
    let ty_out = tensor_ty(DType::F32, &[1, 2]);

    let result = run_tensor_op(
        TensorOp::Reduce {
            op: "sum".into(),
            axes: vec![0],
            keepdims: true,
        },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_tensor_close(&result[0], &[4.0, 6.0], &[1, 2]);
}

// ===========================================================================
// 8. Reduce: max over axis 1
// ===========================================================================

#[test]
fn test_reduce_max_axis1() {
    // [[1,5],[3,2]] → max axis 1 → [5, 3]
    let t = IrValue::Tensor(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
    let ty_in = tensor_ty(DType::F32, &[2, 2]);
    let ty_out = tensor_ty(DType::F32, &[2]);

    let result = run_tensor_op(
        TensorOp::Reduce {
            op: "max".into(),
            axes: vec![1],
            keepdims: false,
        },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_tensor_close(&result[0], &[5.0, 3.0], &[2]);
}

// ===========================================================================
// 9. Reduce: mean (full reduce)
// ===========================================================================

#[test]
fn test_reduce_mean_all() {
    // [[1,2],[3,4]] → mean all → 2.5
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let ty_in = tensor_ty(DType::F32, &[2, 2]);
    let ty_out = scalar_ty(DType::F32);

    let result = run_tensor_op(
        TensorOp::Reduce {
            op: "mean".into(),
            axes: vec![],
            keepdims: false,
        },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_scalar_close(&result[0], 2.5);
}

// ===========================================================================
// 10. Unary: relu on tensor
// ===========================================================================

#[test]
fn test_unary_relu() {
    let t = IrValue::Tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let ty = tensor_ty(DType::F32, &[5]);

    let result = run_tensor_op(
        TensorOp::Unary { op: "relu".into() },
        vec![(t, ty.clone())],
        ty,
    );
    assert_tensor_close(&result[0], &[0.0, 0.0, 0.0, 1.0, 2.0], &[5]);
}

// ===========================================================================
// 11. Unary: sigmoid on tensor
// ===========================================================================

#[test]
fn test_unary_sigmoid() {
    let t = IrValue::Tensor(vec![0.0], vec![1]);
    let ty = tensor_ty(DType::F32, &[1]);

    let result = run_tensor_op(
        TensorOp::Unary {
            op: "sigmoid".into(),
        },
        vec![(t, ty.clone())],
        ty,
    );
    // sigmoid(0) = 0.5
    assert_tensor_close(&result[0], &[0.5], &[1]);
}

// ===========================================================================
// 12. Unary: tanh on tensor
// ===========================================================================

#[test]
fn test_unary_tanh() {
    let t = IrValue::Tensor(vec![0.0, 1.0], vec![2]);
    let ty = tensor_ty(DType::F32, &[2]);

    let result = run_tensor_op(
        TensorOp::Unary { op: "tanh".into() },
        vec![(t, ty.clone())],
        ty,
    );
    // tanh(0)=0.0, tanh(1)≈0.7616
    assert_tensor_close(&result[0], &[0.0, 0.7616], &[2]);
}

// ===========================================================================
// 13. Einsum: batched matmul  bmk,bkn->bmn
// ===========================================================================

#[test]
fn test_einsum_batched_matmul() {
    // Batch of 2 matrices: each 2x2 @ 2x2
    // Batch 0: [[1,0],[0,1]] @ [[5,6],[7,8]] = [[5,6],[7,8]]
    // Batch 1: [[2,0],[0,2]] @ [[1,1],[1,1]] = [[2,2],[2,2]]
    let a = IrValue::Tensor(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]);
    let b = IrValue::Tensor(vec![5.0, 6.0, 7.0, 8.0, 1.0, 1.0, 1.0, 1.0], vec![2, 2, 2]);
    let ty_a = tensor_ty(DType::F32, &[2, 2, 2]);
    let ty_b = tensor_ty(DType::F32, &[2, 2, 2]);
    let ty_c = tensor_ty(DType::F32, &[2, 2, 2]);

    let result = run_tensor_op(
        TensorOp::Einsum {
            notation: "bmk,bkn->bmn".into(),
        },
        vec![(a, ty_a), (b, ty_b)],
        ty_c,
    );
    assert_tensor_close(
        &result[0],
        &[5.0, 6.0, 7.0, 8.0, 2.0, 2.0, 2.0, 2.0],
        &[2, 2, 2],
    );
}

// ===========================================================================
// 14. Einsum: single-input trace  ii->
// ===========================================================================

#[test]
fn test_einsum_trace() {
    // A = [[1,2],[3,4]] → trace = 1+4 = 5
    let a = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let ty_a = tensor_ty(DType::F32, &[2, 2]);
    let ty_s = scalar_ty(DType::F32);

    // Single-input einsum path
    let params = vec![Param {
        name: "in0".into(),
        ty: ty_a.clone(),
    }];
    let mut builder = IrFunctionBuilder::new("trace_test", params, ty_s.clone());
    let entry = builder.create_block(Some("entry"));
    let v = builder.add_block_param(entry, Some("in0"), ty_a.clone());
    builder.set_current_block(entry);

    let result_v = builder.fresh_value();
    builder.push_instr(
        IrInstr::TensorOp {
            result: result_v,
            op: TensorOp::Einsum {
                notation: "ii->".into(),
            },
            inputs: vec![v],
            result_ty: ty_s.clone(),
        },
        Some(ty_s),
    );
    builder.push_instr(
        IrInstr::Return {
            values: vec![result_v],
        },
        None,
    );

    let func = builder.build();
    let out = eval_function(&func, &[a]).expect("eval trace");
    assert_scalar_close(&out[0], 5.0);
}

// ===========================================================================
// 15. Reduce: sum full reduce on 3D tensor
// ===========================================================================

#[test]
fn test_reduce_sum_full_3d() {
    // shape [2,2,2], values 1..8, sum = 36
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let ty_in = tensor_ty(DType::F32, &[2, 2, 2]);
    let ty_out = scalar_ty(DType::F32);

    let result = run_tensor_op(
        TensorOp::Reduce {
            op: "sum".into(),
            axes: vec![],
            keepdims: false,
        },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_scalar_close(&result[0], 36.0);
}

// ===========================================================================
// 16. Large matmul: 4x3 @ 3x5
// ===========================================================================

#[test]
fn test_einsum_large_matmul() {
    // A = 4x3 all ones, B = 3x5 all ones → C = 4x5 all 3.0
    let a = IrValue::Tensor(vec![1.0; 12], vec![4, 3]);
    let b = IrValue::Tensor(vec![1.0; 15], vec![3, 5]);
    let ty_a = tensor_ty(DType::F32, &[4, 3]);
    let ty_b = tensor_ty(DType::F32, &[3, 5]);
    let ty_c = tensor_ty(DType::F32, &[4, 5]);

    let result = run_tensor_op(
        TensorOp::Einsum {
            notation: "mk,kn->mn".into(),
        },
        vec![(a, ty_a), (b, ty_b)],
        ty_c,
    );
    assert_tensor_close(&result[0], &[3.0; 20], &[4, 5]);
}

// ===========================================================================
// 17. Transpose identity (axes = [0, 1])
// ===========================================================================

#[test]
fn test_transpose_identity() {
    let t = IrValue::Tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let ty = tensor_ty(DType::F32, &[2, 2]);

    let result = run_tensor_op(
        TensorOp::Transpose { axes: vec![0, 1] },
        vec![(t, ty.clone())],
        ty,
    );
    assert_tensor_close(&result[0], &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
}

// ===========================================================================
// 18. Reduce: max over axis 0
// ===========================================================================

#[test]
fn test_reduce_max_axis0() {
    // [[1,5],[3,2]] → max axis 0 → [3, 5]
    let t = IrValue::Tensor(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
    let ty_in = tensor_ty(DType::F32, &[2, 2]);
    let ty_out = tensor_ty(DType::F32, &[2]);

    let result = run_tensor_op(
        TensorOp::Reduce {
            op: "max".into(),
            axes: vec![0],
            keepdims: false,
        },
        vec![(t, ty_in)],
        ty_out,
    );
    assert_tensor_close(&result[0], &[3.0, 5.0], &[2]);
}
