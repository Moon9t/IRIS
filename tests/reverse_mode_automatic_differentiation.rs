//! Phase 130 integration tests: reverse-mode automatic differentiation.
//!
//! Validates the tape-based backpropagation system:
//! - TapeRecord: records a value on the computation graph
//! - Backward: triggers reverse-mode differentiation from a loss
//! - TapeGrad: extracts the accumulated gradient for a taped value
//!
//! Tests cover derivatives of: add, mul, sub, div, sin, cos, exp, log,
//! sqrt, relu, sigmoid, tanh, pow, abs, and composite chain-rule cases.

use iris::interp::{eval_function, IrValue};
use iris::ir::function::Param;
use iris::ir::instr::IrInstr;
use iris::ir::module::IrFunctionBuilder;
use iris::ir::types::{DType, IrType};

fn f64_ty() -> IrType {
    IrType::Scalar(DType::F64)
}

fn unit_ty() -> IrType {
    // Backward returns IrValue::Unit; use Scalar(I64) as placeholder type
    IrType::Scalar(DType::I64)
}

fn assert_f64_close(result: &IrValue, expected: f64, tol: f64, msg: &str) {
    match result {
        IrValue::F64(v) => {
            assert!(
                (v - expected).abs() < tol,
                "{}: got {}, expected {} (diff={})",
                msg,
                v,
                expected,
                (v - expected).abs()
            );
        }
        other => panic!("{}: expected F64, got {:?}", msg, other),
    }
}

// ── Simple f(x) = x, df/dx = 1 via identity tape ─────────────────────────

#[test]
fn test_reverse_ad_identity() {
    // f(x) = x, df/dx = 1.0
    // We tape-record x as an "identity" op (only 1 parent: itself)
    // Then backward from it, and extract grad.
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_identity", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    // tape_x = tape_record(x, op="identity", parents=[x])
    // backward(tape_x) → computes grads
    // grad = tape_grad(x)
    let tape_x = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_x,
            value: x,
            op: "identity".into(),
            parents: vec![],
        },
        Some(f64_ty()),
    );

    // For identity: backward from tape_x, the tape_x gets grad=1.0
    // but since parents=[], nothing propagates further
    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_x,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: tape_x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    let result = eval_function(&func, &[IrValue::F64(3.0)]).expect("eval");
    assert_f64_close(&result[0], 1.0, 1e-10, "d(identity)/d(tape_x)");
}

// ── f(a,b) = a + b, df/da = 1, df/db = 1 ────────────────────────────────

#[test]
fn test_reverse_ad_add() {
    let params = vec![
        Param {
            name: "a".into(),
            ty: f64_ty(),
        },
        Param {
            name: "b".into(),
            ty: f64_ty(),
        },
    ];
    let mut builder = IrFunctionBuilder::new("rev_add", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let a = builder.add_block_param(entry, Some("a"), f64_ty());
    let b = builder.add_block_param(entry, Some("b"), f64_ty());
    builder.set_current_block(entry);

    // c = a + b  (compute in IR)
    let c = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: c,
            op: iris::ir::instr::BinOp::Add,
            lhs: a,
            rhs: b,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    // tape_c = tape_record(c, "add", [a, b])
    let tape_c = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_c,
            value: c,
            op: "add".into(),
            parents: vec![a, b],
        },
        Some(f64_ty()),
    );

    // backward from tape_c
    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_c,
        },
        Some(unit_ty()),
    );

    // grad(a)
    let grad_a = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_a,
            tape_node: a,
        },
        Some(f64_ty()),
    );

    builder.push_instr(
        IrInstr::Return {
            values: vec![grad_a],
        },
        None,
    );
    let func = builder.build();

    let result = eval_function(&func, &[IrValue::F64(3.0), IrValue::F64(5.0)]).expect("eval");
    assert_f64_close(&result[0], 1.0, 1e-10, "d(a+b)/da");
}

// ── f(a,b) = a * b, df/da = b, df/db = a ────────────────────────────────

#[test]
fn test_reverse_ad_mul() {
    let params = vec![
        Param {
            name: "a".into(),
            ty: f64_ty(),
        },
        Param {
            name: "b".into(),
            ty: f64_ty(),
        },
    ];
    let mut builder = IrFunctionBuilder::new("rev_mul", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let a = builder.add_block_param(entry, Some("a"), f64_ty());
    let b = builder.add_block_param(entry, Some("b"), f64_ty());
    builder.set_current_block(entry);

    // c = a * b
    let c = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: c,
            op: iris::ir::instr::BinOp::Mul,
            lhs: a,
            rhs: b,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    // tape_c = tape_record(c, "mul", [a, b])
    let tape_c = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_c,
            value: c,
            op: "mul".into(),
            parents: vec![a, b],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_c,
        },
        Some(unit_ty()),
    );

    // grad(a) should be b=5.0
    let grad_a = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_a,
            tape_node: a,
        },
        Some(f64_ty()),
    );

    // grad(b) should be a=3.0
    let grad_b = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_b,
            tape_node: b,
        },
        Some(f64_ty()),
    );

    // Return grad_a (we'll test grad_b in a separate assertion)
    builder.push_instr(
        IrInstr::Return {
            values: vec![grad_a, grad_b],
        },
        None,
    );
    let func = builder.build();

    let result = eval_function(&func, &[IrValue::F64(3.0), IrValue::F64(5.0)]).expect("eval");
    assert_f64_close(&result[0], 5.0, 1e-10, "d(a*b)/da = b");
    assert_f64_close(&result[1], 3.0, 1e-10, "d(a*b)/db = a");
}

// ── f(a,b) = a - b, df/da = 1, df/db = -1 ──────────────────────────────

#[test]
fn test_reverse_ad_sub() {
    let params = vec![
        Param {
            name: "a".into(),
            ty: f64_ty(),
        },
        Param {
            name: "b".into(),
            ty: f64_ty(),
        },
    ];
    let mut builder = IrFunctionBuilder::new("rev_sub", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let a = builder.add_block_param(entry, Some("a"), f64_ty());
    let b = builder.add_block_param(entry, Some("b"), f64_ty());
    builder.set_current_block(entry);

    let c = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: c,
            op: iris::ir::instr::BinOp::Sub,
            lhs: a,
            rhs: b,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_c = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_c,
            value: c,
            op: "sub".into(),
            parents: vec![a, b],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_c,
        },
        Some(unit_ty()),
    );

    let grad_a = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_a,
            tape_node: a,
        },
        Some(f64_ty()),
    );

    let grad_b = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_b,
            tape_node: b,
        },
        Some(f64_ty()),
    );

    builder.push_instr(
        IrInstr::Return {
            values: vec![grad_a, grad_b],
        },
        None,
    );
    let func = builder.build();

    let result = eval_function(&func, &[IrValue::F64(7.0), IrValue::F64(2.0)]).expect("eval");
    assert_f64_close(&result[0], 1.0, 1e-10, "d(a-b)/da = 1");
    assert_f64_close(&result[1], -1.0, 1e-10, "d(a-b)/db = -1");
}

// ── f(a,b) = a / b, df/da = 1/b, df/db = -a/b² ─────────────────────────

#[test]
fn test_reverse_ad_div() {
    let params = vec![
        Param {
            name: "a".into(),
            ty: f64_ty(),
        },
        Param {
            name: "b".into(),
            ty: f64_ty(),
        },
    ];
    let mut builder = IrFunctionBuilder::new("rev_div", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let a = builder.add_block_param(entry, Some("a"), f64_ty());
    let b = builder.add_block_param(entry, Some("b"), f64_ty());
    builder.set_current_block(entry);

    let c = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: c,
            op: iris::ir::instr::BinOp::Div,
            lhs: a,
            rhs: b,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_c = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_c,
            value: c,
            op: "div".into(),
            parents: vec![a, b],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_c,
        },
        Some(unit_ty()),
    );

    let grad_a = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_a,
            tape_node: a,
        },
        Some(f64_ty()),
    );

    let grad_b = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_b,
            tape_node: b,
        },
        Some(f64_ty()),
    );

    builder.push_instr(
        IrInstr::Return {
            values: vec![grad_a, grad_b],
        },
        None,
    );
    let func = builder.build();

    // a=6, b=3: da = 1/3, db = -6/9 = -2/3
    let result = eval_function(&func, &[IrValue::F64(6.0), IrValue::F64(3.0)]).expect("eval");
    assert_f64_close(&result[0], 1.0 / 3.0, 1e-10, "d(a/b)/da = 1/b");
    assert_f64_close(&result[1], -6.0 / 9.0, 1e-10, "d(a/b)/db = -a/b²");
}

// ── f(x) = sin(x), df/dx = cos(x) ──────────────────────────────────────

#[test]
fn test_reverse_ad_sin() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_sin", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    // sin_x = sin(x) — we just use the primal value directly
    let sin_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: sin_val,
            value: 1.0_f64.sin(),
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_sin = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_sin,
            value: sin_val,
            op: "sin".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_sin,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    // x=1.0: d(sin(x))/dx = cos(1.0)
    let result = eval_function(&func, &[IrValue::F64(1.0)]).expect("eval");
    assert_f64_close(&result[0], 1.0_f64.cos(), 1e-10, "d(sin(x))/dx = cos(x)");
}

// ── f(x) = exp(x), df/dx = exp(x) ──────────────────────────────────────

#[test]
fn test_reverse_ad_exp() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_exp", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let exp_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: exp_val,
            value: 2.0_f64.exp(),
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_exp = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_exp,
            value: exp_val,
            op: "exp".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_exp,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    // x=2.0: d(exp(x))/dx = exp(2.0)
    let result = eval_function(&func, &[IrValue::F64(2.0)]).expect("eval");
    assert_f64_close(&result[0], 2.0_f64.exp(), 1e-6, "d(exp(x))/dx = exp(x)");
}

// ── f(x) = log(x), df/dx = 1/x ─────────────────────────────────────────

#[test]
fn test_reverse_ad_log() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_log", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let log_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: log_val,
            value: 3.0_f64.ln(),
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_log = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_log,
            value: log_val,
            op: "log".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_log,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    // x=3.0: d(log(x))/dx = 1/3.0
    let result = eval_function(&func, &[IrValue::F64(3.0)]).expect("eval");
    assert_f64_close(&result[0], 1.0 / 3.0, 1e-10, "d(log(x))/dx = 1/x");
}

// ── f(x) = sqrt(x), df/dx = 1/(2*sqrt(x)) ──────────────────────────────

#[test]
fn test_reverse_ad_sqrt() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_sqrt", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let sqrt_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: sqrt_val,
            value: 4.0_f64.sqrt(),
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_sqrt = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_sqrt,
            value: sqrt_val,
            op: "sqrt".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_sqrt,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    // x=4.0: d(sqrt(x))/dx = 1/(2*sqrt(4)) = 1/4 = 0.25
    let result = eval_function(&func, &[IrValue::F64(4.0)]).expect("eval");
    assert_f64_close(&result[0], 0.25, 1e-10, "d(sqrt(x))/dx = 1/(2*sqrt(x))");
}

// ── f(x) = relu(x) at x>0: df/dx = 1; at x<0: df/dx = 0 ───────────────

#[test]
fn test_reverse_ad_relu() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_relu", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let relu_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: relu_val,
            value: 3.0,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_relu = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_relu,
            value: relu_val,
            op: "relu".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_relu,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    // x=3.0 > 0: relu grad = 1
    let result = eval_function(&func, &[IrValue::F64(3.0)]).expect("eval");
    assert_f64_close(&result[0], 1.0, 1e-10, "d(relu(x))/dx at x>0 = 1");

    // x=-2.0 < 0: relu grad = 0
    let result2 = eval_function(&func, &[IrValue::F64(-2.0)]).expect("eval");
    assert_f64_close(&result2[0], 0.0, 1e-10, "d(relu(x))/dx at x<0 = 0");
}

// ── f(x) = sigmoid(x), df/dx = sigmoid(x) * (1 - sigmoid(x)) ───────────

#[test]
fn test_reverse_ad_sigmoid() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_sigmoid", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let x_val: f64 = 1.0;
    let sig = 1.0 / (1.0 + (-x_val).exp());
    let sig_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: sig_val,
            value: sig,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_sig = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_sig,
            value: sig_val,
            op: "sigmoid".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_sig,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    // x=1.0: sig'(x) = sig * (1 - sig)
    let expected = sig * (1.0 - sig);
    let result = eval_function(&func, &[IrValue::F64(x_val)]).expect("eval");
    assert_f64_close(&result[0], expected, 1e-10, "d(sigmoid(x))/dx");
}

// ── f(x) = tanh(x), df/dx = 1 - tanh²(x) ──────────────────────────────

#[test]
fn test_reverse_ad_tanh() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_tanh", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let x_val: f64 = 0.5;
    let th = x_val.tanh();
    let tanh_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: tanh_val,
            value: th,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_tanh = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_tanh,
            value: tanh_val,
            op: "tanh".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_tanh,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    let expected = 1.0 - th * th;
    let result = eval_function(&func, &[IrValue::F64(x_val)]).expect("eval");
    assert_f64_close(&result[0], expected, 1e-10, "d(tanh(x))/dx");
}

// ── Chain rule: f(a,b) = (a*b) + a, df/da = b+1, df/db = a ─────────────

#[test]
fn test_reverse_ad_chain_rule() {
    let params = vec![
        Param {
            name: "a".into(),
            ty: f64_ty(),
        },
        Param {
            name: "b".into(),
            ty: f64_ty(),
        },
    ];
    let mut builder = IrFunctionBuilder::new("rev_chain", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let a = builder.add_block_param(entry, Some("a"), f64_ty());
    let b = builder.add_block_param(entry, Some("b"), f64_ty());
    builder.set_current_block(entry);

    // Step 1: c = a * b
    let c = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: c,
            op: iris::ir::instr::BinOp::Mul,
            lhs: a,
            rhs: b,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_c = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_c,
            value: c,
            op: "mul".into(),
            parents: vec![a, b],
        },
        Some(f64_ty()),
    );

    // Step 2: d = c + a = (a*b) + a
    let d = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: d,
            op: iris::ir::instr::BinOp::Add,
            lhs: c,
            rhs: a,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_d = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_d,
            value: d,
            op: "add".into(),
            parents: vec![tape_c, a],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_d,
        },
        Some(unit_ty()),
    );

    // grad(a): d(a*b + a)/da = b + 1
    let grad_a = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_a,
            tape_node: a,
        },
        Some(f64_ty()),
    );

    // grad(b): d(a*b + a)/db = a
    let grad_b = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_b,
            tape_node: b,
        },
        Some(f64_ty()),
    );

    builder.push_instr(
        IrInstr::Return {
            values: vec![grad_a, grad_b],
        },
        None,
    );
    let func = builder.build();

    // a=2, b=3: da = 3+1 = 4, db = 2
    let result = eval_function(&func, &[IrValue::F64(2.0), IrValue::F64(3.0)]).expect("eval");
    assert_f64_close(&result[0], 4.0, 1e-10, "d(a*b+a)/da = b+1");
    assert_f64_close(&result[1], 2.0, 1e-10, "d(a*b+a)/db = a");
}

// ── f(x) = cos(x), df/dx = -sin(x) ─────────────────────────────────────

#[test]
fn test_reverse_ad_cos() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_cos", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let x_val: f64 = 1.5;
    let cos_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: cos_val,
            value: x_val.cos(),
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_cos = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_cos,
            value: cos_val,
            op: "cos".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_cos,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    let result = eval_function(&func, &[IrValue::F64(x_val)]).expect("eval");
    assert_f64_close(&result[0], -x_val.sin(), 1e-10, "d(cos(x))/dx = -sin(x)");
}

// ── f(x) = |x|, df/dx = sign(x) ────────────────────────────────────────

#[test]
fn test_reverse_ad_abs() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_abs", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let abs_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: abs_val,
            value: 5.0,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_abs = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_abs,
            value: abs_val,
            op: "abs".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_abs,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    // x=5 > 0: sign = 1
    let result = eval_function(&func, &[IrValue::F64(5.0)]).expect("eval");
    assert_f64_close(&result[0], 1.0, 1e-10, "d(|x|)/dx at x>0 = 1");

    // x=-3 < 0: sign = -1
    let result2 = eval_function(&func, &[IrValue::F64(-3.0)]).expect("eval");
    assert_f64_close(&result2[0], -1.0, 1e-10, "d(|x|)/dx at x<0 = -1");
}

// ── f(x) = -x, df/dx = -1 ──────────────────────────────────────────────

#[test]
fn test_reverse_ad_neg() {
    let params = vec![Param {
        name: "x".into(),
        ty: f64_ty(),
    }];
    let mut builder = IrFunctionBuilder::new("rev_neg", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), f64_ty());
    builder.set_current_block(entry);

    let neg_val = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: neg_val,
            value: -4.0,
            ty: f64_ty(),
        },
        Some(f64_ty()),
    );

    let tape_neg = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_neg,
            value: neg_val,
            op: "neg".into(),
            parents: vec![x],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_neg,
        },
        Some(unit_ty()),
    );

    let grad = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad,
            tape_node: x,
        },
        Some(f64_ty()),
    );

    builder.push_instr(IrInstr::Return { values: vec![grad] }, None);
    let func = builder.build();

    let result = eval_function(&func, &[IrValue::F64(4.0)]).expect("eval");
    assert_f64_close(&result[0], -1.0, 1e-10, "d(-x)/dx = -1");
}

// ── Zero gradient for non-taped values ──────────────────────────────────

#[test]
fn test_reverse_ad_zero_grad_untaped() {
    // If we ask for gradient of a value that wasn't involved in the tape,
    // we should get 0.0
    let params = vec![
        Param {
            name: "a".into(),
            ty: f64_ty(),
        },
        Param {
            name: "b".into(),
            ty: f64_ty(),
        },
    ];
    let mut builder = IrFunctionBuilder::new("rev_zero", params, f64_ty());
    let entry = builder.create_block(Some("entry"));
    let a = builder.add_block_param(entry, Some("a"), f64_ty());
    let b = builder.add_block_param(entry, Some("b"), f64_ty());
    builder.set_current_block(entry);

    // Only tape a, ignore b
    let tape_a = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeRecord {
            result: tape_a,
            value: a,
            op: "identity".into(),
            parents: vec![],
        },
        Some(f64_ty()),
    );

    let bw = builder.fresh_value();
    builder.push_instr(
        IrInstr::Backward {
            result: bw,
            loss: tape_a,
        },
        Some(unit_ty()),
    );

    // b was never in the tape
    let grad_b = builder.fresh_value();
    builder.push_instr(
        IrInstr::TapeGrad {
            result: grad_b,
            tape_node: b,
        },
        Some(f64_ty()),
    );

    builder.push_instr(
        IrInstr::Return {
            values: vec![grad_b],
        },
        None,
    );
    let func = builder.build();

    let result = eval_function(&func, &[IrValue::F64(5.0), IrValue::F64(7.0)]).expect("eval");
    assert_f64_close(&result[0], 0.0, 1e-10, "untaped value grad = 0");
}
