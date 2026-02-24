//! Integration tests for Phase 5:
//! - DCE: dead code elimination removes unused instructions
//! - CSE: common subexpression elimination deduplicates identical instructions
//! - OpExpand: abstract calls to elementwise activations become concrete TensorOps
//! - ShapeCheck: tensor shape consistency validation

use iris::ir::function::Param;
use iris::ir::instr::{IrInstr, TensorOp};
use iris::ir::module::{IrFunctionBuilder, IrModule};
use iris::ir::types::{DType, Dim, IrType, Shape};
use iris::pass::type_infer::TypeInferPass;
use iris::pass::validate::ValidatePass;
use iris::pass::{CsePass, DcePass, OpExpandPass, PassManager, ShapeCheckPass};
use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. DCE: removes a dead constant that is never used
// ---------------------------------------------------------------------------

#[test]
fn test_dce_removes_dead_const() {
    // `99` is bound to `ignored` but never referenced; DCE must remove it.
    let src = r#"
def withunused(x: f32) -> f32 {
    val ignored = 99;
    x
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        !output.contains("const.i"),
        "DCE should remove the dead const.i instruction\n{}",
        output
    );
    assert!(output.contains("return"), "return must still be present\n{}", output);
}

// ---------------------------------------------------------------------------
// 2. DCE: preserves every instruction that contributes to the return value
// ---------------------------------------------------------------------------

#[test]
fn test_dce_preserves_used_instrs() {
    let src = r#"
def addtwo(x: f32, y: f32) -> f32 {
    x + y
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("= add "),
        "DCE must not remove the live add instruction\n{}",
        output
    );
    assert!(output.contains("return"), "return must be present\n{}", output);
}

// ---------------------------------------------------------------------------
// 3. CSE: deduplicates two identical float constants
// ---------------------------------------------------------------------------

#[test]
fn test_cse_deduplicates_const() {
    // Two identical `1.0` literals produce two `const.f` instructions before CSE.
    // After CSE only one should survive.
    let src = r#"
def samecst(x: f32) -> f32 {
    val a = 1.0;
    val b = 1.0;
    a + b
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    let count = output.matches("const.f").count();
    assert_eq!(
        count, 1,
        "CSE should leave exactly 1 const.f, found {}\n{}",
        count, output
    );
}

// ---------------------------------------------------------------------------
// 4. CSE: deduplicates an identical binary operation
// ---------------------------------------------------------------------------

#[test]
fn test_cse_deduplicates_binop() {
    // `x + y` is computed twice; CSE should eliminate the duplicate.
    // Before CSE: 3 `add` instructions. After CSE: 2.
    let src = r#"
def dupbinop(x: f32, y: f32) -> f32 {
    val a = x + y;
    val b = x + y;
    a + b
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    let add_count = output.matches("= add ").count();
    assert!(
        add_count < 3,
        "CSE should reduce 3 add instructions to 2, found {}\n{}",
        add_count,
        output
    );
}

// ---------------------------------------------------------------------------
// 5. OpExpand: ReLU call is replaced by tensorop.unary.relu
// ---------------------------------------------------------------------------

#[test]
fn test_op_expand_relu() {
    let src = r#"
model ReLUNet {
  input x: tensor<f32, [1, 64]>
  layer h ReLU(x)
  output h
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("unary.relu"),
        "OpExpandPass should replace call @ReLU with tensorop.unary.relu\n{}",
        output
    );
    assert!(
        !output.contains("call @ReLU"),
        "call @ReLU should have been replaced by OpExpandPass\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 6. OpExpand: Dense call is preserved (requires weights; not elementwise)
// ---------------------------------------------------------------------------

#[test]
fn test_op_expand_preserves_dense() {
    let src = r#"
model DenseNet {
  input x: tensor<f32, [1, 64]>
  layer h Dense(units=128)
  output h
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("call @Dense"),
        "Dense should remain as a Call after OpExpandPass\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 7. ShapeCheck: a model with consistent tensor shapes passes all passes
// ---------------------------------------------------------------------------

#[test]
fn test_shape_check_valid() {
    let src = r#"
model ValidNet {
  input x: tensor<f32, [1, 64]>
  layer h ReLU(x)
  output h
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "valid model should pass all passes: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// 8. ShapeCheck: rejects a reshape whose element count does not match
// ---------------------------------------------------------------------------

#[test]
fn test_shape_check_invalid_reshape() {
    // Reshape from [2, 4] (8 elements) to [3, 3] (9 elements) — mismatch.
    let mut module = IrModule::new("reshape_test");

    let in_ty = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(2), Dim::Literal(4)]),
    };
    let out_ty = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(3), Dim::Literal(3)]),
    };

    let params = vec![Param { name: "x".into(), ty: in_ty.clone() }];
    let mut builder = IrFunctionBuilder::new("bad_reshape", params, out_ty.clone());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), in_ty);
    builder.set_current_block(entry);

    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::TensorOp {
            result,
            op: TensorOp::Reshape,
            inputs: vec![x],
            result_ty: out_ty.clone(),
        },
        Some(out_ty),
    );
    builder.push_instr(IrInstr::Return { values: vec![result] }, None);
    module.add_function(builder.build()).unwrap();

    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    pm.add_pass(OpExpandPass);
    pm.add_pass(DcePass);
    pm.add_pass(CsePass);
    pm.add_pass(ShapeCheckPass);

    let err = pm.run(&mut module);
    assert!(err.is_err(), "bad reshape must be rejected by ShapeCheckPass");
    let (pass_name, _) = err.unwrap_err();
    assert_eq!(pass_name, "shape-check", "error should originate from shape-check pass");
}
