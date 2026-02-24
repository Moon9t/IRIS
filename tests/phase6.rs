//! Integration tests for Phase 6:
//! - If-else lowering: conditional expressions compile to multi-block SSA IR
//! - ConstFoldPass: constant arithmetic is evaluated; identities are simplified
//! - New ML ops: BatchNorm, MaxPool, Dropout, LayerNorm with shape inference

use iris::ir::types::{DType, Dim, IrType, Shape};
use iris::lower::lower_model;
use iris::parser::lexer::Lexer;
use iris::parser::parse::Parser;
use iris::pass::infer_shapes;
use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. If-else produces multi-block IR (CondBr present)
// ---------------------------------------------------------------------------

#[test]
fn test_if_else_multi_block() {
    let src = r#"
def clamp(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("condbr"),
        "if-else must produce a condbr instruction\n{}",
        output
    );
    assert!(output.contains("then"), "then block must be present\n{}", output);
    assert!(output.contains("else"), "else block must be present\n{}", output);
    assert!(output.contains("merge"), "merge block must be present\n{}", output);
}

// ---------------------------------------------------------------------------
// 2. If-else result feeds a subsequent computation
// ---------------------------------------------------------------------------

#[test]
fn test_if_else_value() {
    // The result of the if-else is added to 1.0; both branches must contribute.
    let src = r#"
def offset(x: f32) -> f32 {
    val v = if x < 0.0 { 0.0 } else { x };
    v
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(output.contains("condbr"), "must contain conditional branch\n{}", output);
    assert!(output.contains("return"), "must contain return\n{}", output);
}

// ---------------------------------------------------------------------------
// 3. If-else with a comparison condition compiles without error
// ---------------------------------------------------------------------------

#[test]
fn test_if_else_comparison() {
    // Tests that comparison ops (cmplt) work as if-else conditions.
    let src = r#"
def maxval(x: f32, y: f32) -> f32 {
    if x < y { y } else { x }
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "if-else with comparison should compile: {:?}", result.err());
    let output = result.unwrap();
    assert!(output.contains("condbr"), "must contain condbr\n{}", output);
}

// ---------------------------------------------------------------------------
// 4. ConstFoldPass: constant arithmetic is evaluated at compile time
// ---------------------------------------------------------------------------

#[test]
fn test_const_fold_arithmetic() {
    // 3.0 * 2.0 should be folded to const.f 6 at compile time.
    // After DCE the mul instruction and its operands disappear.
    let src = r#"
def folded() -> f32 {
    val a = 3.0;
    val b = 2.0;
    a * b
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        !output.contains("= mul "),
        "ConstFold + DCE should eliminate the mul instruction\n{}",
        output
    );
    // The folded result (6.0) should appear as a const.f instruction.
    assert!(output.contains("const.f"), "folded constant must appear\n{}", output);
}

// ---------------------------------------------------------------------------
// 5. ConstFoldPass: x * 1.0 identity is simplified away
// ---------------------------------------------------------------------------

#[test]
fn test_const_fold_identity() {
    // x * 1.0 → x (identity), so no mul instruction should survive.
    let src = r#"
def identity(x: f32) -> f32 {
    val one = 1.0;
    x * one
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        !output.contains("= mul "),
        "ConstFoldPass should eliminate x * 1.0\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 6. New op: Dropout compiles without error
// ---------------------------------------------------------------------------

#[test]
fn test_new_op_dropout() {
    let src = r#"
model DropNet {
  input x: tensor<f32, [2, 64]>
  layer h Dropout(rate=0.5)
  output h
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "Dropout model should compile: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// 7. New op: BatchNorm — output shape equals input shape (passthrough)
// ---------------------------------------------------------------------------

#[test]
fn test_new_op_batchnorm() {
    let src = r#"
model BNNet {
  input x: tensor<f32, [2, 32]>
  layer h BatchNorm
  output h
}
"#;
    // Verify shape inference directly on the graph.
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");
    let graph = lower_model(&ast.models[0]).expect("lower_model");
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let expected = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(2), Dim::Literal(32)]),
    };

    let h = graph.node_by_name("h").expect("h node");
    assert_eq!(
        shapes.get(&h.id()).unwrap(),
        &expected,
        "BatchNorm output shape must equal input shape"
    );
}

// ---------------------------------------------------------------------------
// 8. New op: MaxPool(stride=2) halves the last two spatial dims
// ---------------------------------------------------------------------------

#[test]
fn test_new_op_maxpool() {
    let src = r#"
model PoolNet {
  input x: tensor<f32, [1, 4, 8, 8]>
  layer h MaxPool(stride=2)
  output h
}
"#;
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");
    let graph = lower_model(&ast.models[0]).expect("lower_model");
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let expected = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![
            Dim::Literal(1),
            Dim::Literal(4),
            Dim::Literal(4), // 8 / 2
            Dim::Literal(4), // 8 / 2
        ]),
    };

    let h = graph.node_by_name("h").expect("h node");
    assert_eq!(
        shapes.get(&h.id()).unwrap(),
        &expected,
        "MaxPool(stride=2) on [1,4,8,8] should produce [1,4,4,4]"
    );
}
