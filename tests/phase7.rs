//! Integration tests for Phase 7:
//! - Unary operators: `-x` (neg) and `!b` (not) parse, lower, and survive all passes
//! - ConstFoldPass folds `-3.0` to a constant (no `neg` in output)
//! - Real LLVM IR body: arithmetic ops emit `fadd`/`fmul`, if-else emits `fcmp`/`phi`
//! - New ML ops: Conv2D, Flatten, GlobalAveragePool with shape inference

use iris::ir::types::{DType, Dim, IrType, Shape};
use iris::lower::lower_model;
use iris::parser::lexer::Lexer;
use iris::parser::parse::Parser;
use iris::pass::infer_shapes;
use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Unary negation appears in IR
// ---------------------------------------------------------------------------

#[test]
fn test_unary_neg_in_ir() {
    let src = r#"
def negate(x: f32) -> f32 {
    -x
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("neg"),
        "IR must contain a neg instruction\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 2. Boolean NOT appears in IR
// ---------------------------------------------------------------------------

#[test]
fn test_unary_not_in_ir() {
    let src = r#"
def invert(b: bool) -> bool {
    !b
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    assert!(
        output.contains("not"),
        "IR must contain a not instruction\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 3. ConstFoldPass folds -3.0 to a constant (no neg survives)
// ---------------------------------------------------------------------------

#[test]
fn test_const_fold_neg() {
    let src = r#"
def neg_const() -> f32 {
    -3.0
}
"#;
    let output = compile(src, "test", EmitKind::Ir).expect("compile");
    // After ConstFold the neg instruction is replaced by const.f -3
    assert!(
        !output.contains("= neg "),
        "ConstFold should eliminate neg of a literal constant\n{}",
        output
    );
    assert!(
        output.contains("const.f"),
        "folded constant must appear as const.f\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 4. LLVM body: scalar addition emits fadd float
// ---------------------------------------------------------------------------

#[test]
fn test_llvm_body_arithmetic() {
    let src = r#"
def add(x: f32, y: f32) -> f32 {
    x + y
}
"#;
    let output = compile(src, "test", EmitKind::Llvm).expect("compile");
    assert!(
        output.contains("fadd float"),
        "LLVM output must contain 'fadd float' for f32 addition\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 5. LLVM body: if-else emits fcmp, br i1, and phi float
// ---------------------------------------------------------------------------

#[test]
fn test_llvm_body_condbr() {
    let src = r#"
def clamp(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}
"#;
    let output = compile(src, "test", EmitKind::Llvm).expect("compile");
    assert!(
        output.contains("fcmp"),
        "LLVM output must contain 'fcmp' for float comparison\n{}",
        output
    );
    assert!(
        output.contains("br i1"),
        "LLVM output must contain 'br i1' for conditional branch\n{}",
        output
    );
    assert!(
        output.contains("phi float"),
        "LLVM output must contain 'phi float' for if-else merge\n{}",
        output
    );
}

// ---------------------------------------------------------------------------
// 6. New op: Conv2D(filters=16, kernel_size=3, stride=1, padding=0) on [1,3,8,8]
//    → output shape [1,16,6,6]
// ---------------------------------------------------------------------------

#[test]
fn test_new_op_conv2d() {
    let src = r#"
model ConvNet {
  input x: tensor<f32, [1, 3, 8, 8]>
  layer c Conv2D(filters=16, kernel_size=3, stride=1, padding=0)
  output c
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
            Dim::Literal(16),
            Dim::Literal(6), // (8 + 0 - 3) / 1 + 1 = 6
            Dim::Literal(6),
        ]),
    };

    let c = graph.node_by_name("c").expect("c node");
    assert_eq!(
        shapes.get(&c.id()).unwrap(),
        &expected,
        "Conv2D(kernel=3, stride=1, padding=0) on [1,3,8,8] must produce [1,16,6,6]"
    );
}

// ---------------------------------------------------------------------------
// 7. New op: Flatten on [2, 4, 8, 8] → [2, 256]
// ---------------------------------------------------------------------------

#[test]
fn test_new_op_flatten() {
    let src = r#"
model FlatNet {
  input x: tensor<f32, [2, 4, 8, 8]>
  layer f Flatten
  output f
}
"#;
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");
    let graph = lower_model(&ast.models[0]).expect("lower_model");
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let expected = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![
            Dim::Literal(2),
            Dim::Literal(256), // 4 * 8 * 8
        ]),
    };

    let f = graph.node_by_name("f").expect("f node");
    assert_eq!(
        shapes.get(&f.id()).unwrap(),
        &expected,
        "Flatten on [2,4,8,8] must produce [2,256]"
    );
}

// ---------------------------------------------------------------------------
// 8. New op: GlobalAveragePool on [1, 8, 4, 4] → [1, 8, 1, 1]
// ---------------------------------------------------------------------------

#[test]
fn test_new_op_globalavgpool() {
    let src = r#"
model GapNet {
  input x: tensor<f32, [1, 8, 4, 4]>
  layer g GlobalAveragePool
  output g
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
            Dim::Literal(8),
            Dim::Literal(1),
            Dim::Literal(1),
        ]),
    };

    let g = graph.node_by_name("g").expect("g node");
    assert_eq!(
        shapes.get(&g.id()).unwrap(),
        &expected,
        "GlobalAveragePool on [1,8,4,4] must produce [1,8,1,1]"
    );
}
