//! Integration tests for Phase 4:
//! - Explicit data-flow input refs in layer args
//! - Add and Concat shape inference
//! - DeadNodePass graph optimization
//! - ONNX text export

use iris::ir::graph::GraphIr;
use iris::ir::types::{DType, Dim, IrType, Shape};
use iris::lower::lower_model;
use iris::parser::lexer::Lexer;
use iris::parser::parse::Parser;
use iris::pass::infer_shapes;
use iris::pass::{DeadNodePass, GraphPass, GraphPassManager};
use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn parse_graph(src: &str) -> GraphIr {
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");
    assert_eq!(ast.models.len(), 1, "expected exactly one model");
    lower_model(&ast.models[0]).expect("lower_model")
}

// ---------------------------------------------------------------------------
// 1. Explicit single input ref resolves correctly
// ---------------------------------------------------------------------------

#[test]
fn test_explicit_single_ref() {
    let src = r#"
model M {
  input x: tensor<f32, [4, 64]>
  layer h ReLU(x)
  output h
}
"#;
    let graph = parse_graph(src);
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let x = graph.node_by_name("x").expect("x");
    let h = graph.node_by_name("h").expect("h");

    let x_ty = shapes.get(&x.id()).unwrap();
    let h_ty = shapes.get(&h.id()).unwrap();

    assert_eq!(h_ty, x_ty, "ReLU(x) should pass through x's shape");
}

// ---------------------------------------------------------------------------
// 2. Explicit multi-input ref: Add(h1, x) — both resolve
// ---------------------------------------------------------------------------

#[test]
fn test_explicit_multi_ref() {
    let src = r#"
model M {
  input x: tensor<f32, [4, 64]>
  layer h1 Dense(units=64)
  layer skip Add(h1, x)
  output skip
}
"#;
    let graph = parse_graph(src);
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let x = graph.node_by_name("x").expect("x");
    let skip = graph.node_by_name("skip").expect("skip");

    let x_ty = shapes.get(&x.id()).unwrap();
    let skip_ty = shapes.get(&skip.id()).unwrap();

    // Add shape = first input's shape
    assert_eq!(skip_ty, x_ty, "Add output should equal first input shape");
}

// ---------------------------------------------------------------------------
// 3. Add shape inference — same as input shape
// ---------------------------------------------------------------------------

#[test]
fn test_add_shape_inferred() {
    let src = r#"
model M {
  input a: tensor<f32, [2, 8]>
  input b: tensor<f32, [2, 8]>
  layer sum Add(a, b)
  output sum
}
"#;
    let graph = parse_graph(src);
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let expected = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(2), Dim::Literal(8)]),
    };

    let sum = graph.node_by_name("sum").expect("sum");
    assert_eq!(shapes.get(&sum.id()).unwrap(), &expected);
}

// ---------------------------------------------------------------------------
// 4. Concat shape inference — sums the concat axis
// ---------------------------------------------------------------------------

#[test]
fn test_concat_shape_inferred() {
    let src = r#"
model M {
  input a: tensor<f32, [2, 32]>
  input b: tensor<f32, [2, 16]>
  layer cat Concat(a, b, axis=1)
  output cat
}
"#;
    let graph = parse_graph(src);
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let expected = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(2), Dim::Literal(48)]),
    };

    let cat = graph.node_by_name("cat").expect("cat");
    assert_eq!(shapes.get(&cat.id()).unwrap(), &expected, "Concat should sum axis=1 dims");
}

// ---------------------------------------------------------------------------
// 5. DeadNodePass removes unreachable nodes
// ---------------------------------------------------------------------------

#[test]
fn test_dead_node_elim() {
    // "dead" is never referenced by any output — it should be removed.
    let src = r#"
model M {
  input x: tensor<f32, [1, 10]>
  layer live Softmax
  layer dead ReLU(x)
  output live
}
"#;
    let mut graph = parse_graph(src);
    let node_count_before = graph.nodes().len();

    DeadNodePass.run(&mut graph).expect("DeadNodePass");

    let node_count_after = graph.nodes().len();
    assert!(
        node_count_after < node_count_before,
        "DeadNodePass should reduce node count: {} -> {}",
        node_count_before,
        node_count_after
    );
    assert!(
        graph.node_by_name("dead").is_none(),
        "dead node should be removed"
    );
    assert!(
        graph.node_by_name("live").is_some(),
        "live node must remain"
    );
}

// ---------------------------------------------------------------------------
// 6. DeadNodePass via GraphPassManager
// ---------------------------------------------------------------------------

#[test]
fn test_dead_node_pass_manager() {
    let src = r#"
model M {
  input x: tensor<f32, [1, 10]>
  layer used Softmax
  layer unused ReLU(x)
  output used
}
"#;
    let mut graph = parse_graph(src);

    let mut gpm = GraphPassManager::new();
    gpm.add_pass(DeadNodePass);
    gpm.run(&mut graph).expect("GraphPassManager::run");

    assert!(graph.node_by_name("unused").is_none(), "unused node removed");
    assert!(graph.node_by_name("used").is_some(), "used node present");
}

// ---------------------------------------------------------------------------
// 7. ONNX emit contains required structural elements
// ---------------------------------------------------------------------------

const NET_SRC: &str = r#"
model Net {
  input x: tensor<f32, [1, 784]>
  layer h1 Dense(units=128)
  layer out Softmax
  output out
}
"#;

#[test]
fn test_emit_onnx_basic() {
    let output = compile(NET_SRC, "net_module", EmitKind::Onnx).expect("compile onnx");

    assert!(output.contains("ir_version: 7"), "missing ir_version\n{}", output);
    assert!(output.contains("\"Net\""), "missing graph name\n{}", output);
    assert!(output.contains("Gemm"), "Dense should map to Gemm\n{}", output);
    assert!(output.contains("Softmax"), "Softmax should appear\n{}", output);
    assert!(output.contains("input"), "missing input declaration\n{}", output);
    assert!(output.contains("output"), "missing output declaration\n{}", output);
}

// ---------------------------------------------------------------------------
// 8. Mixed explicit refs and keyword params parse and lower correctly
// ---------------------------------------------------------------------------

#[test]
fn test_mixed_refs_and_params() {
    let src = r#"
model M {
  input x: tensor<f32, [4, 256]>
  layer h Dense(x, units=64)
  output h
}
"#;
    let graph = parse_graph(src);
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let expected = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(4), Dim::Literal(64)]),
    };

    let h = graph.node_by_name("h").expect("h node");
    assert_eq!(
        shapes.get(&h.id()).unwrap(),
        &expected,
        "Dense(x, units=64) output shape"
    );
}
