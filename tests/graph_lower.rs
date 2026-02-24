//! Integration tests for Phase 3: shape inference + GraphIr → IrModule lowering.
//!
//! All tests go through the parser + lower_model() path, keeping integration
//! tests decoupled from `pub(crate)` GraphIr internals.

use iris::ir::types::{DType, Dim, IrType, Shape};
use iris::lower::lower_model;
use iris::parser::lexer::Lexer;
use iris::parser::parse::Parser;
use iris::pass::infer_shapes;
use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// Shared source strings
// ---------------------------------------------------------------------------

const NET_SRC: &str = r#"
model Net {
  input x: tensor<f32, [1, 784]>
  layer h1 Dense(units=128)
  layer out Softmax
  output out
}
"#;

const CHAIN_SRC: &str = r#"
model Chain {
  input x: tensor<f32, [4, 256]>
  layer d1 Dense(units=64)
  layer r1 ReLU
  layer sm Softmax
  output sm
}
"#;

/// Parse and lower a model source string to a GraphIr.
fn parse_graph(src: &str) -> iris::ir::graph::GraphIr {
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");
    assert_eq!(ast.models.len(), 1, "expected exactly one model");
    lower_model(&ast.models[0]).expect("lower_model")
}

// ---------------------------------------------------------------------------
// 1. Dense: [1, 784] → [1, 128]
// ---------------------------------------------------------------------------

#[test]
fn test_infer_shapes_dense() {
    let graph = parse_graph(NET_SRC);
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let dense_node = graph.node_by_name("h1").expect("h1 node");
    let dense_ty = shapes.get(&dense_node.id()).expect("Dense type");

    assert_eq!(
        dense_ty,
        &IrType::Tensor {
            dtype: DType::F32,
            shape: Shape(vec![Dim::Literal(1), Dim::Literal(128)]),
        },
        "Dense(units=128) should replace last dim: got {:?}",
        dense_ty
    );
}

// ---------------------------------------------------------------------------
// 2. Softmax: same shape as predecessor (passthrough)
// ---------------------------------------------------------------------------

#[test]
fn test_infer_shapes_softmax() {
    let src = r#"
model S {
  input x: tensor<f32, [1, 10]>
  layer sm Softmax
  output sm
}
"#;
    let graph = parse_graph(src);
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let input_node = graph.node_by_name("x").expect("input x");
    let sm_node = graph.node_by_name("sm").expect("sm node");

    let input_ty = shapes.get(&input_node.id()).expect("input type");
    let sm_ty = shapes.get(&sm_node.id()).expect("softmax type");

    assert_eq!(sm_ty, input_ty, "Softmax should pass through shape unchanged");
}

// ---------------------------------------------------------------------------
// 3. Chain: Dense → ReLU → Softmax — all shapes propagate correctly
// ---------------------------------------------------------------------------

#[test]
fn test_infer_shapes_chain() {
    let graph = parse_graph(CHAIN_SRC);
    let shapes = infer_shapes(&graph).expect("infer_shapes");

    let expected = IrType::Tensor {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Literal(4), Dim::Literal(64)]),
    };

    let d1 = graph.node_by_name("d1").expect("d1");
    let r1 = graph.node_by_name("r1").expect("r1");
    let sm = graph.node_by_name("sm").expect("sm");

    assert_eq!(shapes.get(&d1.id()).unwrap(), &expected, "Dense output shape");
    assert_eq!(shapes.get(&r1.id()).unwrap(), &expected, "ReLU passthrough shape");
    assert_eq!(shapes.get(&sm.id()).unwrap(), &expected, "Softmax passthrough shape");
}

// ---------------------------------------------------------------------------
// 4. lower_graph_to_ir: function has correct name and param types
// ---------------------------------------------------------------------------

#[test]
fn test_lower_graph_to_ir_params() {
    use iris::lower::lower_graph_to_ir;

    let graph = parse_graph(NET_SRC);
    let shapes = infer_shapes(&graph).unwrap();
    let func = lower_graph_to_ir(&graph, &shapes).expect("lower_graph_to_ir");

    assert_eq!(func.name, "Net");
    assert_eq!(func.params.len(), 1);
    assert_eq!(func.params[0].name, "x");
    assert_eq!(
        func.params[0].ty,
        IrType::Tensor {
            dtype: DType::F32,
            shape: Shape(vec![Dim::Literal(1), Dim::Literal(784)]),
        }
    );
    // Return type should be the Softmax output shape = [1, 128]
    assert_eq!(
        func.return_ty,
        IrType::Tensor {
            dtype: DType::F32,
            shape: Shape(vec![Dim::Literal(1), Dim::Literal(128)]),
        }
    );
}

// ---------------------------------------------------------------------------
// 5. lower_graph_to_ir: each layer becomes a Call instruction in IR text
// ---------------------------------------------------------------------------

#[test]
fn test_lower_graph_to_ir_has_call_instrs() {
    use iris::codegen::emit_ir_text;
    use iris::ir::module::IrModule;
    use iris::lower::lower_graph_to_ir;

    let graph = parse_graph(NET_SRC);
    let shapes = infer_shapes(&graph).unwrap();
    let func = lower_graph_to_ir(&graph, &shapes).unwrap();

    let mut module = IrModule::new("test");
    module.add_function(func).unwrap();
    let ir = emit_ir_text(&module).unwrap();

    assert!(ir.contains("call @Dense"), "IR should contain call @Dense\n{}", ir);
    assert!(ir.contains("call @Softmax"), "IR should contain call @Softmax\n{}", ir);
    assert!(ir.contains("return"), "IR should contain return\n{}", ir);
}

// ---------------------------------------------------------------------------
// 6. End-to-end: compile model with EmitKind::Ir
// ---------------------------------------------------------------------------

#[test]
fn test_compile_model_emit_ir() {
    let output = compile(NET_SRC, "net_module", EmitKind::Ir).expect("compile should succeed");

    assert!(output.contains("def Net"), "output:\n{}", output);
    assert!(output.contains("call @Dense"), "output:\n{}", output);
    assert!(output.contains("call @Softmax"), "output:\n{}", output);
    assert!(output.contains("return"), "output:\n{}", output);
}

// ---------------------------------------------------------------------------
// 7. End-to-end: compile model with EmitKind::Llvm
// ---------------------------------------------------------------------------

#[test]
fn test_compile_model_emit_llvm() {
    let output = compile(NET_SRC, "net_module", EmitKind::Llvm).expect("llvm stub should succeed");

    assert!(output.contains("define"), "output:\n{}", output);
    assert!(output.contains("@Net"), "output:\n{}", output);
    assert!(output.contains("ptr"), "output:\n{}", output);
}

// ---------------------------------------------------------------------------
// 8. Unknown op returns an error
// ---------------------------------------------------------------------------

#[test]
fn test_unknown_op_returns_error() {
    let src = r#"
model Bad {
  input x: tensor<f32, [1, 10]>
  layer h UnknownOp
  output h
}
"#;
    let result = compile(src, "bad", EmitKind::Ir);
    assert!(result.is_err(), "unknown op should fail");
    let msg = result.unwrap_err().to_string();
    // The error message comes from LowerError::UnknownOp
    assert!(
        msg.contains("UnknownOp") || msg.contains("shape inference") || msg.contains("lowering"),
        "error was: {}",
        msg
    );
}
