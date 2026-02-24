//! Tests for the model DSL: parsing, graph lowering, graph text emission,
//! and error cases.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// Helper source strings
// ---------------------------------------------------------------------------

const SIMPLE_MODEL: &str = r#"
model Net {
  input x: tensor<f32, [1, 784]>
  layer h1 Dense(units=128)
  layer out Softmax
  output out
}
"#;

const NO_PARAM_LAYER: &str = r#"
model Simple {
  input x: tensor<f32, [10]>
  layer relu ReLU
  output relu
}
"#;

const FN_AND_MODEL: &str = r#"
def add(x: f32, y: f32) -> f32 {
    x + y
}

model Net {
  input x: tensor<f32, [4]>
  layer out Linear(units=2)
  output out
}
"#;

// ---------------------------------------------------------------------------
// 1. Parse simple sequential model — field counts correct
// ---------------------------------------------------------------------------

#[test]
fn test_parse_simple_sequential_model() {
    use iris::parser::lexer::Lexer;
    use iris::parser::parse::Parser;

    let tokens = Lexer::new(SIMPLE_MODEL).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");

    assert_eq!(ast.models.len(), 1);
    let model = &ast.models[0];
    assert_eq!(model.name.name, "Net");
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.layers.len(), 2);
    assert_eq!(model.outputs.len(), 1);

    assert_eq!(model.inputs[0].name.name, "x");
    assert_eq!(model.layers[0].name.name, "h1");
    assert_eq!(model.layers[0].op.name, "Dense");
    assert_eq!(model.layers[0].params.len(), 1);
    assert_eq!(model.layers[1].name.name, "out");
    assert_eq!(model.layers[1].op.name, "Softmax");
    assert_eq!(model.layers[1].params.len(), 0);
    assert_eq!(model.outputs[0].name.name, "out");
}

// ---------------------------------------------------------------------------
// 2. Parse model with int, float, bool, and string params
// ---------------------------------------------------------------------------

#[test]
fn test_parse_model_with_int_float_bool_string_params() {
    use iris::parser::ast::AstExpr;
    use iris::parser::lexer::Lexer;
    use iris::parser::parse::Parser;

    let src = r#"
model ParamTest {
  input x: tensor<f32, [1]>
  layer l1 Op(a=42, b=3.14, c=true, d="hello")
  output l1
}
"#;
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");

    let params = &ast.models[0].layers[0].params;
    assert_eq!(params.len(), 4);
    assert_eq!(params[0].key.name, "a");
    assert!(matches!(params[0].value, AstExpr::IntLit { value: 42, .. }));
    assert_eq!(params[1].key.name, "b");
    assert!(matches!(params[1].value, AstExpr::FloatLit { .. }));
    assert_eq!(params[2].key.name, "c");
    assert!(matches!(params[2].value, AstExpr::BoolLit { value: true, .. }));
    assert_eq!(params[3].key.name, "d");
    assert!(matches!(params[3].value, AstExpr::StringLit { .. }));
}

// ---------------------------------------------------------------------------
// 3. Layer with no params (no parentheses)
// ---------------------------------------------------------------------------

#[test]
fn test_parse_layer_no_params() {
    use iris::parser::lexer::Lexer;
    use iris::parser::parse::Parser;

    let tokens = Lexer::new(NO_PARAM_LAYER).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");

    let layer = &ast.models[0].layers[0];
    assert_eq!(layer.name.name, "relu");
    assert_eq!(layer.op.name, "ReLU");
    assert!(layer.params.is_empty());
}

// ---------------------------------------------------------------------------
// 4. File with both `fn` and `model` top-level definitions
// ---------------------------------------------------------------------------

#[test]
fn test_fn_and_model_coexist() {
    use iris::parser::lexer::Lexer;
    use iris::parser::parse::Parser;

    let tokens = Lexer::new(FN_AND_MODEL).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");

    assert_eq!(ast.functions.len(), 1);
    assert_eq!(ast.functions[0].name.name, "add");
    assert_eq!(ast.models.len(), 1);
    assert_eq!(ast.models[0].name.name, "Net");
}

// ---------------------------------------------------------------------------
// 5. End-to-end: compile with EmitKind::Graph produces expected tokens
// ---------------------------------------------------------------------------

#[test]
fn test_emit_graph_text() {
    let output = compile(SIMPLE_MODEL, "net_module", EmitKind::Graph)
        .expect("graph emit should succeed");

    assert!(output.contains("model Net"), "output:\n{}", output);
    assert!(output.contains("input x"), "output:\n{}", output);
    assert!(output.contains("layer h1"), "output:\n{}", output);
    assert!(output.contains("Dense"), "output:\n{}", output);
    assert!(output.contains("layer out"), "output:\n{}", output);
    assert!(output.contains("Softmax"), "output:\n{}", output);
    assert!(output.contains("output out"), "output:\n{}", output);
}

// ---------------------------------------------------------------------------
// 6. LowerError::UndefinedLayer on bad output name
// ---------------------------------------------------------------------------

#[test]
fn test_error_undefined_output() {
    let src = r#"
model Bad {
  input x: tensor<f32, [1]>
  layer h Dense
  output nonexistent
}
"#;
    let result = compile(src, "bad", EmitKind::Graph);
    assert!(result.is_err(), "should fail on undefined output");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("undefined") || msg.contains("lowering"),
        "error was: {}",
        msg
    );
}

// ---------------------------------------------------------------------------
// 7. LowerError::DuplicateNode on duplicate layer name
// ---------------------------------------------------------------------------

#[test]
fn test_error_duplicate_layer_name() {
    let src = r#"
model Bad {
  input x: tensor<f32, [1]>
  layer h Dense
  layer h ReLU
  output h
}
"#;
    let result = compile(src, "bad", EmitKind::Graph);
    assert!(result.is_err(), "should fail on duplicate layer name");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("duplicate") || msg.contains("lowering"),
        "error was: {}",
        msg
    );
}

// ---------------------------------------------------------------------------
// 8. LowerError::InvalidLayerParam on expression (non-literal) param
// ---------------------------------------------------------------------------

#[test]
fn test_error_layer_param_not_literal() {
    let src = r#"
model Bad {
  input x: tensor<f32, [1]>
  layer h Dense(units=a)
  output h
}
"#;
    let result = compile(src, "bad", EmitKind::Graph);
    assert!(result.is_err(), "should fail on non-literal param");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("literal") || msg.contains("lowering") || msg.contains("hyperparameter"),
        "error was: {}",
        msg
    );
}
