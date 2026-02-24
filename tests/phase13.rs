//! Phase 13 integration tests: struct types and named records.

use iris::interp::{eval_function, IrValue};
use iris::ir::instr::{IrInstr};
use iris::ir::module::IrFunctionBuilder;
use iris::ir::types::{DType, IrType};
use iris::{compile, EmitKind};

fn f32_ty() -> IrType {
    IrType::Scalar(DType::F32)
}
fn i64_ty() -> IrType {
    IrType::Scalar(DType::I64)
}

// ---------------------------------------------------------------------------
// 1. Lexer: `struct` keyword and `.` token are recognized
// ---------------------------------------------------------------------------
#[test]
fn test_struct_keyword_lexed() {
    // A minimal struct definition should parse without error.
    let src = "record Empty {}";
    // Compilation will fail at lowering (no functions), but parsing must succeed.
    // Use a valid program to verify the lexer:
    let src2 = "record Point { x: f32, y: f32 } def zero() -> f32 { 0.0 }";
    let result = compile(src2, "test", EmitKind::Ir);
    assert!(result.is_ok(), "struct keyword should be recognized: {:?}", result.err());
    let _ = src; // just to use it
}

// ---------------------------------------------------------------------------
// 2. Parser: struct definition with multiple fields
// ---------------------------------------------------------------------------
#[test]
fn test_struct_definition_parsed() {
    let src = "record Vec2 { x: f32, y: f32 }  def zero() -> f32 { 0.0 }";
    let out = compile(src, "test", EmitKind::Ir).expect("should compile");
    // IR output won't mention struct defs directly, but no error means parsing OK
    assert!(out.contains("def zero"));
}

// ---------------------------------------------------------------------------
// 3. Parser: struct literal expression
// ---------------------------------------------------------------------------
#[test]
fn test_struct_literal_compiles() {
    let src = r#"
record Point { x: f32, y: f32 }
def make_point() -> Point {
    Point { x: 1.0, y: 2.0 }
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "struct literal should compile: {:?}", result.err());
    let out = result.unwrap();
    assert!(out.contains("make_struct"), "expected make_struct instruction");
}

// ---------------------------------------------------------------------------
// 4. Parser: field access expression
// ---------------------------------------------------------------------------
#[test]
fn test_field_access_compiles() {
    let src = r#"
record Point { x: f32, y: f32 }
def get_x(p: Point) -> f32 {
    p.x
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "field access should compile: {:?}", result.err());
    let out = result.unwrap();
    assert!(out.contains("get_field"), "expected get_field instruction");
}

// ---------------------------------------------------------------------------
// 5. IR printer: MakeStruct and GetField emitted correctly
// ---------------------------------------------------------------------------
#[test]
fn test_struct_ir_text() {
    let src = r#"
record Pair { a: i64, b: i64 }
def make(x: i64, y: i64) -> Pair {
    Pair { a: x, b: y }
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(out.contains("make_struct"), "expected make_struct in IR");
    assert!(out.contains("%Pair"), "expected struct type name in IR");
}

// ---------------------------------------------------------------------------
// 6. Interpreter: MakeStruct builds a Struct value
// ---------------------------------------------------------------------------
#[test]
fn test_interp_make_struct() {
    // Build: fn pair() -> struct { 10, 20 }
    let struct_ty = IrType::Struct {
        name: "Pair".into(),
        fields: vec![
            ("a".into(), i64_ty()),
            ("b".into(), i64_ty()),
        ],
    };
    let mut b = IrFunctionBuilder::new("pair", vec![], struct_ty.clone());
    let entry = b.create_block(Some("entry"));
    b.set_current_block(entry);

    let a = b.fresh_value();
    b.push_instr(
        IrInstr::ConstInt { result: a, value: 10, ty: i64_ty() },
        Some(i64_ty()),
    );
    let bv = b.fresh_value();
    b.push_instr(
        IrInstr::ConstInt { result: bv, value: 20, ty: i64_ty() },
        Some(i64_ty()),
    );
    let s = b.fresh_value();
    b.push_instr(
        IrInstr::MakeStruct { result: s, fields: vec![a, bv], result_ty: struct_ty.clone() },
        Some(struct_ty),
    );
    b.push_instr(IrInstr::Return { values: vec![s] }, None);
    let func = b.build();

    let result = eval_function(&func, &[]).expect("should eval");
    assert_eq!(result, vec![IrValue::Struct(vec![IrValue::I64(10), IrValue::I64(20)])]);
}

// ---------------------------------------------------------------------------
// 7. Interpreter: GetField extracts the correct value
// ---------------------------------------------------------------------------
#[test]
fn test_interp_get_field() {
    let struct_ty = IrType::Struct {
        name: "Pair".into(),
        fields: vec![
            ("a".into(), f32_ty()),
            ("b".into(), f32_ty()),
        ],
    };

    let mut b = IrFunctionBuilder::new("get_b", vec![], f32_ty());
    let entry = b.create_block(Some("entry"));
    b.set_current_block(entry);

    let fa = b.fresh_value();
    b.push_instr(
        IrInstr::ConstFloat { result: fa, value: 1.0, ty: f32_ty() },
        Some(f32_ty()),
    );
    let fb = b.fresh_value();
    b.push_instr(
        IrInstr::ConstFloat { result: fb, value: 99.0, ty: f32_ty() },
        Some(f32_ty()),
    );
    let s = b.fresh_value();
    b.push_instr(
        IrInstr::MakeStruct { result: s, fields: vec![fa, fb], result_ty: struct_ty.clone() },
        Some(struct_ty),
    );
    let field_val = b.fresh_value();
    b.push_instr(
        IrInstr::GetField { result: field_val, base: s, field_index: 1, result_ty: f32_ty() },
        Some(f32_ty()),
    );
    b.push_instr(IrInstr::Return { values: vec![field_val] }, None);
    let func = b.build();

    let result = eval_function(&func, &[]).expect("should eval");
    assert_eq!(result, vec![IrValue::F32(99.0)]);
}

// ---------------------------------------------------------------------------
// 8. End-to-end: struct roundtrip through compile() + field access
// ---------------------------------------------------------------------------
#[test]
fn test_struct_e2e_field_access() {
    let src = r#"
record Pt { x: f32, y: f32 }
def get_y() -> f32 {
    val p = Pt { x: 3.0, y: 7.0 }
    p.y
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "7", "expected y=7");
}
