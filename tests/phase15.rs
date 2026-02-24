//! Phase 15 integration tests: enum types and pattern matching.

use iris::interp::{eval_function, IrValue};
use iris::ir::instr::IrInstr;
use iris::ir::module::IrFunctionBuilder;
use iris::ir::types::{DType, IrType};
use iris::{compile, EmitKind};

fn i64_ty() -> IrType {
    IrType::Scalar(DType::I64)
}

// ---------------------------------------------------------------------------
// 1. Lexer/parser: `choice` keyword is recognized
// ---------------------------------------------------------------------------
#[test]
fn test_choice_keyword_lexed() {
    let src = r#"
choice Color { Red, Green, Blue }
def zero() -> i64 { 0 }
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(
        result.is_ok(),
        "choice keyword should be recognized: {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// 2. Parser: enum definition with multiple variants
// ---------------------------------------------------------------------------
#[test]
fn test_choice_variants_parsed() {
    let src = r#"
choice Direction { North, South, East, West }
def go() -> i64 { 1 }
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(out.contains("def go"), "IR should contain the function");
}

// ---------------------------------------------------------------------------
// 3. Enum variant construction: `Color.Red` lowers to make_variant
// ---------------------------------------------------------------------------
#[test]
fn test_enum_variant_construction_ir() {
    let src = r#"
choice Color { Red, Green, Blue }
def get_red() -> Color {
    Color.Red
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(
        out.contains("make_variant"),
        "IR should contain make_variant: {}",
        out
    );
}

// ---------------------------------------------------------------------------
// 4. `when` expression compiles
// ---------------------------------------------------------------------------
#[test]
fn test_when_expression_compiles() {
    let src = r#"
choice Color { Red, Green, Blue }
def color_code(c: Color) -> i64 {
    when c {
        Color.Red => 0
        Color.Green => 1
        Color.Blue => 2
    }
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(
        result.is_ok(),
        "when expression should compile: {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// 5. IR contains switch_variant and make_variant
// ---------------------------------------------------------------------------
#[test]
fn test_when_ir_has_switch_variant() {
    let src = r#"
choice Dir { Up, Down }
def pick(d: Dir) -> i64 {
    when d {
        Dir.Up => 10
        Dir.Down => 20
    }
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(
        out.contains("switch_variant"),
        "IR should contain switch_variant: {}",
        out
    );
}

// ---------------------------------------------------------------------------
// 6. Interpreter: MakeVariant produces IrValue::Enum(idx)
// ---------------------------------------------------------------------------
#[test]
fn test_interp_make_variant() {
    let enum_ty = IrType::Enum {
        name: "Color".into(),
        variants: vec!["Red".into(), "Green".into(), "Blue".into()],
    };
    let mut b = IrFunctionBuilder::new("get_green", vec![], enum_ty.clone());
    let entry = b.create_block(Some("entry"));
    b.set_current_block(entry);

    let v = b.fresh_value();
    b.push_instr(
        IrInstr::MakeVariant {
            result: v,
            variant_idx: 1,
            result_ty: enum_ty,
        },
        Some(i64_ty()),
    );
    b.push_instr(IrInstr::Return { values: vec![v] }, None);
    let func = b.build();

    let result = eval_function(&func, &[]).expect("should eval");
    assert_eq!(result, vec![IrValue::Enum(1)]);
}

// ---------------------------------------------------------------------------
// 7. Interpreter: SwitchVariant dispatches to the correct arm
// ---------------------------------------------------------------------------
#[test]
fn test_interp_switch_variant() {
    let enum_ty = IrType::Enum {
        name: "Dir".into(),
        variants: vec!["Up".into(), "Down".into()],
    };
    let mut b = IrFunctionBuilder::new("pick", vec![], i64_ty());
    let entry = b.create_block(Some("entry"));
    let up_bb = b.create_block(Some("arm_up"));
    let down_bb = b.create_block(Some("arm_down"));
    let merge_bb = b.create_block(Some("merge"));
    let merge_param = b.add_block_param(merge_bb, Some("result"), i64_ty());

    b.set_current_block(entry);
    let tag = b.fresh_value();
    b.push_instr(
        IrInstr::MakeVariant {
            result: tag,
            variant_idx: 0,
            result_ty: enum_ty,
        },
        Some(i64_ty()),
    );
    b.push_instr(
        IrInstr::SwitchVariant {
            scrutinee: tag,
            arms: vec![(0, up_bb), (1, down_bb)],
            default_block: None,
        },
        None,
    );

    b.set_current_block(up_bb);
    let ten = b.fresh_value();
    b.push_instr(
        IrInstr::ConstInt {
            result: ten,
            value: 10,
            ty: i64_ty(),
        },
        Some(i64_ty()),
    );
    b.push_instr(
        IrInstr::Br {
            target: merge_bb,
            args: vec![ten],
        },
        None,
    );

    b.set_current_block(down_bb);
    let twenty = b.fresh_value();
    b.push_instr(
        IrInstr::ConstInt {
            result: twenty,
            value: 20,
            ty: i64_ty(),
        },
        Some(i64_ty()),
    );
    b.push_instr(
        IrInstr::Br {
            target: merge_bb,
            args: vec![twenty],
        },
        None,
    );

    b.set_current_block(merge_bb);
    b.push_instr(
        IrInstr::Return {
            values: vec![merge_param],
        },
        None,
    );

    let func = b.build();
    let result = eval_function(&func, &[]).expect("should eval");
    assert_eq!(result, vec![IrValue::I64(10)]);
}

// ---------------------------------------------------------------------------
// 8. End-to-end: enum + when through compile() + EmitKind::Eval
// ---------------------------------------------------------------------------
#[test]
fn test_enum_e2e_eval() {
    let src = r#"
choice Season { Spring, Summer, Autumn, Winter }
def temp() -> i64 {
    val s = Season.Summer
    when s {
        Season.Spring => 15
        Season.Summer => 30
        Season.Autumn => 10
        Season.Winter => 0
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "30", "Summer should map to 30");
}
