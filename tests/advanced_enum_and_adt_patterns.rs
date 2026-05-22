//! Phase 117 integration tests: advanced enum and ADT patterns.

use iris::{compile, EmitKind};

// ── 1. Basic choice with pattern matching ───────────────────────────────────
#[test]
fn test_choice_basic_match() {
    let src = r#"
choice Color { Red, Green, Blue }
def to_num(c: Color) -> i64 {
    when c { Color.Red => 1, Color.Green => 2, Color.Blue => 3 }
}
def f() -> i64 {
    to_num(Color.Green)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 2. Choice with data ────────────────────────────────────────────────────
#[test]
fn test_choice_with_data() {
    let src = r#"
choice Expr {
    Lit(i64),
    Neg(i64),
}
def eval_expr(e: Expr) -> i64 {
    when e {
        Expr.Lit(n) => n,
        Expr.Neg(n) => 0 - n,
    }
}
def f() -> i64 {
    eval_expr(Expr.Lit(42)) + eval_expr(Expr.Neg(8))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "34");
}

// ── 3. Choice used in condition ─────────────────────────────────────────────
#[test]
fn test_choice_in_condition() {
    let src = r#"
choice Dir { Up, Down, Left, Right }
def is_vertical(d: Dir) -> bool {
    when d { Dir.Up => true, Dir.Down => true, Dir.Left => false, Dir.Right => false }
}
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    bool_to_i64(is_vertical(Dir.Up)) + bool_to_i64(is_vertical(Dir.Left))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 4. Choice with wildcard pattern ─────────────────────────────────────────
#[test]
fn test_choice_wildcard() {
    let src = r#"
choice Fruit { Apple, Banana, Cherry, Durian }
def is_apple(fr: Fruit) -> i64 {
    when fr { Fruit.Apple => 1, _ => 0 }
}
def f() -> i64 {
    is_apple(Fruit.Apple) + is_apple(Fruit.Cherry)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 5. Choice values computed ──────────────────────────────────────────────
#[test]
fn test_choice_values_in_list() {
    let src = r#"
choice Op { Add, Sub, Mul }
def op_val(o: Op) -> i64 {
    when o { Op.Add => 1, Op.Sub => 2, Op.Mul => 3 }
}
def f() -> i64 {
    op_val(Op.Add) + op_val(Op.Sub) + op_val(Op.Mul)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ── 6. when on integer with guard-like patterns ─────────────────────────────
#[test]
fn test_choice_guard() {
    let src = r#"
def classify(x: i64) -> str {
    if x == 0 { "zero" } else { if x > 0 { "positive" } else { "negative" } }
}
def f() -> str {
    classify(5)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "positive");
}

// ── 7. when on integer with guard negative ──────────────────────────────────
#[test]
fn test_choice_guard_negative() {
    let src = r#"
def classify2(x: i64) -> str {
    if x == 0 { "zero" } else { if x > 0 { "positive" } else { "negative" } }
}
def f() -> str {
    classify2(0 - 3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "negative");
}

// ── 8. Choice function called with different variants ───────────────────────
#[test]
fn test_choice_multi_variant() {
    let src = r#"
choice Token { Num(i64), Plus, Minus }
def tok_val(t: Token) -> i64 {
    when t {
        Token.Num(n) => n,
        Token.Plus => 0 - 1,
        Token.Minus => 0 - 2,
    }
}
def f() -> i64 {
    tok_val(Token.Num(100)) + tok_val(Token.Plus) + tok_val(Token.Minus)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "97");
}
