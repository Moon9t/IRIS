//! Phase 16 integration tests: mutable bindings (`var` keyword + ident assign).
//!
//! `var x = expr` is syntactically identical to `val x = expr` — both lower to
//! an SSA binding. Plain `x = expr` re-maps `x` in the scope (SSA rebinding).
//! Inside loops, ident-assign targets are tracked as loop variables so the
//! while-header block parameters are threaded correctly.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. `var` keyword is accepted by the lexer/parser
// ---------------------------------------------------------------------------
#[test]
fn test_var_keyword_lexed() {
    let src = r#"
def f() -> i64 {
    var x = 5
    x
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "var keyword should be accepted: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// 2. `var` produces the same IR structure as `val`
// ---------------------------------------------------------------------------
#[test]
fn test_var_same_ir_as_val() {
    let src_var = r#"
def f() -> i64 { var x = 42; x }
"#;
    let src_val = r#"
def f() -> i64 { val x = 42; x }
"#;
    let out_var = compile(src_var, "test", EmitKind::Ir).expect("var should compile");
    let out_val = compile(src_val, "test", EmitKind::Ir).expect("val should compile");
    assert!(out_var.contains("const.i 42"), "var IR missing const: {}", out_var);
    assert!(out_val.contains("const.i 42"), "val IR missing const: {}", out_val);
}

// ---------------------------------------------------------------------------
// 3. Plain ident assignment compiles (x = expr after var x = init)
// ---------------------------------------------------------------------------
#[test]
fn test_var_ident_assign_compiles() {
    let src = r#"
def f() -> i64 {
    var x = 1
    x = 2
    x
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "ident assign should compile: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// 4. Ident assignment rebinds — the updated value is returned
// ---------------------------------------------------------------------------
#[test]
fn test_var_ident_assign_eval() {
    let src = r#"
def f() -> i64 {
    var x = 1
    x = 2
    x
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "2", "rebind should yield 2, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. Multiple sequential rebinds — last assignment wins
// ---------------------------------------------------------------------------
#[test]
fn test_var_rebind_multiple_times_eval() {
    let src = r#"
def f() -> i64 {
    var x = 10
    x = 20
    x = 30
    x
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "30", "last rebind should be 30, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. Mixing `var` and `val` in the same block
// ---------------------------------------------------------------------------
#[test]
fn test_var_mixed_with_val() {
    let src = r#"
def f() -> i64 {
    val a = 5
    var b = 10
    b = a + b
    b
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "15", "5 + 10 = 15, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. var rebinding inside a while loop (single loop variable)
// ---------------------------------------------------------------------------
#[test]
fn test_var_in_while_loop_eval() {
    let src = r#"
def count() -> i64 {
    var i = 0
    while i < 5 {
        i = i + 1
    }
    i
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "5", "counter should reach 5, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. Accumulator loop with two var loop variables
// ---------------------------------------------------------------------------
#[test]
fn test_var_accumulator_loop() {
    let src = r#"
def sum_to_five() -> i64 {
    var acc = 0
    var i = 1
    while i < 6 {
        acc = acc + i
        i = i + 1
    }
    acc
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // 1 + 2 + 3 + 4 + 5 = 15
    assert_eq!(out.trim(), "15", "sum 1..5 = 15, got: {}", out.trim());
}
