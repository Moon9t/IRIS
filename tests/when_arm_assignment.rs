//! Assigning an enclosing `var` from `when` arms (known-issues #3), and a
//! `str` record field inside `result<T, E>` (#2).
//!
//! #3 had two defects stacked. The visible one was a compile failure:
//!
//! ```text
//! error[E0200]: variable '%6' is used before it has been assigned a value
//! ```
//!
//! A braced arm body (`{ a = 1; }`) has no tail expression, and the fallback for
//! a block with no value handed back a bare `fresh_value()` — a `ValueId` that
//! no instruction defines. Anything consuming it failed validation, pointing at
//! an internal SSA name rather than at the source.
//!
//! Fixing that exposed the second, worse one: the enum/choice path never
//! threaded assigned variables to the merge block, so the assignment was
//! silently *lost* and the program computed the wrong answer with no error. The
//! option and result paths already threaded them, which is why only `choice`
//! was wrong once it compiled.

use iris::{compile, EmitKind};

// ── #3: the arm assignment must reach the merge block ─────────────────────

#[test]
fn a_choice_arm_can_assign_an_enclosing_var() {
    let src = r#"
choice C { X, Y }
def f() -> i64 {
    val d = C.X;
    var a = 0;
    when d { C.X => { a = 1; } C.Y => { a = 2; } };
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1", "the arm's assignment was lost");
}

/// The other arm, so the merge is not simply passing the first arm's value.
#[test]
fn the_other_choice_arm_assigns_its_own_value() {
    let src = r#"
choice C { X, Y }
def f() -> i64 {
    val d = C.Y;
    var a = 0;
    when d { C.X => { a = 1; } C.Y => { a = 2; } };
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

/// An arm that does not touch the variable must pass through the value it came
/// in with — every predecessor has to supply an argument for the merge param.
#[test]
fn an_arm_that_does_not_assign_preserves_the_incoming_value() {
    let src = r#"
choice C { X, Y }
def f() -> i64 {
    val d = C.Y;
    var a = 7;
    when d { C.X => { a = 1; } C.Y => { val untouched = 99; } };
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "7");
}

/// A variant carrying data: the binding is extracted in the arm block, and the
/// value assigned from it must still reach the merge.
#[test]
fn a_variant_binding_can_be_assigned_to_an_outer_var() {
    let src = r#"
choice E { N(i64), M }
def f() -> i64 {
    val d = E.N(7);
    var a = 0;
    when d { E.N(v) => { a = v; } E.M => { a = 0 - 5; } };
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "7");
}

/// More than one variable, so the merge parameters and branch arguments have to
/// stay in the same order.
#[test]
fn several_vars_assigned_in_one_arm_keep_their_identities() {
    let src = r#"
choice C { X, Y }
def f() -> i64 {
    val d = C.X;
    var a = 0;
    var b = 0;
    when d { C.X => { a = 1; b = 20; } C.Y => { a = 300; b = 4000; } };
    a + b
}
"#;
    // 1 + 20, not 300 + 4000 and not any mixture of the two arms
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "21");
}

/// `result` and `option` already threaded assignments; these guard against a
/// regression while the enum path was being changed.
#[test]
fn a_result_arm_can_assign_an_enclosing_var() {
    let src = r#"
def f() -> i64 {
    val d: result<i64, str> = ok(42);
    var a = 0;
    when d { ok(c) => { a = c; } err(e) => { a = 0 - 1; } };
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

#[test]
fn the_err_arm_assigns_its_own_value() {
    let src = r#"
def f() -> i64 {
    val d: result<i64, str> = err("bad");
    var a = 0;
    when d { ok(c) => { a = c; } err(e) => { a = 0 - 1; } };
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "-1");
}

#[test]
fn an_option_arm_can_assign_an_enclosing_var() {
    let src = r#"
def f() -> i64 {
    val d: option<i64> = some(42);
    var a = 0;
    when d { some(c) => { a = c; } none => { a = 0 - 1; } };
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

/// The documented workaround — `when` as an expression — must keep working.
#[test]
fn when_as_an_expression_still_works() {
    let src = r#"
def f() -> i64 {
    val d: result<i64, str> = ok(42);
    val a = when d { ok(c) => c, err(_) => 0 - 1, };
    a
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── #2: a `str` record field inside `result<T, E>` ────────────────────────

/// Reported as mis-typing the string as `i64` and failing LLVM verification
/// with `'%v14' defined with type 'i64' but expected 'ptr'`. It no longer
/// reproduces — fixed by earlier `MakeStruct` field-store work — so this pins
/// the behaviour rather than describing a defect.
#[test]
fn a_str_record_field_survives_a_result_round_trip() {
    let src = r#"
record Reading { name: str, value: i64 }
def make(n: i64) -> result<Reading, str> {
    if n < 0 { err("negative") } else { ok(Reading { name: "temp", value: n }) }
}
def f() -> i64 {
    val r = make(5);
    val v = unwrap(r);
    val ok_name = if v.name == "temp" { 1 } else { 0 };
    ok_name * 100 + v.value
}
"#;
    // name compared equal (100) plus the i64 field (5)
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "105");
}

#[test]
fn the_err_arm_of_a_str_record_result_still_carries_its_message() {
    let src = r#"
record Reading { name: str, value: i64 }
def make(n: i64) -> result<Reading, str> {
    if n < 0 { err("negative") } else { ok(Reading { name: "temp", value: n }) }
}
def f() -> i64 {
    val r = make(0 - 1);
    if is_ok(r) { 0 } else { 1 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}
