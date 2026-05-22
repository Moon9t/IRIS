//! Phase 124 integration tests: mutable variables and assignment patterns.

use iris::{compile, EmitKind};

// ── 1. Basic var mutation ───────────────────────────────────────────────────
#[test]
fn test_var_mutation() {
    let src = r#"
def f() -> i64 {
    var x = 10
    x = 20
    x
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "20");
}

// ── 2. Multiple mutations ───────────────────────────────────────────────────
#[test]
fn test_var_multiple_mutations() {
    let src = r#"
def f() -> i64 {
    var x = 0
    x = x + 1
    x = x + 2
    x = x + 3
    x
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ── 3. Var in loop ─────────────────────────────────────────────────────────
#[test]
fn test_var_in_loop() {
    let src = r#"
def f() -> i64 {
    var sum = 0
    for i in 1..11 {
        sum = sum + i
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "55");
}

// ── 4. Multiple vars ───────────────────────────────────────────────────────
#[test]
fn test_multiple_vars() {
    let src = r#"
def f() -> i64 {
    var a = 1
    var b = 2
    var c = 3
    a = a + b
    b = b + c
    c = a + b
    c
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // a=3, b=5, c=8
    assert_eq!(result.trim(), "8");
}

// ── 5. Var string ──────────────────────────────────────────────────────────
#[test]
fn test_var_string() {
    let src = r#"
def f() -> str {
    var s = "hello"
    s = concat(s, " world")
    s
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello world");
}

// ── 6. Var boolean toggle ──────────────────────────────────────────────────
#[test]
fn test_var_bool_toggle() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    var flag = true
    flag = false
    bool_to_i64(flag)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ── 7. Swap pattern ────────────────────────────────────────────────────────
#[test]
fn test_var_swap() {
    let src = r#"
def f() -> i64 {
    var a = 10
    var b = 20
    val tmp = a
    a = b
    b = tmp
    a * 100 + b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2010");
}

// ── 8. Accumulator pattern ─────────────────────────────────────────────────
#[test]
fn test_var_accumulator() {
    let src = r#"
def f() -> i64 {
    var product = 1
    for i in 1..6 {
        product = product * i
    }
    product
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 1*2*3*4*5 = 120
    assert_eq!(result.trim(), "120");
}
