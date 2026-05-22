//! Phase 20 integration tests: Logical operators (`&&`, `||`) and early `return`.
//!
//! `&&` / `||` use short-circuit evaluation — the RHS is only evaluated when
//! necessary. `return expr` exits the function immediately with the given value.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. `&&` basic truth table
// ---------------------------------------------------------------------------
#[test]
fn test_and_basic_eval() {
    // true && true → true
    let src = r#"
def f() -> bool {
    true && true
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "true",
        "true && true should be true, got: {}",
        out.trim()
    );

    // true && false → false
    let src2 = r#"
def f() -> bool {
    true && false
}
"#;
    let out2 = compile(src2, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out2.trim(),
        "false",
        "true && false should be false, got: {}",
        out2.trim()
    );

    // false && true → false
    let src3 = r#"
def f() -> bool {
    false && true
}
"#;
    let out3 = compile(src3, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out3.trim(),
        "false",
        "false && true should be false, got: {}",
        out3.trim()
    );
}

// ---------------------------------------------------------------------------
// 2. `||` basic truth table
// ---------------------------------------------------------------------------
#[test]
fn test_or_basic_eval() {
    // false || true → true
    let src = r#"
def f() -> bool {
    false || true
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "true",
        "false || true should be true, got: {}",
        out.trim()
    );

    // false || false → false
    let src2 = r#"
def f() -> bool {
    false || false
}
"#;
    let out2 = compile(src2, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out2.trim(),
        "false",
        "false || false should be false, got: {}",
        out2.trim()
    );

    // true || false → true
    let src3 = r#"
def f() -> bool {
    true || false
}
"#;
    let out3 = compile(src3, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out3.trim(),
        "true",
        "true || false should be true, got: {}",
        out3.trim()
    );
}

// ---------------------------------------------------------------------------
// 3. `&&` with comparison expressions
// ---------------------------------------------------------------------------
#[test]
fn test_and_with_comparisons_eval() {
    let src = r#"
def f() -> bool {
    val x = 5
    val y = 10
    x < y && y < 20
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "true",
        "5<10 && 10<20 should be true, got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 4. `||` with comparison expressions
// ---------------------------------------------------------------------------
#[test]
fn test_or_with_comparisons_eval() {
    let src = r#"
def f() -> bool {
    val x = 5
    x > 100 || x < 10
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "true",
        "5>100 || 5<10 should be true, got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 5. Precedence: `&&` binds tighter than `||`
// ---------------------------------------------------------------------------
#[test]
fn test_precedence_and_or_eval() {
    // `false || true && true` should be `false || (true && true)` = true
    let src = r#"
def f() -> bool {
    false || true && true
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "true",
        "false || (true && true) should be true, got: {}",
        out.trim()
    );

    // `true || false && false` should be `true || (false && false)` = true
    let src2 = r#"
def f() -> bool {
    true || false && false
}
"#;
    let out2 = compile(src2, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out2.trim(),
        "true",
        "true || (false && false) should be true, got: {}",
        out2.trim()
    );
}

// ---------------------------------------------------------------------------
// 6. Early return from function
// ---------------------------------------------------------------------------
#[test]
fn test_early_return_eval() {
    let src = r#"
def f() -> i64 {
    return 42;
    99
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "42",
        "early return should yield 42, got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 7. Conditional early return
// ---------------------------------------------------------------------------
#[test]
fn test_conditional_return_eval() {
    // When x <= 10, the if-else yields 0 (else branch), then x+1 is the tail.
    // When x > 10, the then branch early-returns 999.
    let src = r#"
def main() -> i64 {
    guard_return(5)
}

def guard_return(x: i64) -> i64 {
    if x > 10 {
        return 999
    } else {
        0
    };
    x + 1
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "6",
        "guard_return(5) should be 6, got: {}",
        out.trim()
    );
}

// ---------------------------------------------------------------------------
// 8. Logical operators produce valid IR
// ---------------------------------------------------------------------------
#[test]
fn test_logical_ops_ir() {
    let src = r#"
def f() -> bool {
    val a = true
    val b = false
    a && b || !a
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(
        result.is_ok(),
        "logical ops should compile to IR: {:?}",
        result.err()
    );
    let out = result.unwrap();
    // IR should contain condbr instructions from short-circuit lowering.
    assert!(
        out.contains("condbr"),
        "IR should contain condbr for short-circuit: {}",
        out
    );
}
