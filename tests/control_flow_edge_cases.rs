//! Phase 111 integration tests: control flow edge cases.

use iris::{compile, EmitKind};

// ── 1. Nested if-else ───────────────────────────────────────────────────────
#[test]
fn test_nested_if_else() {
    let src = r#"
def classify(x: i64) -> str {
    if x > 0 {
        if x > 100 { "big" } else { "small" }
    } else {
        "negative"
    }
}
def f() -> str {
    classify(50)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "small");
}

// ── 2. Deeply nested if ────────────────────────────────────────────────────
#[test]
fn test_deep_nested_if() {
    let src = r#"
def f() -> i64 {
    val x = 42
    if x > 10 {
        if x > 20 {
            if x > 30 {
                if x > 40 {
                    4
                } else { 3 }
            } else { 2 }
        } else { 1 }
    } else { 0 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "4");
}

// ── 3. While loop counting ───────────────────────────────────────────────
#[test]
fn test_while_count() {
    let src = r#"
def f() -> i64 {
    var i = 0
    var sum = 0
    while i < 5 {
        sum = sum + i
        i = i + 1
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 0+1+2+3+4 = 10
    assert_eq!(result.trim(), "10");
}

// ── 4. While loop with conditional arithmetic ───────────────────────────
#[test]
fn test_while_conditional() {
    let src = r#"
def f() -> i64 {
    var i = 0
    var sum = 0
    while i < 10 {
        sum = sum + i * (i % 2)
        i = i + 1
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // adds i when odd: 1+3+5+7+9 = 25
    assert_eq!(result.trim(), "25");
}

// ── 5. For range loop ───────────────────────────────────────────────────────
#[test]
fn test_for_range() {
    let src = r#"
def f() -> i64 {
    var sum = 0
    for i in 1..6 {
        sum = sum + i
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 1+2+3+4+5 = 15
    assert_eq!(result.trim(), "15");
}

// ── 6. Nested loops ────────────────────────────────────────────────────────
#[test]
fn test_nested_loops() {
    let src = r#"
def f() -> i64 {
    var total = 0
    for i in 1..4 {
        for j in 1..4 {
            total = total + i * j
        }
    }
    total
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // (1+2+3)*(1+2+3) = 36
    assert_eq!(result.trim(), "36");
}

// ── 7. Early return from function ───────────────────────────────────────────
#[test]
fn test_early_return() {
    let src = r#"
def f() -> i64 {
    return 42;
    99
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 8. Logical operators in conditions ──────────────────────────────────────
#[test]
fn test_logical_operators() {
    let src = r#"
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val a = true
    val b = false
    val r1 = bool_to_i64(a && b)
    val r2 = bool_to_i64(a || b)
    r1 * 10 + r2
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // a&&b = false(0), a||b = true(1) => 0*10 + 1 = 1
    assert_eq!(result.trim(), "1");
}
