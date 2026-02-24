//! Phase 18 integration tests: `for i in start..end { body }` range loops.
//!
//! `for i in 0..n { body }` is syntactic sugar for a while loop that
//! initialises i = 0, checks i < n, executes body, then increments i by 1.
//! The start and end expressions are evaluated once before the loop.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. `for` keyword is accepted by the lexer/parser
// ---------------------------------------------------------------------------
#[test]
fn test_for_keyword_lexed() {
    let src = r#"
def f() -> i64 {
    var acc = 0
    for i in 0..5 { acc = acc + 1 }
    acc
}
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "for keyword should be lexed: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// 2. IR contains a header/body/merge structure for the for loop
// ---------------------------------------------------------------------------
#[test]
fn test_for_loop_ir_structure() {
    let src = r#"
def f() -> i64 {
    var acc = 0
    for i in 0..3 { acc = acc + 1 }
    acc
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(out.contains("for_header"), "IR should have for_header: {}", out);
    assert!(out.contains("for_body"), "IR should have for_body: {}", out);
    assert!(out.contains("for_merge"), "IR should have for_merge: {}", out);
}

// ---------------------------------------------------------------------------
// 3. Simple counter: loop body executes exactly n times
// ---------------------------------------------------------------------------
#[test]
fn test_for_loop_counter_eval() {
    let src = r#"
def count_to_5() -> i64 {
    var acc = 0
    for i in 0..5 {
        acc = acc + 1
    }
    acc
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "5", "loop should run 5 times, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. Accumulator: sum 0+1+2+3+4 = 10
// ---------------------------------------------------------------------------
#[test]
fn test_for_loop_sum_eval() {
    let src = r#"
def sum_range() -> i64 {
    var acc = 0
    for i in 0..5 {
        acc = acc + i
    }
    acc
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // 0+1+2+3+4 = 10
    assert_eq!(out.trim(), "10", "sum 0..5 = 10, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. Zero-iteration range: 0..0 runs body 0 times
// ---------------------------------------------------------------------------
#[test]
fn test_for_loop_zero_iterations() {
    let src = r#"
def empty_range() -> i64 {
    var acc = 99
    for i in 0..0 {
        acc = 0
    }
    acc
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "99", "zero-iteration loop should not modify acc, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. Loop variable is accessible inside the body
// ---------------------------------------------------------------------------
#[test]
fn test_for_loop_var_accessible() {
    let src = r#"
def last_i() -> i64 {
    var last = 0
    for i in 1..6 {
        last = i
    }
    last
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // last assignment is i=5 (loop runs for i=1,2,3,4,5)
    assert_eq!(out.trim(), "5", "last i should be 5, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. Nested for loops (inner uses outer accumulator)
// ---------------------------------------------------------------------------
#[test]
fn test_nested_for_loops_eval() {
    let src = r#"
def multiplication() -> i64 {
    var acc = 0
    for i in 0..3 {
        for j in 0..3 {
            acc = acc + 1
        }
    }
    acc
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // 3 * 3 = 9 iterations
    assert_eq!(out.trim(), "9", "3x3 nested loop = 9 iterations, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. For loop with non-zero start
// ---------------------------------------------------------------------------
#[test]
fn test_for_loop_nonzero_start() {
    let src = r#"
def sum_3_to_6() -> i64 {
    var acc = 0
    for i in 3..7 {
        acc = acc + i
    }
    acc
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // 3+4+5+6 = 18
    assert_eq!(out.trim(), "18", "sum 3..7 = 18, got: {}", out.trim());
}
