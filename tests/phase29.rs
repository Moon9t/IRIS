//! Phase 29 integration tests: barrier() and parallel_reduce
//!
//! barrier() is a sync point that is a no-op in the interpreter.

use iris::{compile, EmitKind};

// 1. barrier() compiles to IR containing "barrier"
#[test]
fn test_barrier_compiles_to_ir() {
    let src = r#"
def f() -> i64 {
    par for i in 0..4 {
        val x = i * 2
    }
    barrier();
    42
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(out.contains("barrier"), "IR should contain barrier, got:\n{}", out);
}

// 2. barrier() is a no-op - value after it is returned
#[test]
fn test_barrier_noop() {
    let src = r#"
def f() -> i64 {
    barrier();
    99
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "99", "barrier is no-op, should return 99, got: {}", out.trim());
}

// 3. multiple barriers work
#[test]
fn test_multiple_barriers() {
    let src = r#"
def f() -> i64 {
    barrier();
    barrier();
    barrier();
    7
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "7", "three barriers still return 7, got: {}", out.trim());
}

// 4. barrier between par for loops
#[test]
fn test_barrier_between_par_fors() {
    let src = r#"
def f() -> i64 {
    par for i in 0..3 {
        val x = i + 1
    }
    barrier();
    par for j in 0..3 {
        val y = j * 2
    }
    barrier();
    42
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "barrier between par fors should return 42, got: {}", out.trim());
}

// 5. barrier in a conditional branch
#[test]
fn test_barrier_in_conditional() {
    let src = r#"
def f() -> i64 {
    val x = 10
    if x > 5 {
        barrier();
        100
    } else {
        200
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "100", "barrier in conditional should return 100, got: {}", out.trim());
}

// 6. barrier with variable access before and after
#[test]
fn test_barrier_with_vars() {
    let src = r#"
def f() -> i64 {
    val a = 5;
    barrier();
    val b = a + 3
    b
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "8", "barrier with vars should return 8, got: {}", out.trim());
}

// 7. barrier IR contains the barrier instruction text
#[test]
fn test_barrier_ir_text() {
    let src = r#"
def f() -> i64 {
    val x = 1;
    barrier();
    x
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(out.contains("barrier"), "IR should contain barrier keyword, got:\n{}", out);
}

// 8. barrier with arithmetic - value preserved across barrier
#[test]
fn test_barrier_preserves_value() {
    let src = r#"
def f() -> i64 {
    val result = 3 * 7;
    barrier();
    result
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "21", "barrier should preserve 3*7=21, got: {}", out.trim());
}
