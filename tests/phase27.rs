//! Phase 27 integration tests: `par for` parallel range loops.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. par for compiles to IR containing par_for
// ---------------------------------------------------------------------------
#[test]
fn test_par_for_ir() {
    let src = r#"
def f() -> i64 {
    par for i in 0..3 {
        val x = i * 2
    }
    0
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("par_for"),
        "IR should contain par_for, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 2. par for with zero-range does nothing, returns 0
// ---------------------------------------------------------------------------
#[test]
fn test_par_for_zero_iters() {
    let src = r#"
def f() -> i64 {
    par for i in 0..0 {
        val x = i + 1
    }
    42
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "zero-range par for should return 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. par for range 0..5 runs 5 iterations (counted via atomic)
// ---------------------------------------------------------------------------
#[test]
fn test_par_for_count() {
    let src = r#"
def f() -> i64 {
    val acc = atomic_new(0)
    par for i in 0..5 {
        atomic_add(acc, 1)
    }
    atomic_load(acc)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "5", "par for 0..5 should run 5 times, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. par for accumulates sum 0+1+2+3+4 = 10
// ---------------------------------------------------------------------------
#[test]
fn test_par_for_accumulate() {
    let src = r#"
def f() -> i64 {
    val acc = atomic_new(0)
    par for i in 0..5 {
        atomic_add(acc, i)
    }
    atomic_load(acc)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "10", "sum 0+1+2+3+4 = 10, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. par for with nonzero start: 2..5 runs 3 iterations
// ---------------------------------------------------------------------------
#[test]
fn test_par_for_start_nonzero() {
    let src = r#"
def f() -> i64 {
    val acc = atomic_new(0)
    par for i in 2..5 {
        atomic_add(acc, 1)
    }
    atomic_load(acc)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "3", "2..5 runs 3 iterations, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. par for loop var is accessible inside body
// ---------------------------------------------------------------------------
#[test]
fn test_par_for_loop_var() {
    let src = r#"
def f() -> i64 {
    val acc = atomic_new(0)
    par for i in 0..4 {
        atomic_add(acc, i * i)
    }
    atomic_load(acc)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // 0^2 + 1^2 + 2^2 + 3^2 = 0 + 1 + 4 + 9 = 14
    assert_eq!(out.trim(), "14", "sum of squares 0..4 = 14, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. par for body with conditional (if/else based on comparison)
// ---------------------------------------------------------------------------
#[test]
fn test_par_for_cond_body() {
    let src = r#"
def f() -> i64 {
    val acc = atomic_new(0)
    par for i in 0..6 {
        if i < 3 {
            atomic_add(acc, 1)
        } else {
            atomic_add(acc, 0)
        }
    }
    atomic_load(acc)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // i < 3 for 0..6: true for 0,1,2 → 3 increments
    assert_eq!(out.trim(), "3", "i < 3 in 0..6 gives 3 increments, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. par for body uses captured outer variable (read-only)
// ---------------------------------------------------------------------------
#[test]
fn test_par_for_capture_read() {
    let src = r#"
def f() -> i64 {
    val multiplier = 3
    val acc = atomic_new(0)
    par for i in 0..4 {
        atomic_add(acc, i * multiplier)
    }
    atomic_load(acc)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // (0+1+2+3) * 3 = 18
    assert_eq!(out.trim(), "18", "sum(0..4) * 3 = 18, got: {}", out.trim());
}
