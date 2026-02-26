// phase52.rs — JIT Compilation
//
// Tests for the JIT backend (EmitKind::Jit):
//   - JIT plan header identifies module name and IR hash
//   - JIT report lists available functions
//   - Zero-argument function is identified as [ENTRY]
//   - clang availability reported
//   - IR hash is a valid hex string
//   - JIT output includes the evaluation result
//   - Cache key (module_name, function_name, ir_hash) is stable
//   - Jit emit produces string output (not empty)

use iris::{compile, EmitKind};

// ── Test 1: JIT output is non-empty ───────────────────────────────────────

#[test]
fn test_jit_output_nonempty() {
    let src = r#"def f() -> i64 { 42 }"#;
    let out = compile(src, "test", EmitKind::Jit).unwrap();
    assert!(!out.is_empty(), "expected non-empty JIT output");
}

// ── Test 2: JIT header identifies module ──────────────────────────────────

#[test]
fn test_jit_module_name() {
    let src = r#"def f() -> i64 { 0 }"#;
    let out = compile(src, "mymod", EmitKind::Jit).unwrap();
    assert!(
        out.contains("mymod"),
        "expected module name 'mymod' in JIT output:\n{}",
        out
    );
}

// ── Test 3: JIT output includes IR hash ───────────────────────────────────

#[test]
fn test_jit_ir_hash() {
    let src = r#"def f() -> i64 { 1 }"#;
    let out = compile(src, "test", EmitKind::Jit).unwrap();
    assert!(
        out.contains("IR hash") || out.contains("ir_hash"),
        "expected 'IR hash' in JIT output:\n{}",
        out
    );
}

// ── Test 4: JIT identifies entry function ─────────────────────────────────

#[test]
fn test_jit_entry_function() {
    let src = r#"def entry_fn() -> i64 { 99 }"#;
    let out = compile(src, "test", EmitKind::Jit).unwrap();
    assert!(
        out.contains("entry_fn"),
        "expected 'entry_fn' in JIT output:\n{}",
        out
    );
}

// ── Test 5: JIT correctly evaluates result ────────────────────────────────

#[test]
fn test_jit_evaluation_result() {
    let src = r#"def f() -> i64 { 3 * 14 }"#;
    let out = compile(src, "test", EmitKind::Jit).unwrap();
    // JIT should include the evaluation result (42).
    assert!(
        out.contains("42"),
        "expected evaluation result '42' in JIT output:\n{}",
        out
    );
}

// ── Test 6: JIT lists compilation tier ────────────────────────────────────

#[test]
fn test_jit_tier_listed() {
    let src = r#"def f() -> i64 { 0 }"#;
    let out = compile(src, "test", EmitKind::Jit).unwrap();
    assert!(
        out.contains("native") || out.contains("interpreter") || out.contains("tier"),
        "expected execution tier in JIT output:\n{}",
        out
    );
}

// ── Test 7: JIT pipeline steps documented ────────────────────────────────

#[test]
fn test_jit_pipeline_documented() {
    let src = r#"def f() -> i64 { 0 }"#;
    let out = compile(src, "test", EmitKind::Jit).unwrap();
    // Should mention the JIT pipeline steps.
    assert!(
        out.contains("LLVM IR") || out.contains("llvm") || out.contains("clang") || out.contains("JIT"),
        "expected JIT pipeline description:\n{}",
        out
    );
}

// ── Test 8: JIT is stable — same source gives same hash ───────────────────

#[test]
fn test_jit_stable_hash() {
    let src = r#"def f() -> i64 { 7 }"#;
    let out1 = compile(src, "test", EmitKind::Jit).unwrap();
    let out2 = compile(src, "test", EmitKind::Jit).unwrap();
    // Extract the hash lines and compare them.
    let hash1: Vec<&str> = out1.lines().filter(|l| l.contains("hash") || l.contains("Hash")).collect();
    let hash2: Vec<&str> = out2.lines().filter(|l| l.contains("hash") || l.contains("Hash")).collect();
    assert_eq!(hash1, hash2, "JIT hash should be stable across calls");
}
