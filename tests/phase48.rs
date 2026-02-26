// phase48.rs — LLVM IR Codegen (Real)
//
// Tests that the LLVM emitter now produces proper LLVM IR with:
//   - target triple and data layout header
//   - global string constants for ConstStr values
//   - getelementptr inbounds for string references
//   - declare statements for iris runtime helper functions

use iris::{compile, EmitKind};

// ── Test 1: target triple is present ──────────────────────────────────────

#[test]
fn test_target_triple() {
    let src = r#"def f() -> i64 { 42 }"#;
    let ir = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(
        ir.contains("target triple"),
        "expected 'target triple' in LLVM output:\n{}",
        ir
    );
    assert!(
        ir.contains("x86_64"),
        "expected 'x86_64' in target triple:\n{}",
        ir
    );
}

// ── Test 2: data layout is present ────────────────────────────────────────

#[test]
fn test_data_layout() {
    let src = r#"def f() -> i64 { 1 }"#;
    let ir = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(
        ir.contains("target datalayout"),
        "expected 'target datalayout' in LLVM output:\n{}",
        ir
    );
}

// ── Test 3: string constants emit as LLVM global constants ────────────────

#[test]
fn test_string_global_constant() {
    let src = r#"def f() -> str { "hello" }"#;
    let ir = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(
        ir.contains("@.str.0"),
        "expected '@.str.0' global in LLVM output:\n{}",
        ir
    );
    assert!(
        ir.contains("private unnamed_addr constant"),
        "expected 'private unnamed_addr constant':\n{}",
        ir
    );
    assert!(
        ir.contains("hello"),
        "expected string content 'hello' in LLVM output:\n{}",
        ir
    );
}

// ── Test 4: ConstStr uses getelementptr inbounds ──────────────────────────

#[test]
fn test_string_uses_gep() {
    let src = r#"def f() -> str { "world" }"#;
    let ir = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(
        ir.contains("getelementptr inbounds"),
        "expected 'getelementptr inbounds' for string ref:\n{}",
        ir
    );
    assert!(
        ir.contains("@.str.0"),
        "expected '@.str.0' in getelementptr:\n{}",
        ir
    );
}

// ── Test 5: declare statements for iris_print are emitted ─────────────────

#[test]
fn test_declare_iris_print() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(
        ir.contains("declare void @iris_print"),
        "expected 'declare void @iris_print':\n{}",
        ir
    );
}

// ── Test 6: declare statements for string ops are emitted ─────────────────

#[test]
fn test_declare_string_ops() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(
        ir.contains("declare i64 @iris_str_len"),
        "expected 'declare i64 @iris_str_len':\n{}",
        ir
    );
    assert!(
        ir.contains("declare ptr @iris_str_concat"),
        "expected 'declare ptr @iris_str_concat':\n{}",
        ir
    );
}

// ── Test 7: arithmetic evaluation still works correctly ───────────────────

#[test]
fn test_arithmetic_eval_still_works() {
    let src = r#"def f() -> i64 { 3 * 4 + 2 }"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "14");
}

// ── Test 8: identical strings are deduplicated to a single global ─────────

#[test]
fn test_string_deduplication() {
    let src = r#"
def f() -> str {
    val a = "hello"
    val b = "hello"
    a
}
"#;
    let ir = compile(src, "test", EmitKind::Llvm).unwrap();
    // Both 'a' and 'b' reference the same literal; only one global should appear.
    let count = ir.matches("private unnamed_addr constant").count();
    assert_eq!(
        count, 1,
        "expected exactly 1 string global (deduped), got {}:\n{}",
        count, ir
    );
    assert!(ir.contains("@.str.0"), "expected @.str.0:\n{}", ir);
    assert!(!ir.contains("@.str.1"), "unexpected @.str.1 (not deduped):\n{}", ir);
}
