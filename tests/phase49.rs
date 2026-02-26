// phase49.rs — Complete LLVM IR Backend
//
// Tests for the enhanced LLVM IR emitter (EmitKind::LlvmComplete):
//   - Named struct type declarations (%Name = type { ... })
//   - Fixed-size scalar arrays via alloca + GEP
//   - Typed user-defined function call signatures (not opaque ptr)
//   - nounwind willreturn attributes on pure functions
//   - nsw flag on integer arithmetic
//   - LLVM intrinsic declares at top of module
//   - Target triple/datalayout header

use iris::{compile, EmitKind};

// ── Test 1: target triple and datalayout present ───────────────────────────

#[test]
fn test_llvm_complete_header() {
    let src = r#"def f() -> i64 { 42 }"#;
    let ir = compile(src, "test", EmitKind::LlvmComplete).unwrap();
    assert!(
        ir.contains("target triple"),
        "expected 'target triple' in LlvmComplete output:\n{}",
        ir
    );
    assert!(
        ir.contains("target datalayout"),
        "expected 'target datalayout' in LlvmComplete output:\n{}",
        ir
    );
    assert!(
        ir.contains("x86_64"),
        "expected 'x86_64' in target triple:\n{}",
        ir
    );
}

// ── Test 2: struct type declarations ──────────────────────────────────────

#[test]
fn test_struct_type_declaration() {
    let src = r#"
record Point { x: i64, y: i64 }
def make() -> i64 { 0 }
"#;
    let ir = compile(src, "test", EmitKind::LlvmComplete).unwrap();
    assert!(
        ir.contains("%Point = type"),
        "expected named struct type '%Point = type' in output:\n{}",
        ir
    );
    assert!(
        ir.contains("i64"),
        "expected i64 fields in struct type:\n{}",
        ir
    );
}

// ── Test 3: scalar array uses alloca ──────────────────────────────────────

#[test]
fn test_scalar_array_alloca() {
    let src = r#"
def f() -> i64 {
    val arr = [1, 2, 3]
    arr[0]
}
"#;
    let ir = compile(src, "test", EmitKind::LlvmComplete).unwrap();
    assert!(
        ir.contains("alloca"),
        "expected 'alloca' for scalar array in LlvmComplete output:\n{}",
        ir
    );
}

// ── Test 4: typed user function calls ─────────────────────────────────────

#[test]
fn test_typed_function_call() {
    let src = r#"
def add(x: i64, y: i64) -> i64 { x + y }
def main() -> i64 { add(3, 4) }
"#;
    let ir = compile(src, "test", EmitKind::LlvmComplete).unwrap();
    // Typed call should include the actual type, not just ptr.
    assert!(
        ir.contains("call i64 @add(i64"),
        "expected typed call 'call i64 @add(i64 ...' in LlvmComplete:\n{}",
        ir
    );
}

// ── Test 5: nounwind willreturn on pure functions ──────────────────────────

#[test]
fn test_pure_function_attributes() {
    let src = r#"def pure_fn(x: i64) -> i64 { x * 2 }"#;
    let ir = compile(src, "test", EmitKind::LlvmComplete).unwrap();
    assert!(
        ir.contains("nounwind") || ir.contains("willreturn"),
        "expected nounwind/willreturn attributes on pure function:\n{}",
        ir
    );
}

// ── Test 6: nsw flag on integer arithmetic ─────────────────────────────────

#[test]
fn test_nsw_integer_arithmetic() {
    let src = r#"def f(x: i64) -> i64 { x + 1 }"#;
    let ir = compile(src, "test", EmitKind::LlvmComplete).unwrap();
    assert!(
        ir.contains("add nsw"),
        "expected 'add nsw' (no signed wrap) in LlvmComplete arithmetic:\n{}",
        ir
    );
}

// ── Test 7: LLVM intrinsic declarations present ────────────────────────────

#[test]
fn test_llvm_intrinsic_declares() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::LlvmComplete).unwrap();
    assert!(
        ir.contains("declare double @llvm.sqrt.f64"),
        "expected '@llvm.sqrt.f64' declare:\n{}",
        ir
    );
    assert!(
        ir.contains("declare double @llvm.sin.f64"),
        "expected '@llvm.sin.f64' declare:\n{}",
        ir
    );
    assert!(
        ir.contains("declare double @llvm.fabs.f64"),
        "expected '@llvm.fabs.f64' declare:\n{}",
        ir
    );
}

// ── Test 8: result still correct when using LlvmComplete ──────────────────

#[test]
fn test_llvm_complete_eval_still_works() {
    let src = r#"def f() -> i64 { 6 * 7 }"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
    // Also verify LlvmComplete doesn't break the function definition.
    let ir = compile(src, "test", EmitKind::LlvmComplete).unwrap();
    assert!(ir.contains("define i64 @f()"), "expected 'define i64 @f()':\n{}", ir);
}
