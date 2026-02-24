//! Phase 17 integration tests: function call type resolution.
//!
//! Previously, Call instructions always used `IrType::Infer` as the return
//! type, which failed `ValidatePass`. Now we pre-collect function signatures
//! and emit the correct concrete return type at each call site.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Calling a no-arg helper function compiles through the full pipeline
// ---------------------------------------------------------------------------
#[test]
fn test_call_noarg_helper_compiles() {
    let src = r#"
def helper() -> i64 { 42 }
def main_fn() -> i64 { helper() }
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "inter-function call should compile: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// 2. Call IR does NOT contain "Infer" for the result type
// ---------------------------------------------------------------------------
#[test]
fn test_call_result_type_not_infer() {
    let src = r#"
def helper() -> i64 { 42 }
def main_fn() -> i64 { helper() }
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(!out.contains("Infer"), "IR should not contain Infer: {}", out);
}

// ---------------------------------------------------------------------------
// 3. Cross-function call evaluates correctly
// ---------------------------------------------------------------------------
#[test]
fn test_call_eval_noarg() {
    let src = r#"
def main_fn() -> i64 { helper() }
def helper() -> i64 { 42 }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "helper() should return 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. Calling a helper with arguments compiles
// ---------------------------------------------------------------------------
#[test]
fn test_call_with_args_compiles() {
    let src = r#"
def add(a: i64, b: i64) -> i64 { a + b }
def main_fn() -> i64 { add(10, 32) }
"#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "call with args should compile: {:?}", result.err());
}

// ---------------------------------------------------------------------------
// 5. Call with args evaluates correctly
// ---------------------------------------------------------------------------
#[test]
fn test_call_with_args_eval() {
    let src = r#"
def main_fn() -> i64 { add(10, 32) }
def add(a: i64, b: i64) -> i64 { a + b }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "add(10, 32) should be 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 6. Chained calls: f() calls g() which calls h()
// ---------------------------------------------------------------------------
#[test]
fn test_call_chain_eval() {
    let src = r#"
def main_fn() -> i64 { double(triple(2)) }
def double(x: i64) -> i64 { x * 2 }
def triple(x: i64) -> i64 { x * 3 }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // double(triple(2)) = double(6) = 12
    assert_eq!(out.trim(), "12", "double(triple(2)) = 12, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. Multiple calls to the same function
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_calls_to_same_fn() {
    let src = r#"
def main_fn() -> i64 {
    val a = square(3)
    val b = square(4)
    a + b
}
def square(x: i64) -> i64 { x * x }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    // 9 + 16 = 25 (Pythagorean triple check)
    assert_eq!(out.trim(), "25", "3^2 + 4^2 = 25, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. Call result used in arithmetic expression
// ---------------------------------------------------------------------------
#[test]
fn test_call_result_in_expr() {
    let src = r#"
def main_fn() -> i64 { get_base() + 8 }
def get_base() -> i64 { 34 }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "34 + 8 = 42, got: {}", out.trim());
}
