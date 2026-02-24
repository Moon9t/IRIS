//! Phase 9 integration tests: while/loop/break/continue.

use iris::{compile, EmitKind};

#[test]
fn test_while_basic() {
    let src = "def count() -> i64 { val x = 0; while x < 5 { val x = x + 1 } x }";
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(ir.contains("cmplt"), "IR should contain cmplt: {}", ir);
    assert!(ir.contains("while_header"), "IR should have while_header block: {}", ir);
}

#[test]
fn test_while_zero_iterations() {
    let src = "def zero() -> i64 { val x = 0; while false { val x = x + 1 } x }";
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "should compile: {:?}", result.err());
}

#[test]
fn test_loop_with_break() {
    let src = "def find() -> bool { loop { break } false }";
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    assert!(ir.contains("loop_merge"), "IR should contain loop_merge: {}", ir);
}

#[test]
fn test_while_ir_has_back_edge() {
    let src = "def count() -> i64 { val x = 0; while x < 10 { val x = x + 1 } x }";
    let ir = compile(src, "test", EmitKind::Ir).expect("should compile");
    let br_count = ir.matches("br ").count();
    assert!(br_count >= 2, "should have at least 2 br instructions, got {}: {}", br_count, ir);
}

#[test]
fn test_break_exits_loop() {
    let src = r#"
        def search(x: i64) -> i64 {
            val result = 0
            while x > 0 {
                val result = x
                break
            }
            result
        }
    "#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "should compile: {:?}", result.err());
}

#[test]
fn test_continue_skips_body() {
    let src = r#"
        def skip(x: i64) -> i64 {
            val y = 0
            while x > 0 {
                val x = x - 1
                continue
            }
            y
        }
    "#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "should compile: {:?}", result.err());
}

#[test]
fn test_nested_while() {
    let src = r#"
        def nested() -> i64 {
            val i = 0
            while i < 3 {
                val j = 0
                while j < 3 {
                    val j = j + 1
                }
                val i = i + 1
            }
            i
        }
    "#;
    let result = compile(src, "test", EmitKind::Ir);
    assert!(result.is_ok(), "nested while should compile: {:?}", result.err());
}

#[test]
fn test_while_llvm() {
    let src = "def count() -> i64 { val x = 0; while x < 5 { val x = x + 1 } x }";
    let llvm = compile(src, "test", EmitKind::Llvm).expect("should compile to LLVM");
    assert!(llvm.contains("br label"), "LLVM IR should contain 'br label': {}", llvm);
}
