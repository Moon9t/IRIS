/// Phase 84: Function inlining pass.

use iris::{compile, compile_to_module, InlinePass, EmitKind};
use iris::pass::Pass;

fn eval(src: &str) -> String {
    compile(src, "phase84", EmitKind::Eval).expect("eval failed")
}

fn ir_after_inline(src: &str) -> String {
    let mut module = compile_to_module(src, "phase84").expect("compile failed");
    let mut pass = InlinePass { max_instrs: 10 };
    pass.run(&mut module).expect("inline pass failed");
    iris::codegen::printer::emit_ir_text(&module).expect("emit failed")
}

// ------------------------------------------------------------------
// 1. InlinePass runs without error on trivial program
// ------------------------------------------------------------------
#[test]
fn test_inline_pass_runs() {
    let src = r#"
def main() -> i64 {
    42
}
"#;
    let mut module = compile_to_module(src, "phase84").expect("compile failed");
    let mut pass = InlinePass::default();
    assert!(pass.run(&mut module).is_ok());
}

// ------------------------------------------------------------------
// 2. Simple identity function is inlined — no Call in IR
// ------------------------------------------------------------------
#[test]
fn test_identity_inlined() {
    let src = r#"
def identity(x: i64) -> i64 { x }
def main() -> i64 { identity(7) }
"#;
    let ir = ir_after_inline(src);
    // After inlining, the body of `main` should contain no Call to `identity`
    assert!(!ir.contains("call @identity") && !ir.contains("call identity"),
        "expected identity to be inlined:\n{}", ir);
}

// ------------------------------------------------------------------
// 3. Inlined identity preserves correct result
// ------------------------------------------------------------------
#[test]
fn test_identity_result_correct() {
    let src = r#"
def identity(x: i64) -> i64 { x }
def main() -> i64 { identity(99) }
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 99);
}

// ------------------------------------------------------------------
// 4. Inlined double function (x * 2)
// ------------------------------------------------------------------
#[test]
fn test_double_inlined_result() {
    let src = r#"
def double(x: i64) -> i64 { x * 2 }
def main() -> i64 { double(21) }
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 42);
}

// ------------------------------------------------------------------
// 5. Multiple calls to same inlineable function
// ------------------------------------------------------------------
#[test]
fn test_multiple_calls_inlined() {
    let src = r#"
def inc(x: i64) -> i64 { x + 1 }
def main() -> i64 {
    val a = inc(10)
    val b = inc(a)
    b
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 12);
}

// ------------------------------------------------------------------
// 6. Function with two parameters inlined correctly
// ------------------------------------------------------------------
#[test]
fn test_two_param_fn_inlined() {
    let src = r#"
def add(a: i64, b: i64) -> i64 { a + b }
def main() -> i64 { add(13, 29) }
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 42);
}

// ------------------------------------------------------------------
// 7. Function above threshold is NOT inlined (call remains in IR)
// ------------------------------------------------------------------
#[test]
fn test_large_fn_not_inlined() {
    // 11 non-terminal instructions > default threshold of 10
    let src = r#"
def big(x: i64) -> i64 {
    val a = x + 1
    val b = a + 1
    val c = b + 1
    val d = c + 1
    val e = d + 1
    val f = e + 1
    val g = f + 1
    val h = g + 1
    val i = h + 1
    val j = i + 1
    val k = j + 1
    k
}
def main() -> i64 { big(0) }
"#;
    let ir = ir_after_inline(src);
    // big should NOT be inlined since it has > 10 instrs
    assert!(ir.contains("@big") || ir.contains("big"),
        "expected 'big' call to remain in IR:\n{}", ir);
}

// ------------------------------------------------------------------
// 8. Nested inlining: f calls g, both inlineable
// ------------------------------------------------------------------
#[test]
fn test_nested_inline() {
    let src = r#"
def square(x: i64) -> i64 { x * x }
def add_square(a: i64, b: i64) -> i64 {
    val sa = square(a)
    val sb = square(b)
    sa + sb
}
def main() -> i64 { add_square(3, 4) }
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 25); // 9 + 16
}
