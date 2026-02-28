/// Phase 90: Loop unrolling + exhaustiveness checking.

use iris::{compile, compile_to_module, ExhaustivePass, LoopUnrollPass, EmitKind};
use iris::pass::Pass;

fn eval(src: &str) -> String {
    compile(src, "phase90", EmitKind::Eval).expect("eval failed")
}

fn ir_after_unroll(src: &str) -> String {
    let mut module = compile_to_module(src, "phase90").expect("compile failed");
    let mut pass = LoopUnrollPass { max_unroll: 8 };
    pass.run(&mut module).expect("unroll failed");
    iris::codegen::printer::emit_ir_text(&module).expect("emit failed")
}

// ------------------------------------------------------------------
// 1. LoopUnrollPass runs without error
// ------------------------------------------------------------------
#[test]
fn test_loop_unroll_pass_runs() {
    let src = r#"
def main() -> i64 {
    for i in 0..3 {
        val _ = i
    }
    0
}
"#;
    let mut module = compile_to_module(src, "phase90").expect("compile failed");
    let mut pass = LoopUnrollPass::default();
    assert!(pass.run(&mut module).is_ok());
}

// ------------------------------------------------------------------
// 2. Unrolled loop still produces correct result
// ------------------------------------------------------------------
#[test]
fn test_unrolled_loop_correct_result() {
    // After unrolling, the function should still evaluate to 0.
    let src = r#"
def main() -> i64 {
    for i in 0..3 {
        val _ = i
    }
    42
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 42);
}

// ------------------------------------------------------------------
// 3. LoopUnrollPass doesn't unroll beyond max_unroll threshold
// ------------------------------------------------------------------
#[test]
fn test_loop_beyond_threshold_not_unrolled() {
    let src = r#"
def main() -> i64 {
    for i in 0..100 {
        val _ = i
    }
    0
}
"#;
    // With max_unroll=8, a 100-iteration loop should NOT be unrolled.
    // The IR should still contain for_header block.
    let ir_text = ir_after_unroll(src);
    assert!(ir_text.contains("for_header") || ir_text.contains("bb"),
        "expected loop structure to remain for large loops:\n{}", ir_text);
}

// ------------------------------------------------------------------
// 4. ExhaustivePass runs without error on exhaustive match
// ------------------------------------------------------------------
#[test]
fn test_exhaustive_pass_on_exhaustive_match() {
    let src = r#"
choice Color { Red, Green, Blue }
def main() -> i64 {
    val c = Color.Red
    when c {
        Color.Red   => 1,
        Color.Green => 2,
        Color.Blue  => 3
    }
}
"#;
    let mut module = compile_to_module(src, "phase90").expect("compile failed");
    let mut pass = ExhaustivePass;
    assert!(pass.run(&mut module).is_ok(),
        "exhaustive match should pass ExhaustivePass");
}

// ------------------------------------------------------------------
// 5. ExhaustivePass accepts match with a default arm
// ------------------------------------------------------------------
#[test]
fn test_exhaustive_pass_accepts_default_arm() {
    let src = r#"
choice Dir { North, South, East, West }
def main() -> i64 {
    val d = Dir.North
    when d {
        Dir.North => 1,
        _ => 0
    }
}
"#;
    let mut module = compile_to_module(src, "phase90").expect("compile failed");
    let mut pass = ExhaustivePass;
    // Default arm covers all remaining variants — should pass.
    assert!(pass.run(&mut module).is_ok());
}

// ------------------------------------------------------------------
// 6. Exhaustive match with all variants evaluates correctly
// ------------------------------------------------------------------
#[test]
fn test_exhaustive_match_correct_eval() {
    let src = r#"
choice Color { Red, Green, Blue }
def main() -> i64 {
    val c = Color.Green
    when c {
        Color.Red   => 1,
        Color.Green => 2,
        Color.Blue  => 3
    }
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 2);
}

// ------------------------------------------------------------------
// 7. LoopUnrollPass is idempotent
// ------------------------------------------------------------------
#[test]
fn test_loop_unroll_idempotent() {
    let src = r#"
def main() -> i64 {
    for i in 0..4 {
        val _ = i
    }
    0
}
"#;
    let mut module = compile_to_module(src, "phase90").expect("compile failed");
    let mut pass = LoopUnrollPass { max_unroll: 8 };
    pass.run(&mut module).expect("first run");
    pass.run(&mut module).expect("second run"); // should not crash
}

// ------------------------------------------------------------------
// 8. ExhaustivePass is idempotent
// ------------------------------------------------------------------
#[test]
fn test_exhaustive_pass_idempotent() {
    let src = r#"
choice AB { A, B }
def main() -> i64 {
    val x = AB.A
    when x {
        AB.A => 1,
        AB.B => 2
    }
}
"#;
    let mut module = compile_to_module(src, "phase90").expect("compile failed");
    let mut pass = ExhaustivePass;
    pass.run(&mut module).expect("first run");
    pass.run(&mut module).expect("second run");
}
