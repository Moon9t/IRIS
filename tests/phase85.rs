/// Phase 85: Hindley-Milner type inference.

use iris::{compile, compile_to_module, HmTypeInferPass, EmitKind};
use iris::pass::Pass;

fn eval(src: &str) -> String {
    compile(src, "phase85", EmitKind::Eval).expect("eval failed")
}

// ------------------------------------------------------------------
// 1. HmTypeInferPass runs without error on a simple integer program
// ------------------------------------------------------------------
#[test]
fn test_hm_pass_runs_on_int() {
    let src = r#"
def main() -> i64 { 42 }
"#;
    let mut module = compile_to_module(src, "phase85").expect("compile failed");
    let mut pass = HmTypeInferPass;
    assert!(pass.run(&mut module).is_ok());
}

// ------------------------------------------------------------------
// 2. Pass doesn't change semantics — int arithmetic still correct
// ------------------------------------------------------------------
#[test]
fn test_hm_int_arithmetic_unchanged() {
    let src = r#"
def main() -> i64 {
    val x = 6
    val y = 7
    x * y
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 42);
}

// ------------------------------------------------------------------
// 3. Pass runs on float program
// ------------------------------------------------------------------
#[test]
fn test_hm_pass_runs_on_float() {
    let src = r#"
def main() -> f32 { 3.14 }
"#;
    let mut module = compile_to_module(src, "phase85").expect("compile failed");
    let mut pass = HmTypeInferPass;
    assert!(pass.run(&mut module).is_ok());
}

// ------------------------------------------------------------------
// 4. Float computation semantics preserved
// ------------------------------------------------------------------
#[test]
fn test_hm_float_result_correct() {
    let src = r#"
def main() -> f32 {
    val x = 2.0
    val y = 3.0
    x + y
}
"#;
    let v: f32 = eval(src).trim().parse().unwrap();
    assert!((v - 5.0_f32).abs() < 1e-6, "expected 5.0, got {v}");
}

// ------------------------------------------------------------------
// 5. Pass runs on boolean program
// ------------------------------------------------------------------
#[test]
fn test_hm_pass_runs_on_bool() {
    let src = r#"
def main() -> i64 {
    val b = true
    if b { 1 } else { 0 }
}
"#;
    let mut module = compile_to_module(src, "phase85").expect("compile failed");
    let mut pass = HmTypeInferPass;
    assert!(pass.run(&mut module).is_ok());
}

// ------------------------------------------------------------------
// 6. Unification resolves Infer type for comparison result
// ------------------------------------------------------------------
#[test]
fn test_hm_comparison_produces_bool() {
    let src = r#"
def main() -> i64 {
    val a = 3
    val b = 5
    val c = a < b
    if c { 1 } else { 0 }
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 1);
}

// ------------------------------------------------------------------
// 7. Pass is idempotent (running twice on same module)
// ------------------------------------------------------------------
#[test]
fn test_hm_pass_idempotent() {
    let src = r#"
def main() -> i64 {
    val x = 10
    val y = x + 5
    y
}
"#;
    let mut module = compile_to_module(src, "phase85").expect("compile failed");
    let mut pass = HmTypeInferPass;
    pass.run(&mut module).expect("first run failed");
    pass.run(&mut module).expect("second run failed");
}

// ------------------------------------------------------------------
// 8. Infer resolved for cast target type
// ------------------------------------------------------------------
#[test]
fn test_hm_cast_resolves_type() {
    let src = r#"
def main() -> f32 {
    val x = 5
    x to f32
}
"#;
    let mut module = compile_to_module(src, "phase85").expect("compile failed");
    let mut pass = HmTypeInferPass;
    assert!(pass.run(&mut module).is_ok());
    // Also check the program still produces correct result
    let v: f32 = eval(src).trim().parse().unwrap();
    assert!((v - 5.0_f32).abs() < 1e-6, "expected 5.0, got {v}");
}
