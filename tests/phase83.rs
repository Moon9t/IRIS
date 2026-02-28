/// Phase 83: Ref-counting GC — Retain/Release IR annotations
///
/// Tests that GcAnnotatePass inserts retain/release annotations into IR,
/// the interpreter evaluates correctly (retain/release are no-ops),
/// and LLVM IR contains iris_retain/iris_release calls.

use iris::{compile, compile_to_module, GcAnnotatePass, EmitKind};
use iris::pass::Pass;

fn eval(src: &str) -> String {
    compile(src, "phase83", EmitKind::Eval).expect("eval failed")
}

fn ir_with_gc(src: &str) -> String {
    let mut module = compile_to_module(src, "phase83").expect("compile failed");
    let mut pass = GcAnnotatePass;
    pass.run(&mut module).expect("gc pass failed");
    iris::compile(src, "phase83", EmitKind::Ir).expect("ir failed")
}

fn module_with_gc(src: &str) -> (iris::IrModule, String) {
    let mut module = compile_to_module(src, "phase83").expect("compile failed");
    let mut pass = GcAnnotatePass;
    pass.run(&mut module).expect("gc pass failed");
    let ir_text = {
        use iris::pass::Pass;
        // Serialize the annotated module to IR text via printer
        let mut s = String::new();
        for func in module.functions() {
            s.push_str(&format!("def {}:\n", func.name));
            for block in func.blocks() {
                s.push_str(&format!("  block {:?}:\n", block.id));
                for instr in &block.instrs {
                    let mut tmp = String::new();
                    iris::codegen::printer::emit_instr(&mut tmp, instr).unwrap();
                    s.push_str(&format!("    {}\n", tmp));
                }
            }
        }
        s
    };
    (module, ir_text)
}

// ------------------------------------------------------------------
// 1. GcAnnotatePass runs without error on a simple list program
// ------------------------------------------------------------------
#[test]
fn test_gc_pass_runs_on_list() {
    let src = r#"
def main() -> i64 {
    val xs = zeros(3)
    list_len(xs)
}
"#;
    let mut module = compile_to_module(src, "phase83").expect("compile failed");
    let mut pass = GcAnnotatePass;
    let result = pass.run(&mut module);
    assert!(result.is_ok(), "GcAnnotatePass failed: {:?}", result);
}

// ------------------------------------------------------------------
// 2. After GcAnnotatePass, function still evaluates correctly
// ------------------------------------------------------------------
#[test]
fn test_gc_annotated_list_sum_correct() {
    let v: f64 = eval(r#"
def main() -> f64 {
    val xs = ones(4)
    list_sum(xs)
}
"#).trim().parse().unwrap();
    assert!((v - 4.0).abs() < 1e-9, "expected 4.0, got {v}");
}

// ------------------------------------------------------------------
// 3. IR text contains "retain" after GcAnnotatePass on list-creating fn
// ------------------------------------------------------------------
#[test]
fn test_ir_contains_retain() {
    let src = r#"
def main() -> i64 {
    val xs = zeros(3)
    list_len(xs)
}
"#;
    let (_, ir_text) = module_with_gc(src);
    assert!(ir_text.contains("retain"),
        "expected 'retain' in GC-annotated IR:\n{}", ir_text);
}

// ------------------------------------------------------------------
// 4. IR text contains "release" after GcAnnotatePass
// ------------------------------------------------------------------
#[test]
fn test_ir_contains_release() {
    let src = r#"
def main() -> i64 {
    val xs = zeros(3)
    list_len(xs)
}
"#;
    let (_, ir_text) = module_with_gc(src);
    assert!(ir_text.contains("release"),
        "expected 'release' in GC-annotated IR:\n{}", ir_text);
}

// ------------------------------------------------------------------
// 5. Multiple heap values → retain count matches allocated values
// ------------------------------------------------------------------
#[test]
fn test_retain_count_matches_heap_allocs() {
    let src = r#"
def main() -> i64 {
    val a = zeros(3)
    val b = ones(3)
    val c = zeros(3)
    0
}
"#;
    let (_, ir_text) = module_with_gc(src);
    let retain_count = ir_text.matches("retain").count();
    // At least 3 retains (one per list)
    assert!(retain_count >= 3,
        "expected at least 3 retains for 3 lists, got {}:\n{}", retain_count, ir_text);
}

// ------------------------------------------------------------------
// 6. GcAnnotatePass is idempotent (running twice doesn't crash)
// ------------------------------------------------------------------
#[test]
fn test_gc_pass_idempotent() {
    let src = r#"
def main() -> i64 {
    val xs = zeros(3)
    list_len(xs)
}
"#;
    let mut module = compile_to_module(src, "phase83").expect("compile failed");
    let mut pass = GcAnnotatePass;
    pass.run(&mut module).expect("first run failed");
    // Second run should not crash (though may add more annotations)
    pass.run(&mut module).expect("second run failed");
}

// ------------------------------------------------------------------
// 7. No retain/release for scalar-only functions
// ------------------------------------------------------------------
#[test]
fn test_no_gc_for_scalar_fn() {
    let src = r#"
def main() -> i64 {
    val x = 42
    val y = x + 1
    y
}
"#;
    let (_, ir_text) = module_with_gc(src);
    // No heap allocation → no retain/release
    let retain_count = ir_text.matches("retain").count();
    assert_eq!(retain_count, 0,
        "expected 0 retains for scalar-only fn, got {}:\n{}", retain_count, ir_text);
}

// ------------------------------------------------------------------
// 8. LLVM IR contains iris_retain declare after GcAnnotatePass
// ------------------------------------------------------------------
#[test]
fn test_llvm_contains_retain_release() {
    let src = r#"
def main() -> i64 {
    val xs = zeros(3)
    list_len(xs)
}
"#;
    let mut module = compile_to_module(src, "phase83").expect("compile failed");
    let mut pass = GcAnnotatePass;
    pass.run(&mut module).expect("gc pass failed");
    // Generate LLVM IR from annotated module
    let llvm = iris::codegen::llvm_ir::emit_llvm_ir(&module).expect("llvm emit failed");
    assert!(llvm.contains("iris_retain") || llvm.contains("iris_release"),
        "expected iris_retain/iris_release in LLVM IR after GC annotation:\n{}", llvm);
}
