/// Phase 81: FFI / extern declarations
///
/// Tests: extern def syntax, IR emit, LLVM declare emit, interpreter dispatch,
///        call_extern in IR text, arg passing, multiple extern fns.

use iris::{compile, compile_to_module, EmitKind};

fn eval(src: &str) -> String {
    compile(src, "phase81", EmitKind::Eval).expect("eval failed")
}

fn ir(src: &str) -> String {
    compile(src, "phase81", EmitKind::Ir).expect("ir failed")
}

fn llvm(src: &str) -> String {
    compile(src, "phase81", EmitKind::Llvm).expect("llvm failed")
}

fn module(src: &str) -> iris::IrModule {
    compile_to_module(src, "phase81").expect("compile_to_module failed")
}

// ------------------------------------------------------------------
// 1. extern def is parsed and registered in IrModule.extern_fns
// ------------------------------------------------------------------
#[test]
fn test_extern_fn_registered_in_module() {
    let src = r#"
extern def my_c_func(x: i64, y: f64) -> f64
def main() -> i64 { 0 }
"#;
    let m = module(src);
    let found = m.extern_fns.iter().any(|e| e.name == "my_c_func");
    assert!(found, "extern fn 'my_c_func' not registered in IrModule.extern_fns");
}

// ------------------------------------------------------------------
// 2. extern fn has correct param count and return type (debug repr)
// ------------------------------------------------------------------
#[test]
fn test_extern_fn_types() {
    let src = r#"
extern def add_doubles(a: f64, b: f64) -> f64
def main() -> i64 { 0 }
"#;
    let m = module(src);
    let ext = m.extern_fns.iter().find(|e| e.name == "add_doubles").unwrap();
    assert_eq!(ext.param_types.len(), 2, "expected 2 params");
    let ret_debug = format!("{:?}", ext.ret_ty);
    assert!(ret_debug.contains("F64") || ret_debug.contains("f64"),
        "expected f64 return type, got: {}", ret_debug);
}

// ------------------------------------------------------------------
// 3. call_extern appears in IR text
// ------------------------------------------------------------------
#[test]
fn test_call_extern_in_ir() {
    let src = r#"
extern def square(x: f64) -> f64
def main() -> i64 {
    val _ = square(3.0 to f64)
    0
}
"#;
    let ir_text = ir(src);
    assert!(
        ir_text.contains("call_extern @square"),
        "expected 'call_extern @square' in IR:\n{}", ir_text
    );
}

// ------------------------------------------------------------------
// 4. LLVM IR emits a declare for the extern fn
// ------------------------------------------------------------------
#[test]
fn test_llvm_declare_emitted() {
    let src = r#"
extern def cblas_ddot(n: i64, x: f64, y: f64) -> f64
def main() -> i64 { 0 }
"#;
    let llvm_text = llvm(src);
    assert!(
        llvm_text.contains("declare") && llvm_text.contains("@cblas_ddot"),
        "expected 'declare ... @cblas_ddot' in LLVM IR:\n{}", llvm_text
    );
}

// ------------------------------------------------------------------
// 5. Calling an unknown extern fn in interpreter returns zero stub
// ------------------------------------------------------------------
#[test]
fn test_interpreter_unknown_extern_returns_zero() {
    let src = r#"
extern def some_extern_fn(x: i64) -> i64
def main() -> i64 {
    val x = some_extern_fn(42)
    x + 1
}
"#;
    // Unknown stub → returns 0 (i64), so 0 + 1 = 1
    let result = eval(src);
    assert_eq!(result.trim(), "1", "expected 1 from (stub_returns_0)+1, got: {}", result);
}

// ------------------------------------------------------------------
// 6. Multiple extern fns registered correctly
// ------------------------------------------------------------------
#[test]
fn test_multiple_extern_fns() {
    let src = r#"
extern def fn_a(x: i64) -> i64
extern def fn_b(x: f64, y: f64) -> f64
extern def fn_c() -> i64
def main() -> i64 { 0 }
"#;
    let m = module(src);
    assert_eq!(m.extern_fns.len(), 3, "expected 3 extern fns, got {}", m.extern_fns.len());
    let names: Vec<&str> = m.extern_fns.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"fn_a"), "fn_a not in extern_fns");
    assert!(names.contains(&"fn_b"), "fn_b not in extern_fns");
    assert!(names.contains(&"fn_c"), "fn_c not in extern_fns");
}

// ------------------------------------------------------------------
// 7. extern fn with no params
// ------------------------------------------------------------------
#[test]
fn test_extern_fn_no_params() {
    let src = r#"
extern def get_time() -> i64
def main() -> i64 {
    get_time()
}
"#;
    let ir_text = ir(src);
    assert!(ir_text.contains("call_extern @get_time"),
        "expected 'call_extern @get_time' in IR:\n{}", ir_text);
    let llvm_text = llvm(src);
    assert!(llvm_text.contains("@get_time"), "expected @get_time in LLVM IR");
}

// ------------------------------------------------------------------
// 8. extern fn return value flows through arithmetic
// ------------------------------------------------------------------
#[test]
fn test_extern_fn_result_used_in_arithmetic() {
    let src = r#"
extern def my_const() -> i64
def main() -> i64 {
    val x = my_const()
    x + 10
}
"#;
    // Unknown stub → returns 0 (i64), so 0 + 10 = 10
    let result = eval(src);
    assert_eq!(result.trim(), "10", "expected 10 from (stub_returns_0)+10, got: {}", result);
}
