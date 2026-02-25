//! Phase 28 integration tests: async/await
//!
//! async def is syntactic sugar -- it compiles to a normal IR function.
//! await expr lowers to a normal function call.

use iris::{compile, EmitKind};

// 1. async def compiles to IR (checking IR contains both functions)
#[test]
fn test_async_def_compiles_to_ir() {
    let src = r#"
def train() -> i64 {
    val batch = await fetch(0)
    batch
}
async def fetch(id: i64) -> i64 { id * 2 }
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(out.contains("fetch"), "IR should contain fetch function, got:\n{}", out);
    assert!(out.contains("train"), "IR should contain train function, got:\n{}", out);
}

// 2. await call returns correct value
#[test]
fn test_await_call_returns_value() {
    let src = r#"
def train() -> i64 {
    val batch = await fetch(3)
    batch
}
async def fetch(id: i64) -> i64 { id * 2 }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "6", "await fetch(3) should return 6, got: {}", out.trim());
}

// 3. await with arithmetic in async body
#[test]
fn test_async_basic_arithmetic() {
    let src = r#"
def run() -> i64 {
    await compute(5)
}
async def compute(x: i64) -> i64 { x + 10 }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "15", "await compute(5) should return 15, got: {}", out.trim());
}

// 4. multiple awaits in one function
#[test]
fn test_async_multiple_awaits() {
    let src = r#"
def run() -> i64 {
    val a = await double(2)
    val b = await triple(3)
    a + b
}
async def double(x: i64) -> i64 { x * 2 }
async def triple(x: i64) -> i64 { x * 3 }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "13", "double(2)+triple(3) should return 13, got: {}", out.trim());
}

// 5. async function can be called without await
#[test]
fn test_async_no_await_still_works() {
    let src = r#"
def run() -> i64 {
    pure_fn(4)
}
async def pure_fn(x: i64) -> i64 { x * x }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "16", "pure_fn(4) should return 16, got: {}", out.trim());
}

// 6. await in let binding with subsequent use
#[test]
fn test_await_in_let_binding() {
    let src = r#"
def run() -> i64 {
    val x = await get_val()
    x + 1
}
async def get_val() -> i64 { 42 }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "43", "await value() + 1 should return 43, got: {}", out.trim());
}

// 7. async with conditional body
#[test]
fn test_async_with_conditional() {
    let src = r#"
def run() -> i64 {
    await fetch_cond(1)
}
async def fetch_cond(flag: i64) -> i64 {
    if flag == 1 { 100 } else { 200 }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "100", "async with conditional should return 100, got: {}", out.trim());
}

// 8. chained async calls
#[test]
fn test_async_chain() {
    let src = r#"
def pipeline() -> i64 {
    val a = await step1(5)
    await step2(a)
}
async def step1(x: i64) -> i64 { x + 1 }
async def step2(x: i64) -> i64 { x * 2 }
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "12", "pipeline should return 12, got: {}", out.trim());
}
