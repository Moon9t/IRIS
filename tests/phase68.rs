//! Phase 68 integration tests: first-class functions (function references as values).

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Assign a named function to a variable and call it
// ---------------------------------------------------------------------------
#[test]
fn test_fn_ref_assign_and_call() {
    let src = r#"
def double(x: i64) -> i64 { x * 2 }
def f() -> i64 {
    val fn_ref = double
    fn_ref(21)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// 2. Pass a function as an argument (higher-order)
// ---------------------------------------------------------------------------
#[test]
fn test_fn_ref_as_argument() {
    let src = r#"
def square(x: i64) -> i64 { x * x }
def apply(g: (i64) -> i64, x: i64) -> i64 { g(x) }
def f() -> i64 { apply(square, 7) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "49");
}

// ---------------------------------------------------------------------------
// 3. Store function reference and call multiple times
// ---------------------------------------------------------------------------
#[test]
fn test_fn_ref_multiple_calls() {
    let src = r#"
def inc(x: i64) -> i64 { x + 1 }
def f() -> i64 {
    val step = inc
    step(step(step(0)))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ---------------------------------------------------------------------------
// 4. Function reference in a closure wrapper
// ---------------------------------------------------------------------------
#[test]
fn test_fn_ref_in_closure() {
    let src = r#"
def negate(x: i64) -> i64 { 0 - x }
def f() -> i64 {
    val g = |x: i64| negate(x)
    g(10)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "-10");
}

// ---------------------------------------------------------------------------
// 5. Use list.map with a function reference (wrapped in closure)
// ---------------------------------------------------------------------------
#[test]
fn test_fn_ref_with_list_map() {
    let src = r#"
def triple(x: i64) -> i64 { x * 3 }
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    val ys = xs.map(|x: i64| triple(x))
    list_get(ys, 0) + list_get(ys, 1) + list_get(ys, 2)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "18");
}

// ---------------------------------------------------------------------------
// 6. Conditional function selection
// ---------------------------------------------------------------------------
#[test]
fn test_fn_ref_conditional() {
    let src = r#"
def add1(x: i64) -> i64 { x + 1 }
def sub1(x: i64) -> i64 { x - 1 }
def apply_fn(use_add: bool, x: i64) -> i64 {
    when use_add {
        true  => add1(x),
        false => sub1(x),
    }
}
def f() -> i64 {
    apply_fn(true, 10) + apply_fn(false, 10)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "20");
}

// ---------------------------------------------------------------------------
// 7. Apply function twice
// ---------------------------------------------------------------------------
#[test]
fn test_fn_ref_apply_twice() {
    let src = r#"
def dbl(x: i64) -> i64 { x * 2 }
def apply_twice(g: (i64) -> i64, x: i64) -> i64 { g(g(x)) }
def f() -> i64 { apply_twice(dbl, 3) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "12");
}

// ---------------------------------------------------------------------------
// 8. Function reference assigned to val then passed to higher-order fn
// ---------------------------------------------------------------------------
#[test]
fn test_fn_ref_stored_then_passed() {
    let src = r#"
def half(x: i64) -> i64 { x / 2 }
def apply(g: (i64) -> i64, x: i64) -> i64 { g(x) }
def f() -> i64 {
    val fn_half = half
    apply(fn_half, 100)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "50");
}
