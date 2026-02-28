/// Phase 89: Mutable closure captures via cell(v)/cell_get(c)/cell_set(c,v).
///
/// Cells are 1-element lists (Rc-backed), so closures that capture a cell
/// share mutations with the outer scope.

use iris::{compile, EmitKind};

fn eval(src: &str) -> String {
    compile(src, "phase89", EmitKind::Eval).expect("eval failed")
}

fn ir(src: &str) -> String {
    compile(src, "phase89", EmitKind::Ir).expect("ir failed")
}

// ------------------------------------------------------------------
// 1. cell(v) creates a mutable cell, cell_get returns initial value
// ------------------------------------------------------------------
#[test]
fn test_cell_get_initial_value() {
    let src = r#"
def main() -> i64 {
    val c = cell(42)
    cell_get(c)
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 42);
}

// ------------------------------------------------------------------
// 2. cell_set updates the value, cell_get returns new value
// ------------------------------------------------------------------
#[test]
fn test_cell_set_and_get() {
    let src = r#"
def main() -> i64 {
    val c = cell(0)
    val _ = cell_set(c, 99)
    cell_get(c)
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 99);
}

// ------------------------------------------------------------------
// 3. Multiple cell_set calls accumulate correctly
// ------------------------------------------------------------------
#[test]
fn test_cell_set_multiple_times() {
    let src = r#"
def main() -> i64 {
    val c = cell(0)
    val _ = cell_set(c, 10)
    val _ = cell_set(c, 20)
    val _ = cell_set(c, 30)
    cell_get(c)
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 30);
}

// ------------------------------------------------------------------
// 4. Closure captures cell and mutates it
// ------------------------------------------------------------------
#[test]
fn test_closure_mutates_cell() {
    let src = r#"
def main() -> i64 {
    val counter = cell(0)
    val inc = |n: i64| {
        val cur = cell_get(counter)
        val _ = cell_set(counter, cur + n)
        cur + n
    }
    val _ = inc(1)
    val _ = inc(2)
    cell_get(counter)
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 3);
}

// ------------------------------------------------------------------
// 5. Two closures share the same cell
// ------------------------------------------------------------------
#[test]
fn test_two_closures_share_cell() {
    let src = r#"
def main() -> i64 {
    val total = cell(0)
    val add5 = |x: i64| {
        val cur = cell_get(total)
        val _ = cell_set(total, cur + 5)
        cur + 5
    }
    val add10 = |x: i64| {
        val cur = cell_get(total)
        val _ = cell_set(total, cur + 10)
        cur + 10
    }
    val _ = add5(0)
    val _ = add10(0)
    cell_get(total)
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 15);
}

// ------------------------------------------------------------------
// 6. cell works with f64 values
// ------------------------------------------------------------------
#[test]
fn test_cell_f64() {
    let src = r#"
def main() -> f64 {
    val c = cell(1.5)
    val cur = cell_get(c)
    val _ = cell_set(c, cur * 2.0)
    cell_get(c)
}
"#;
    let v: f64 = eval(src).trim().parse().unwrap();
    assert!((v - 3.0).abs() < 1e-9, "expected 3.0, got {v}");
}

// ------------------------------------------------------------------
// 7. IR contains ListNew and ListGet for cell operations
// ------------------------------------------------------------------
#[test]
fn test_ir_shows_cell_as_list_ops() {
    let src = r#"
def main() -> i64 {
    val c = cell(7)
    cell_get(c)
}
"#;
    let ir_text = ir(src);
    assert!(ir_text.contains("list_new") || ir_text.contains("ListNew"),
        "expected list_new in IR (cell desugars to list):\n{}", ir_text);
}

// ------------------------------------------------------------------
// 8. cell_set inside a loop accumulates correctly
// ------------------------------------------------------------------
#[test]
fn test_cell_loop_accumulate() {
    let src = r#"
def main() -> i64 {
    val acc = cell(0)
    for i in 0..5 {
        val cur = cell_get(acc)
        val _ = cell_set(acc, cur + i)
    }
    cell_get(acc)
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 10); // 0+1+2+3+4
}
