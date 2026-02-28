/// Phase 86: Operator overloading via trait impls.
///
/// Uses the existing `impl Trait for Type` infrastructure (phase 46).
/// Operators `+`, `-`, `*`, `/` on record types dispatch to
/// `Add__Type__add`, `Sub__Type__sub`, etc.

use iris::{compile, EmitKind};

fn eval(src: &str) -> String {
    compile(src, "phase86", EmitKind::Eval).expect("eval failed")
}

fn ir(src: &str) -> String {
    compile(src, "phase86", EmitKind::Ir).expect("ir failed")
}

// ------------------------------------------------------------------
// 1. Operator + on a record dispatches to Add__Vec2__add
// ------------------------------------------------------------------
#[test]
fn test_add_dispatches_to_impl() {
    let src = r#"
record Vec2 { x: f64, y: f64 }

impl Add for Vec2 {
    def add(a: Vec2, b: Vec2) -> Vec2 {
        Vec2 { x: a.x + b.x, y: a.y + b.y }
    }
}

def main() -> f64 {
    val u = Vec2 { x: 1.0, y: 2.0 }
    val v = Vec2 { x: 3.0, y: 4.0 }
    val w = u + v
    w.x
}
"#;
    let v: f64 = eval(src).trim().parse().unwrap();
    assert!((v - 4.0).abs() < 1e-9, "expected 4.0, got {v}");
}

// ------------------------------------------------------------------
// 2. y-component of vector addition
// ------------------------------------------------------------------
#[test]
fn test_add_y_component() {
    let src = r#"
record Vec2 { x: f64, y: f64 }

impl Add for Vec2 {
    def add(a: Vec2, b: Vec2) -> Vec2 {
        Vec2 { x: a.x + b.x, y: a.y + b.y }
    }
}

def main() -> f64 {
    val u = Vec2 { x: 1.0, y: 2.0 }
    val v = Vec2 { x: 3.0, y: 4.0 }
    val w = u + v
    w.y
}
"#;
    let v: f64 = eval(src).trim().parse().unwrap();
    assert!((v - 6.0).abs() < 1e-9, "expected 6.0, got {v}");
}

// ------------------------------------------------------------------
// 3. Operator - dispatches to Sub__Vec2__sub
// ------------------------------------------------------------------
#[test]
fn test_sub_dispatches_to_impl() {
    let src = r#"
record Vec2 { x: f64, y: f64 }

impl Sub for Vec2 {
    def sub(a: Vec2, b: Vec2) -> Vec2 {
        Vec2 { x: a.x - b.x, y: a.y - b.y }
    }
}

def main() -> f64 {
    val u = Vec2 { x: 5.0, y: 8.0 }
    val v = Vec2 { x: 2.0, y: 3.0 }
    val w = u - v
    w.x
}
"#;
    let v: f64 = eval(src).trim().parse().unwrap();
    assert!((v - 3.0).abs() < 1e-9, "expected 3.0, got {v}");
}

// ------------------------------------------------------------------
// 4. Operator * dispatches to Mul__Scale__mul
// ------------------------------------------------------------------
#[test]
fn test_mul_dispatches_to_impl() {
    let src = r#"
record Scale { v: f64 }

impl Mul for Scale {
    def mul(a: Scale, b: Scale) -> Scale {
        Scale { v: a.v * b.v }
    }
}

def main() -> f64 {
    val a = Scale { v: 3.0 }
    val b = Scale { v: 7.0 }
    val c = a * b
    c.v
}
"#;
    let v: f64 = eval(src).trim().parse().unwrap();
    assert!((v - 21.0).abs() < 1e-9, "expected 21.0, got {v}");
}

// ------------------------------------------------------------------
// 5. IR contains call to mangled operator method name
// ------------------------------------------------------------------
#[test]
fn test_ir_shows_operator_call() {
    let src = r#"
record Vec2 { x: f64, y: f64 }

impl Add for Vec2 {
    def add(a: Vec2, b: Vec2) -> Vec2 {
        Vec2 { x: a.x + b.x, y: a.y + b.y }
    }
}

def main() -> f64 {
    val u = Vec2 { x: 1.0, y: 0.0 }
    val v = Vec2 { x: 0.0, y: 1.0 }
    val w = u + v
    w.x
}
"#;
    let ir_text = ir(src);
    assert!(ir_text.contains("Add__Vec2__add"),
        "expected Add__Vec2__add in IR:\n{}", ir_text);
}

// ------------------------------------------------------------------
// 6. Scalar + still works (not overloaded)
// ------------------------------------------------------------------
#[test]
fn test_scalar_add_unchanged() {
    let src = r#"
def main() -> i64 {
    val x = 20
    val y = 22
    x + y
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 42);
}

// ------------------------------------------------------------------
// 7. Multiple operator impls on same type
// ------------------------------------------------------------------
#[test]
fn test_multiple_ops_same_type() {
    let src = r#"
record Num { v: i64 }

impl Add for Num {
    def add(a: Num, b: Num) -> Num { Num { v: a.v + b.v } }
}

impl Sub for Num {
    def sub(a: Num, b: Num) -> Num { Num { v: a.v - b.v } }
}

def main() -> i64 {
    val a = Num { v: 10 }
    val b = Num { v: 3 }
    val c = a + b
    val d = c - b
    d.v
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 10);
}

// ------------------------------------------------------------------
// 8. Chained operator expressions
// ------------------------------------------------------------------
#[test]
fn test_chained_add_ops() {
    let src = r#"
record Vec2 { x: f64, y: f64 }

impl Add for Vec2 {
    def add(a: Vec2, b: Vec2) -> Vec2 {
        Vec2 { x: a.x + b.x, y: a.y + b.y }
    }
}

def main() -> f64 {
    val a = Vec2 { x: 1.0, y: 0.0 }
    val b = Vec2 { x: 2.0, y: 0.0 }
    val c = Vec2 { x: 3.0, y: 0.0 }
    val d = a + b + c
    d.x
}
"#;
    let v: f64 = eval(src).trim().parse().unwrap();
    assert!((v - 6.0).abs() < 1e-9, "expected 6.0, got {v}");
}
