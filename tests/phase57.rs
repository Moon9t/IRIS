//! Phase 57 integration tests: struct methods (impl RecordName + method call syntax).
//!
//! Tests cover:
//!  1. Simple struct method call returns a field value
//!  2. Method that takes extra args
//!  3. Method that mutates via returned value (method chaining style)
//!  4. Multiple methods on one impl block
//!  5. Method call as part of an expression
//!  6. Standalone impl produces a correctly mangled function name in IR
//!  7. Method call via EmitKind::Eval executes correctly
//!  8. Method on a struct with multiple fields

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Simple method returns a field value
// ---------------------------------------------------------------------------
#[test]
fn test_method_returns_field() {
    let src = r#"
record Point {
    x: i64,
    y: i64,
}

impl Point {
    def get_x(self: Point) -> i64 {
        self.x
    }
}

def f() -> i64 {
    val p = Point { x: 42, y: 7 }
    p.get_x()
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// 2. Method that takes additional arguments
// ---------------------------------------------------------------------------
#[test]
fn test_method_with_extra_args() {
    let src = r#"
record Counter {
    value: i64,
}

impl Counter {
    def add(self: Counter, n: i64) -> i64 {
        self.value + n
    }
}

def f() -> i64 {
    val c = Counter { value: 10 }
    c.add(5)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15");
}

// ---------------------------------------------------------------------------
// 3. Multiple methods on one impl block
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_methods() {
    let src = r#"
record Rect {
    width: i64,
    height: i64,
}

impl Rect {
    def area(self: Rect) -> i64 {
        self.width * self.height
    }
    def perimeter(self: Rect) -> i64 {
        2 * (self.width + self.height)
    }
}

def f() -> i64 {
    val r = Rect { width: 4, height: 3 }
    r.area()
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "12");
}

// ---------------------------------------------------------------------------
// 4. Method used in a conditional expression
// ---------------------------------------------------------------------------
#[test]
fn test_method_in_expression() {
    let src = r#"
record Box {
    val_field: i64,
}

impl Box {
    def is_positive(self: Box) -> bool {
        self.val_field > 0
    }
}

def f() -> i64 {
    val b = Box { val_field: 5 }
    if b.is_positive() {
        1
    } else {
        0
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ---------------------------------------------------------------------------
// 5. Standalone impl produces mangled name in LLVM IR
// ---------------------------------------------------------------------------
#[test]
fn test_method_mangled_name_in_ir() {
    let src = r#"
record Vec2 {
    x: i64,
    y: i64,
}

impl Vec2 {
    def length_sq(self: Vec2) -> i64 {
        self.x * self.x + self.y * self.y
    }
}

def f() -> i64 {
    val v = Vec2 { x: 3, y: 4 }
    v.length_sq()
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    // The mangled function name should appear in the IR.
    assert!(
        ir.contains("Vec2__length_sq"),
        "expected mangled name 'Vec2__length_sq' in IR:\n{}",
        ir
    );
}

// ---------------------------------------------------------------------------
// 6. Method returning the perimeter (second method test)
// ---------------------------------------------------------------------------
#[test]
fn test_method_perimeter() {
    let src = r#"
record Rect {
    width: i64,
    height: i64,
}

impl Rect {
    def area(self: Rect) -> i64 {
        self.width * self.height
    }
    def perimeter(self: Rect) -> i64 {
        2 * (self.width + self.height)
    }
}

def f() -> i64 {
    val r = Rect { width: 4, height: 3 }
    r.perimeter()
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "14");
}

// ---------------------------------------------------------------------------
// 7. Method with f32 return type
// ---------------------------------------------------------------------------
#[test]
fn test_method_f32_field() {
    let src = r#"
record Circle {
    radius: f32,
}

impl Circle {
    def diameter(self: Circle) -> f32 {
        self.radius * 2.0
    }
}

def f() -> f32 {
    val c = Circle { radius: 5.0 }
    c.diameter()
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert!(
        result.trim().starts_with("10"),
        "expected diameter ~10.0, got: {}",
        result.trim()
    );
}

// ---------------------------------------------------------------------------
// 8. Method call from within another function
// ---------------------------------------------------------------------------
#[test]
fn test_method_call_from_function() {
    let src = r#"
record Score {
    points: i64,
    bonus: i64,
}

impl Score {
    def total(self: Score) -> i64 {
        self.points + self.bonus
    }
}

def compute(s: Score) -> i64 {
    s.total()
}

def f() -> i64 {
    val s = Score { points: 100, bonus: 25 }
    compute(s)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "125");
}
