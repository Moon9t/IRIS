//! Phase 114 integration tests: record methods and advanced record usage.

use iris::{compile, EmitKind};

// ── 1. Basic record creation and field access ───────────────────────────────
#[test]
fn test_struct_field_access() {
    let src = r#"
record Point { x: i64, y: i64 }
def f() -> i64 {
    val p = Point { x: 10, y: 20 }
    p.x + p.y
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "30");
}

// ── 2. Record passed to function ────────────────────────────────────────────
#[test]
fn test_struct_as_param() {
    let src = r#"
record Vec2 { x: i64, y: i64 }
def magnitude_sq(v: Vec2) -> i64 {
    v.x * v.x + v.y * v.y
}
def f() -> i64 {
    val v = Vec2 { x: 3, y: 4 }
    magnitude_sq(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "25");
}

// ── 3. Record returned from function ────────────────────────────────────────
#[test]
fn test_struct_return() {
    let src = r#"
record Pair { a: i64, b: i64 }
def make_pair(x: i64, y: i64) -> Pair {
    Pair { a: x, b: y }
}
def f() -> i64 {
    val p = make_pair(3, 7)
    p.a + p.b
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 4. Record with method (impl block) ──────────────────────────────────────
#[test]
fn test_struct_method() {
    let src = r#"
record Counter { count: i64 }
impl Counter {
    def get(self: Counter) -> i64 { self.count }
}
def f() -> i64 {
    val c = Counter { count: 42 }
    c.get()
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 5. Record with method using field computation ───────────────────────────
#[test]
fn test_struct_method_computation() {
    let src = r#"
record Rect { w: i64, h: i64 }
impl Rect {
    def area(self: Rect) -> i64 { self.w * self.h }
}
def f() -> i64 {
    val r = Rect { w: 5, h: 8 }
    r.area()
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "40");
}

// ── 6. Multiple record instances ────────────────────────────────────────────
#[test]
fn test_multiple_struct_instances() {
    let src = r#"
record Point { x: i64, y: i64 }
def f() -> i64 {
    val p1 = Point { x: 1, y: 2 }
    val p2 = Point { x: 3, y: 4 }
    p1.x + p1.y + p2.x + p2.y
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 7. Record with string field ─────────────────────────────────────────────
#[test]
fn test_struct_string_field() {
    let src = r#"
record Person { name: str, age: i64 }
def f() -> str {
    val p = Person { name: "Alice", age: 30 }
    p.name
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Alice");
}

// ── 8. Record with boolean field ────────────────────────────────────────────
#[test]
fn test_struct_bool_field() {
    let src = r#"
record Flag { active: bool, value: i64 }
def f() -> i64 {
    val fl = Flag { active: true, value: 99 }
    if fl.active { fl.value } else { 0 }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "99");
}
