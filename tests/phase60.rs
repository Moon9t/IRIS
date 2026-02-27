//! Phase 60: Enum variants with data (Algebraic Data Types).
//!
//! Tests construction and pattern-matching of enum variants that carry
//! payload fields, alongside backward-compatible unit (tag-only) variants.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Zero-field variant construction and pattern match (backward compat)
// ---------------------------------------------------------------------------
#[test]
fn test_unit_variant_compat() {
    let src = r#"
choice Dir { North, South, East, West }
def go() -> i64 {
    val d = Dir.East
    when d {
        Dir.North => 1,
        Dir.South => 2,
        Dir.East  => 3,
        Dir.West  => 4,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should compile and eval");
    assert_eq!(out.trim(), "3", "East should map to 3");
}

// ---------------------------------------------------------------------------
// 2. Single-field variant: Circle(f32)
// ---------------------------------------------------------------------------
#[test]
fn test_single_field_variant() {
    let src = r#"
choice Shape {
    Circle(f32),
    Point,
}
def area() -> f32 {
    val s = Shape.Circle(5.0)
    when s {
        Shape.Circle(r) => r * r,
        Shape.Point => 0.0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should compile and eval");
    // 5.0 * 5.0 = 25.0
    let v: f32 = out.trim().parse().expect("should be a float");
    assert!((v - 25.0_f32).abs() < 0.001, "Circle(5) area should be 25, got {}", v);
}

// ---------------------------------------------------------------------------
// 3. Two-field variant: Rect(f32, f32)
// ---------------------------------------------------------------------------
#[test]
fn test_two_field_variant() {
    let src = r#"
choice Shape {
    Rect(f32, f32),
    Point,
}
def area() -> f32 {
    val s = Shape.Rect(4.0, 3.0)
    when s {
        Shape.Rect(w, h) => w * h,
        Shape.Point => 0.0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should compile and eval");
    // 4.0 * 3.0 = 12.0
    let v: f32 = out.trim().parse().expect("should be a float");
    assert!((v - 12.0_f32).abs() < 0.001, "Rect(4,3) area should be 12, got {}", v);
}

// ---------------------------------------------------------------------------
// 4. Mixed variants (some with data, some without)
// ---------------------------------------------------------------------------
#[test]
fn test_mixed_variants() {
    let src = r#"
choice Msg {
    Quit,
    Move(i64, i64),
    Write(i64),
}
def handle() -> i64 {
    val m = Msg.Move(10, 20)
    when m {
        Msg.Quit    => 0,
        Msg.Move(x, y) => x + y,
        Msg.Write(n)   => n,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should compile and eval");
    assert_eq!(out.trim(), "30", "Move(10, 20) should give 10 + 20 = 30");
}

// ---------------------------------------------------------------------------
// 5. Nested when with variant data extraction
// ---------------------------------------------------------------------------
#[test]
fn test_nested_when_variant_data() {
    let src = r#"
choice Outer {
    A(i64),
    B,
}
choice Inner {
    X,
    Y(i64),
}
def compute() -> i64 {
    val a = Outer.A(42)
    val inner_val = when a {
        Outer.A(v) => v,
        Outer.B    => 0,
    }
    val b = Inner.Y(inner_val)
    when b {
        Inner.X    => -1,
        Inner.Y(n) => n + 1,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should compile and eval");
    assert_eq!(out.trim(), "43", "A(42) then Y(42) + 1 should be 43");
}

// ---------------------------------------------------------------------------
// 6. Variant data in arithmetic expressions
// ---------------------------------------------------------------------------
#[test]
fn test_variant_data_arithmetic() {
    let src = r#"
choice Pair {
    Values(i64, i64),
    Nothing,
}
def diff() -> i64 {
    val p = Pair.Values(100, 37)
    when p {
        Pair.Values(a, b) => a - b,
        Pair.Nothing => 0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should compile and eval");
    assert_eq!(out.trim(), "63", "100 - 37 should be 63");
}

// ---------------------------------------------------------------------------
// 7. IR contains ExtractVariantField for data variant patterns
// ---------------------------------------------------------------------------
#[test]
fn test_ir_contains_extract_variant_field() {
    let src = r#"
choice Wrapper {
    Wrap(i64),
    Empty,
}
def unwrap_it() -> i64 {
    val w = Wrapper.Wrap(99)
    when w {
        Wrapper.Wrap(v) => v,
        Wrapper.Empty   => 0,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should emit IR");
    assert!(
        out.contains("extract_variant_field"),
        "IR should contain extract_variant_field, got:\n{}",
        out
    );
    assert!(
        out.contains("make_variant 0("),
        "IR should contain make_variant with fields, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 8. Existing no-data enum tests still work (backward compat)
// ---------------------------------------------------------------------------
#[test]
fn test_no_data_enum_backward_compat() {
    let src = r#"
choice Season { Spring, Summer, Autumn, Winter }
def season_num() -> i64 {
    val s = Season.Winter
    when s {
        Season.Spring => 0,
        Season.Summer => 1,
        Season.Autumn => 2,
        Season.Winter => 3,
    }
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "3", "Winter should give 3");
}
