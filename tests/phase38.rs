//! Phase 38 integration tests: type aliases — `type Name = Type`.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Scalar alias: type Meters = f64
// ---------------------------------------------------------------------------
#[test]
fn test_alias_scalar() {
    let src = r#"
type Meters = f64

def f() -> Meters {
    3.14
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f64 = out.trim().parse().expect("should parse as f64");
    assert!((v - 3.14).abs() < 1e-9, "expected 3.14, got {}", v);
}

// ---------------------------------------------------------------------------
// 2. Alias appears in function parameter
// ---------------------------------------------------------------------------
#[test]
fn test_alias_in_param() {
    let src = r#"
type Score = i64

def double_score(s: Score) -> Score {
    s * 2
}
def f() -> Score {
    double_score(21)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "expected 42, got {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. Alias in IR text contains the resolved type, not the alias name
// ---------------------------------------------------------------------------
#[test]
fn test_alias_ir_resolved() {
    let src = r#"
type Counter = i64

def f() -> Counter {
    0
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should emit IR");
    // The IR should use the concrete type (i64), not the alias name "Counter"
    assert!(out.contains("i64"), "IR should use resolved type i64, got:\n{}", out);
}

// ---------------------------------------------------------------------------
// 4. Multiple aliases in one file
// ---------------------------------------------------------------------------
#[test]
fn test_multiple_aliases() {
    let src = r#"
type Width  = i64
type Height = i64

def area(w: Width, h: Height) -> i64 {
    w * h
}
def f() -> i64 {
    area(6, 7)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "expected 42, got {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. Alias for f32
// ---------------------------------------------------------------------------
#[test]
fn test_alias_f32() {
    let src = r#"
type Loss = f32

def f() -> Loss {
    0.5
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    let v: f32 = out.trim().parse().expect("should parse as f32");
    assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {}", v);
}

// ---------------------------------------------------------------------------
// 6. Alias for bool
// ---------------------------------------------------------------------------
#[test]
fn test_alias_bool() {
    let src = r#"
type Flag = bool

def f() -> Flag {
    true
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "true", "expected true, got {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. Alias mixed with const
// ---------------------------------------------------------------------------
#[test]
fn test_alias_with_const() {
    let src = r#"
type Count = i64
const MAX: Count = 100

def f() -> Count {
    MAX
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "100", "expected 100, got {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. Alias in LLVM stub uses resolved type
// ---------------------------------------------------------------------------
#[test]
fn test_alias_llvm() {
    let src = r#"
type Index = i64

def f() -> Index {
    0
}
"#;
    let out = compile(src, "test", EmitKind::Llvm).expect("should emit LLVM stub");
    assert!(out.contains("i64"), "LLVM stub should use i64, got:\n{}", out);
}
