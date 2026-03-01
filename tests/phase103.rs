//! Phase 103 integration tests: printf formatting and generic constraints.

use iris::{compile_multi, EmitKind};

// ── 1. sprintf %d passes value through ──────────────────────────────────────
#[test]
fn test_sprintf_d() {
    let src = r#"
bring std.fmt
def f() -> str {
    var args = list()
    push(args, to_str(42));
    sprintf("%d", args)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 2. sprintf %05d zero-pads to 5 digits ────────────────────────────────────
#[test]
fn test_sprintf_zero_pad() {
    let src = r#"
bring std.fmt
def f() -> str {
    var args = list()
    push(args, to_str(7));
    sprintf("%05d", args)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "00007");
}

// ── 3. sprintf %.3f truncates/rounds float ───────────────────────────────────
#[test]
fn test_sprintf_float_prec() {
    let src = r#"
bring std.fmt
def f() -> str {
    var args = list()
    push(args, to_str(3.14159));
    sprintf("%.3f", args)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3.142");
}

// ── 4. sprintf %-8s left-aligns string ──────────────────────────────────────
#[test]
fn test_sprintf_left_align() {
    let src = r#"
bring std.fmt
def f() -> str {
    var args = list()
    push(args, "hi");
    concat(sprintf("%-8s", args), "|")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hi      |");
}

// ── 5. Generic max function [T where T: Ord] with i64 ────────────────────────
#[test]
fn test_generic_max_i64() {
    let src = r#"
def max_val[T where T: Ord](a: T, b: T) -> T {
    if a > b { a } else { b }
}
def f() -> i64 {
    max_val(10, 25)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "25");
}

// ── 6. Generic max function works with str ───────────────────────────────────
#[test]
fn test_generic_max_str() {
    let src = r#"
def max_str[T where T: Ord](a: T, b: T) -> T {
    if a > b { a } else { b }
}
def f() -> str {
    max_str("apple", "banana")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "banana");
}

// ── 7. Generic equality check [T where T: Eq] ────────────────────────────────
#[test]
fn test_generic_eq() {
    let src = r#"
def is_equal[T where T: Eq](a: T, b: T) -> bool {
    a == b
}
def f() -> bool {
    is_equal(42, 42)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "true");
}

// ── 8. sprintf with literal prefix/suffix text ──────────────────────────────
#[test]
fn test_sprintf_mixed_text() {
    let src = r#"
bring std.fmt
def f() -> str {
    var args = list()
    push(args, to_str(7));
    push(args, "world");
    sprintf("val=%d msg=%s!", args)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "val=7 msg=world!");
}
