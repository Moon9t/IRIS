//! Phase 95: Stdlib Core — split/join builtins, std.math, std.string, std.fmt.

use iris::{compile_multi, EmitKind};

// ── 1. split("a,b,c", ",") returns a list of 3 elements ────────────────────

#[test]
fn test_split_three_elements() {
    // Use list_len to count; list_get to access.
    let src = r#"
def f() -> i64 {
    val parts = split("a,b,c", ",")
    len(parts)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 2. join(["a","b","c"], "-") returns "a-b-c" ─────────────────────────────

#[test]
fn test_join_with_delimiter() {
    let src = r#"
def f() -> str {
    val parts = split("a,b,c", ",")
    join(parts, "-")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "a-b-c");
}

// ── 3. bring std.math; gcd(12, 8) = 4 ──────────────────────────────────────

#[test]
fn test_stdlib_math_gcd() {
    let src = r#"
bring std.math
def f() -> i64 { gcd(12, 8) }
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "4");
}

// ── 4. bring std.math; lcm(4, 6) = 12 ──────────────────────────────────────

#[test]
fn test_stdlib_math_lcm() {
    let src = r#"
bring std.math
def f() -> i64 { lcm(4, 6) }
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "12");
}

// ── 5. bring std.string; pad_left("42", 5, "0") = "00042" ──────────────────

#[test]
fn test_stdlib_string_pad_left() {
    let src = r#"
bring std.string
def f() -> str { pad_left("42", 5, "0") }
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "00042");
}

// ── 6. bring std.string; pad_right("hi", 5, ".") = "hi..." ─────────────────

#[test]
fn test_stdlib_string_pad_right() {
    let src = r#"
bring std.string
def f() -> str { pad_right("hi", 5, ".") }
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hi...");
}

// ── 7. bring std.fmt; pad_int(42, 5) = "   42" ──────────────────────────────

#[test]
fn test_stdlib_fmt_pad_int() {
    let src = r#"
bring std.fmt
def f() -> str { pad_int(42, 5) }
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim_end(), "   42");
}

// ── 8. bring std.fmt; zero_pad_int(7, 4) = "0007" ──────────────────────────

#[test]
fn test_stdlib_fmt_zero_pad_int() {
    let src = r#"
bring std.fmt
def f() -> str { zero_pad_int(7, 4) }
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0007");
}
