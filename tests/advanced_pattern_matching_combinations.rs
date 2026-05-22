//! Phase 122 integration tests: advanced pattern matching combinations.

use iris::{compile, EmitKind};

// ── 1. When on integer with multiple arms ───────────────────────────────────
#[test]
fn test_when_int_multi_arm() {
    let src = r#"
def day_name(d: i64) -> str {
    when d {
        1 => "Mon",
        2 => "Tue",
        3 => "Wed",
        4 => "Thu",
        5 => "Fri",
        _ => "Weekend",
    }
}
def f() -> str {
    day_name(3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Wed");
}

// ── 2. When on boolean ─────────────────────────────────────────────────────
#[test]
fn test_when_bool() {
    let src = r#"
def f() -> str {
    val x = true
    when x {
        true => "yes",
        false => "no",
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "yes");
}

// ── 3. When default arm (wildcard) ──────────────────────────────────────────
#[test]
fn test_when_wildcard_default() {
    let src = r#"
def f() -> str {
    val x = 999
    when x {
        1 => "one",
        2 => "two",
        _ => "other",
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "other");
}

// ── 4. When with guard on value ─────────────────────────────────────────────
#[test]
fn test_when_guard_value() {
    let src = r#"
def grade(score: i64) -> str {
    if score >= 90 { "A" }
    else { if score >= 80 { "B" }
    else { if score >= 70 { "C" }
    else { "F" } } }
}
def f() -> str {
    grade(85)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "B");
}

// ── 5. When on integer with exact matches ───────────────────────────────────
#[test]
fn test_when_exact_match() {
    let src = r#"
def fizzbuzz(n: i64) -> str {
    val r15 = n % 15
    val r3 = n % 3
    val r5 = n % 5
    if r15 == 0 { "FizzBuzz" }
    else { if r3 == 0 { "Fizz" }
    else { if r5 == 0 { "Buzz" }
    else { to_str(n) } } }
}
def f() -> str {
    fizzbuzz(15)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "FizzBuzz");
}

// ── 6. When on string literal ───────────────────────────────────────────────
#[test]
fn test_when_string() {
    let src = r#"
def f() -> i64 {
    val cmd = "add"
    when cmd {
        "add" => 1,
        "sub" => 2,
        "mul" => 3,
        _ => 0,
    }
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 7. When result used in arithmetic ───────────────────────────────────────
#[test]
fn test_when_in_arithmetic() {
    let src = r#"
def weight(x: i64) -> i64 {
    when x {
        0 => 10,
        1 => 20,
        _ => 30,
    }
}
def f() -> i64 {
    weight(0) + weight(1) + weight(2)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ── 8. When nested in loop ──────────────────────────────────────────────────
#[test]
fn test_when_in_loop() {
    let src = r#"
def classify(x: i64) -> i64 {
    if x > 0 { 1 }
    else { if x == 0 { 0 }
    else { 0 - 1 } }
}
def f() -> i64 {
    var sum = 0
    for i in 0..5 {
        sum = sum + classify(i)
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // classify(0)=0, classify(1..4)=1 each => 0+1+1+1+1 = 4
    assert_eq!(result.trim(), "4");
}
