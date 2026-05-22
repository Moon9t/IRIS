//! Phase 115 integration tests: recursive algorithm patterns.

use iris::{compile, EmitKind};

// ── 1. Factorial recursive ──────────────────────────────────────────────────
#[test]
fn test_recursive_factorial() {
    let src = r#"
def factorial(n: i64) -> i64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}
def f() -> i64 {
    factorial(6)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "720");
}

// ── 2. Fibonacci recursive ──────────────────────────────────────────────────
#[test]
fn test_recursive_fibonacci() {
    let src = r#"
def fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
def f() -> i64 {
    fib(10)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "55");
}

// ── 3. Sum of digits recursive ──────────────────────────────────────────────
#[test]
fn test_recursive_digit_sum() {
    let src = r#"
def digit_sum(n: i64) -> i64 {
    if n < 10 { n } else { n % 10 + digit_sum(n / 10) }
}
def f() -> i64 {
    digit_sum(12345)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15");
}

// ── 4. Power function recursive ─────────────────────────────────────────────
#[test]
fn test_recursive_power() {
    let src = r#"
def power(base: i64, exp: i64) -> i64 {
    if exp == 0 { 1 } else { base * power(base, exp - 1) }
}
def f() -> i64 {
    power(2, 10)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1024");
}

// ── 5. GCD recursive ───────────────────────────────────────────────────────
#[test]
fn test_recursive_gcd() {
    let src = r#"
def gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}
def f() -> i64 {
    gcd(48, 18)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ── 6. Count down recursive ────────────────────────────────────────────────
#[test]
fn test_recursive_countdown() {
    let src = r#"
def count(n: i64) -> i64 {
    if n <= 0 { 0 } else { 1 + count(n - 1) }
}
def f() -> i64 {
    count(10)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 7. Mutual recursion (is_even / is_odd) ──────────────────────────────────
#[test]
fn test_mutual_recursion() {
    let src = r#"
def is_even(n: i64) -> bool {
    if n == 0 { true } else { is_odd(n - 1) }
}
def is_odd(n: i64) -> bool {
    if n == 0 { false } else { is_even(n - 1) }
}
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    bool_to_i64(is_even(10)) + bool_to_i64(is_odd(7))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 8. Sum of range recursive ───────────────────────────────────────────────
#[test]
fn test_recursive_range_sum() {
    let src = r#"
def range_sum(lo: i64, hi: i64) -> i64 {
    if lo > hi { 0 } else { lo + range_sum(lo + 1, hi) }
}
def f() -> i64 {
    range_sum(1, 10)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "55");
}
