// Phase 77: Numeric array creation — zeros, ones, fill, linspace, arange

use iris::compile;
use iris::EmitKind;

fn eval(src: &str) -> String {
    compile(src, "test", EmitKind::Eval).expect("compile failed")
}

// zeros(n) -> list of n 0.0 values
#[test]
fn test_zeros_length() {
    let out = eval(r#"
def main() -> i64 {
    val z = zeros(5)
    list_len(z)
}
"#);
    assert_eq!(out.trim(), "5");
}

#[test]
fn test_zeros_values() {
    let out = eval(r#"
def main() -> f64 {
    val z = zeros(3)
    list_get(z, 1)
}
"#);
    assert_eq!(out.trim(), "0");
}

// ones(n) -> list of n 1.0 values
#[test]
fn test_ones_length() {
    let out = eval(r#"
def main() -> i64 {
    val o = ones(4)
    list_len(o)
}
"#);
    assert_eq!(out.trim(), "4");
}

#[test]
fn test_ones_values() {
    let out = eval(r#"
def main() -> f64 {
    val o = ones(3)
    list_get(o, 2)
}
"#);
    assert_eq!(out.trim(), "1");
}

// fill(n, v) -> list of n copies of v
#[test]
fn test_fill_length() {
    let out = eval(r#"
def main() -> i64 {
    val f = fill(6, 7.0)
    list_len(f)
}
"#);
    assert_eq!(out.trim(), "6");
}

#[test]
fn test_fill_value() {
    let out = eval(r#"
def main() -> f64 {
    val f = fill(4, 3.5)
    list_get(f, 0)
}
"#);
    assert_eq!(out.trim(), "3.5");
}

// linspace(start, stop, n) -> n evenly-spaced values
#[test]
fn test_linspace_endpoints() {
    let out = eval(r#"
def main() -> f64 {
    val ls = linspace(0.0, 1.0, 5)
    list_get(ls, 4)
}
"#);
    let v: f64 = out.trim().parse().unwrap();
    assert!((v - 1.0).abs() < 1e-9, "expected ~1.0, got {v}");
}

// arange(start, stop, step) -> list [start, start+step, ...]
#[test]
fn test_arange_count() {
    let out = eval(r#"
def main() -> i64 {
    val a = arange(0.0, 5.0, 1.0)
    list_len(a)
}
"#);
    assert_eq!(out.trim(), "5");
}
