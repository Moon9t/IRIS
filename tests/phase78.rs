// Phase 78: Array reductions — list_sum, list_mean, list_max_val, list_min_val, list_std, list_norm, list_dot

use iris::compile;
use iris::EmitKind;

fn eval(src: &str) -> String {
    compile(src, "test", EmitKind::Eval).expect("compile failed")
}

fn eval_f(src: &str) -> f64 {
    eval(src).trim().parse().unwrap()
}

#[test]
fn test_list_sum() {
    let v = eval_f(r#"
def main() -> f64 {
    val xs = ones(4)
    list_sum(xs)
}
"#);
    assert!((v - 4.0).abs() < 1e-9, "expected 4.0, got {v}");
}

#[test]
fn test_list_mean() {
    let v = eval_f(r#"
def main() -> f64 {
    val xs = fill(5, 2.0)
    list_mean(xs)
}
"#);
    assert!((v - 2.0).abs() < 1e-9, "expected 2.0, got {v}");
}

#[test]
fn test_list_max_val() {
    let v = eval_f(r#"
def main() -> f64 {
    val xs = linspace(1.0, 5.0, 5)
    list_max_val(xs)
}
"#);
    assert!((v - 5.0).abs() < 1e-9, "expected 5.0, got {v}");
}

#[test]
fn test_list_min_val() {
    let v = eval_f(r#"
def main() -> f64 {
    val xs = linspace(1.0, 5.0, 5)
    list_min_val(xs)
}
"#);
    assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {v}");
}

#[test]
fn test_list_std_constant() {
    // std of constant array = 0
    let v = eval_f(r#"
def main() -> f64 {
    val xs = fill(4, 3.0)
    list_std(xs)
}
"#);
    assert!(v.abs() < 1e-6, "expected ~0.0 std for constant array, got {v}");
}

#[test]
fn test_list_norm() {
    // norm of [3,4] = 5
    let v = eval_f(r#"
def main() -> f64 {
    val xs = list()
    push(xs, 3.0);
    push(xs, 4.0);
    list_norm(xs)
}
"#);
    assert!((v - 5.0).abs() < 1e-6, "expected 5.0, got {v}");
}

#[test]
fn test_list_dot() {
    // [1,2,3] · [4,5,6] = 4+10+18 = 32
    let v = eval_f(r#"
def main() -> f64 {
    val a = list()
    push(a, 1.0);
    push(a, 2.0);
    push(a, 3.0);
    val b = list()
    push(b, 4.0);
    push(b, 5.0);
    push(b, 6.0);
    list_dot(a, b)
}
"#);
    assert!((v - 32.0).abs() < 1e-9, "expected 32.0, got {v}");
}

#[test]
fn test_list_sum_zeros() {
    let v = eval_f(r#"
def main() -> f64 {
    val xs = zeros(10)
    list_sum(xs)
}
"#);
    assert!(v.abs() < 1e-9, "expected 0.0, got {v}");
}
