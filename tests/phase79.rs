// Phase 79: Elementwise ops — list_add, list_sub, list_mul_elem, list_scale,
//                              list_relu, list_sigmoid, list_softmax

use iris::compile;
use iris::EmitKind;

fn eval(src: &str) -> String {
    compile(src, "test", EmitKind::Eval).expect("compile failed")
}

fn eval_f(src: &str) -> f64 {
    eval(src).trim().parse().unwrap()
}

#[test]
fn test_list_add() {
    // [1,2,3] + [4,5,6] = [5,7,9]; sum = 21
    let v = eval_f(r#"
def main() -> f64 {
    val a = linspace(1.0, 3.0, 3)
    val b = linspace(4.0, 6.0, 3)
    val c = list_add(a, b)
    list_sum(c)
}
"#);
    assert!((v - 21.0).abs() < 1e-6, "expected 21.0, got {v}");
}

#[test]
fn test_list_sub() {
    // [5,5,5] - [1,2,3] = [4,3,2]; sum = 9
    let v = eval_f(r#"
def main() -> f64 {
    val a = fill(3, 5.0)
    val b = linspace(1.0, 3.0, 3)
    val c = list_sub(a, b)
    list_sum(c)
}
"#);
    assert!((v - 9.0).abs() < 1e-6, "expected 9.0, got {v}");
}

#[test]
fn test_list_mul_elem() {
    // [1,2,3] * [1,2,3] = [1,4,9]; sum = 14
    let v = eval_f(r#"
def main() -> f64 {
    val a = linspace(1.0, 3.0, 3)
    val b = linspace(1.0, 3.0, 3)
    val c = list_mul_elem(a, b)
    list_sum(c)
}
"#);
    assert!((v - 14.0).abs() < 1e-6, "expected 14.0, got {v}");
}

#[test]
fn test_list_scale() {
    // [1,2,3] * 2 = [2,4,6]; sum = 12
    let v = eval_f(r#"
def main() -> f64 {
    val a = linspace(1.0, 3.0, 3)
    val b = list_scale(a, 2.0)
    list_sum(b)
}
"#);
    assert!((v - 12.0).abs() < 1e-6, "expected 12.0, got {v}");
}

#[test]
fn test_list_relu_positive() {
    // relu([1,2,3]) = [1,2,3]
    let v = eval_f(r#"
def main() -> f64 {
    val a = linspace(1.0, 3.0, 3)
    val b = list_relu(a)
    list_sum(b)
}
"#);
    assert!((v - 6.0).abs() < 1e-6, "expected 6.0, got {v}");
}

#[test]
fn test_list_relu_negative_zeroed() {
    // relu([-1, 0, 1]) = [0, 0, 1]; sum = 1
    let v = eval_f(r#"
def main() -> f64 {
    val a = list()
    push(a, -1.0);
    push(a, 0.0);
    push(a, 1.0);
    val b = list_relu(a)
    list_sum(b)
}
"#);
    assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
}

#[test]
fn test_list_sigmoid_range() {
    // sigmoid outputs in (0, 1)
    let v = eval_f(r#"
def main() -> f64 {
    val a = linspace(-2.0, 2.0, 5)
    val b = list_sigmoid(a)
    list_sum(b)
}
"#);
    // sum of sigmoid(-2..2) is symmetric around 0.5*5=2.5
    assert!(v > 0.0 && v < 5.0, "sigmoid sum out of range: {v}");
    assert!((v - 2.5).abs() < 0.5, "sigmoid sum too far from 2.5: {v}");
}

#[test]
fn test_list_softmax_sums_to_one() {
    let v = eval_f(r#"
def main() -> f64 {
    val a = linspace(1.0, 4.0, 4)
    val b = list_softmax(a)
    list_sum(b)
}
"#);
    assert!((v - 1.0).abs() < 1e-6, "softmax should sum to 1.0, got {v}");
}
