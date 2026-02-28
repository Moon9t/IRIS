// Phase 80: Loss functions and training — mse_loss, cross_entropy, list_axpy, sgd_step

use iris::compile;
use iris::EmitKind;

fn eval(src: &str) -> String {
    compile(src, "test", EmitKind::Eval).expect("compile failed")
}

fn eval_f(src: &str) -> f64 {
    eval(src).trim().parse().unwrap()
}

// mse_loss(pred, target) = mean((pred - target)^2)
#[test]
fn test_mse_loss_zero() {
    // same arrays → mse = 0
    let v = eval_f(r#"
def main() -> f64 {
    val a = fill(4, 1.0)
    val b = fill(4, 1.0)
    mse_loss(a, b)
}
"#);
    assert!(v.abs() < 1e-9, "mse of identical arrays should be 0, got {v}");
}

#[test]
fn test_mse_loss_known() {
    // pred=[2], target=[0] => mse = 4.0
    let v = eval_f(r#"
def main() -> f64 {
    val p = list()
    push(p, 2.0);
    val t = list()
    push(t, 0.0);
    mse_loss(p, t)
}
"#);
    assert!((v - 4.0).abs() < 1e-9, "expected 4.0, got {v}");
}

// cross_entropy(pred_probs, targets) = -mean(target * log(pred + eps))
#[test]
fn test_cross_entropy_nonnegative() {
    let v = eval_f(r#"
def main() -> f64 {
    val p = linspace(0.1, 0.9, 5)
    val t = fill(5, 0.5)
    cross_entropy(p, t)
}
"#);
    assert!(v >= 0.0, "cross-entropy must be non-negative, got {v}");
}

#[test]
fn test_cross_entropy_decreases_with_better_pred() {
    // better predictions → lower cross entropy
    let good = eval_f(r#"
def main() -> f64 {
    val p = fill(3, 0.9)
    val t = fill(3, 1.0)
    cross_entropy(p, t)
}
"#);
    let bad = eval_f(r#"
def main() -> f64 {
    val p = fill(3, 0.1)
    val t = fill(3, 1.0)
    cross_entropy(p, t)
}
"#);
    assert!(good < bad, "better preds should give lower CE: good={good}, bad={bad}");
}

// list_axpy(alpha, x, y) = [alpha*x[i] + y[i] for each i]  (BLAS y = alpha*x + y)
#[test]
fn test_list_axpy_basic() {
    // 2*[1,2,3] + [0,0,0] = [2,4,6]; sum=12
    let v = eval_f(r#"
def main() -> f64 {
    val x = linspace(1.0, 3.0, 3)
    val y = zeros(3)
    val z = list_axpy(2.0, x, y)
    list_sum(z)
}
"#);
    assert!((v - 12.0).abs() < 1e-6, "expected 12.0, got {v}");
}

#[test]
fn test_list_axpy_adds_y() {
    // 1*[1,1,1] + [2,2,2] = [3,3,3]; sum=9
    let v = eval_f(r#"
def main() -> f64 {
    val x = ones(3)
    val y = fill(3, 2.0)
    val z = list_axpy(1.0, x, y)
    list_sum(z)
}
"#);
    assert!((v - 9.0).abs() < 1e-6, "expected 9.0, got {v}");
}

// sgd_step(params, grads, lr) updates params in place: params[i] -= lr * grads[i]
#[test]
fn test_sgd_step_decreases_params() {
    // params=[2,2,2], grads=[1,1,1], lr=0.5 → params=[1.5, 1.5, 1.5]; sum=4.5
    let v = eval_f(r#"
def main() -> f64 {
    var params = fill(3, 2.0)
    val grads  = ones(3)
    sgd_step(params, grads, 0.5);
    list_sum(params)
}
"#);
    assert!((v - 4.5).abs() < 1e-6, "expected 4.5, got {v}");
}

#[test]
fn test_sgd_step_zero_grad() {
    // zero gradients → params unchanged; sum=6
    let v = eval_f(r#"
def main() -> f64 {
    var params = fill(3, 2.0)
    val grads  = zeros(3)
    sgd_step(params, grads, 1.0);
    list_sum(params)
}
"#);
    assert!((v - 6.0).abs() < 1e-6, "expected 6.0, got {v}");
}
