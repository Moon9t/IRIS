/// Phase 82: BLAS bindings — matmul, list_dot_blas, list_axpy_blas, list_scale_blas

use iris::{compile, EmitKind};

fn eval(src: &str) -> String {
    compile(src, "phase82", EmitKind::Eval).expect("eval failed")
}

fn eval_f(src: &str) -> f64 {
    eval(src).trim().parse().unwrap_or_else(|e| panic!("parse failed: {e}\n{}", eval(src)))
}

// ------------------------------------------------------------------
// 1. matmul: 2×2 @ 2×2 identity → same matrix, sum = 10
// ------------------------------------------------------------------
#[test]
fn test_matmul_identity() {
    let v = eval_f(r#"
def main() -> f64 {
    val a = list()
    push(a, 1.0);
    push(a, 2.0);
    push(a, 3.0);
    push(a, 4.0);
    val id = list()
    push(id, 1.0);
    push(id, 0.0);
    push(id, 0.0);
    push(id, 1.0);
    val c = matmul(a, 2, 2, id, 2)
    list_sum(c)
}
"#);
    // A@I = A = [1,2,3,4], sum=10
    assert!((v - 10.0).abs() < 1e-9, "expected 10.0, got {v}");
}

// ------------------------------------------------------------------
// 2. matmul: 1×2 @ 2×1 → 1×1 (dot product = 25)
// ------------------------------------------------------------------
#[test]
fn test_matmul_dot_product() {
    let v = eval_f(r#"
def main() -> f64 {
    val a = list()
    push(a, 3.0);
    push(a, 4.0);
    val b = list()
    push(b, 3.0);
    push(b, 4.0);
    val c = matmul(a, 1, 2, b, 1)
    list_get(c, 0)
}
"#);
    assert!((v - 25.0).abs() < 1e-9, "expected 25.0, got {v}");
}

// ------------------------------------------------------------------
// 3. list_dot_blas = 32
// ------------------------------------------------------------------
#[test]
fn test_list_dot_blas() {
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
    list_dot_blas(a, b)
}
"#);
    // 1*4 + 2*5 + 3*6 = 32
    assert!((v - 32.0).abs() < 1e-9, "expected 32.0, got {v}");
}

// ------------------------------------------------------------------
// 4. list_axpy_blas: r[i] = alpha*x[i] + y[i], sum = 36
// ------------------------------------------------------------------
#[test]
fn test_list_axpy_blas() {
    let v = eval_f(r#"
def main() -> f64 {
    val x = list()
    push(x, 1.0);
    push(x, 2.0);
    val y = list()
    push(y, 10.0);
    push(y, 20.0);
    val r = list_axpy_blas(2.0, x, y)
    list_sum(r)
}
"#);
    // alpha=2, x=[1,2], y=[10,20] → [12,24], sum=36
    assert!((v - 36.0).abs() < 1e-9, "expected 36.0, got {v}");
}

// ------------------------------------------------------------------
// 5. list_scale_blas: r[i] = x[i] * alpha, sum = 18
// ------------------------------------------------------------------
#[test]
fn test_list_scale_blas() {
    let v = eval_f(r#"
def main() -> f64 {
    val x = list()
    push(x, 1.0);
    push(x, 2.0);
    push(x, 3.0);
    val r = list_scale_blas(x, 3.0)
    list_sum(r)
}
"#);
    // [1,2,3]*3 = [3,6,9], sum=18
    assert!((v - 18.0).abs() < 1e-9, "expected 18.0, got {v}");
}

// ------------------------------------------------------------------
// 6. matmul 2×3 @ 3×2 → C[0,0] = 58
// ------------------------------------------------------------------
#[test]
fn test_matmul_2x3_3x2() {
    let v = eval_f(r#"
def main() -> f64 {
    val a = list()
    push(a, 1.0); push(a, 2.0); push(a, 3.0);
    push(a, 4.0); push(a, 5.0); push(a, 6.0);
    val b = list()
    push(b, 7.0); push(b, 8.0);
    push(b, 9.0); push(b, 10.0);
    push(b, 11.0); push(b, 12.0);
    val c = matmul(a, 2, 3, b, 2)
    list_get(c, 0)
}
"#);
    // C[0,0] = 1*7+2*9+3*11 = 58
    assert!((v - 58.0).abs() < 1e-9, "expected 58.0, got {v}");
}

// ------------------------------------------------------------------
// 7. list_dot_blas of zeros = 0
// ------------------------------------------------------------------
#[test]
fn test_list_dot_blas_zeros() {
    let v = eval_f(r#"
def main() -> f64 {
    val a = zeros(5)
    val b = ones(5)
    list_dot_blas(a, b)
}
"#);
    assert!(v.abs() < 1e-9, "expected 0.0, got {v}");
}

// ------------------------------------------------------------------
// 8. matmul result has correct length m*n = 12
// ------------------------------------------------------------------
#[test]
fn test_matmul_result_length() {
    let result = eval(r#"
def main() -> i64 {
    val a = fill(6, 1.0)
    val b = fill(8, 1.0)
    val c = matmul(a, 3, 2, b, 4)
    list_len(c)
}
"#);
    assert_eq!(result.trim(), "12", "expected length 12, got: {}", result);
}
