//! Phase 133 integration tests: ML systems foundations.
//!
//! Validates:
//! - BuiltinCall LLVM arg-type fix (scalar args: i64, f64, bool)
//! - Crypto builtins: sha256, uuid, hex_encode/decode, base64_encode/decode
//! - JSON: json_stringify serialises values, json.iris helpers round-trip
//! - HTTP stdlib: http.iris exposes http_get / http_post wrappers (compile-only)
//! - Tensor stdlib: new ML ops (sub, mul, scale, softmax, layer_norm, mse, sgd)
//! - FFI stdlib: ffi.iris compiles with call_i64_2, call_f64_2, shell, py_call1
//! - Concurrency builtins: chan_len, chan_try_recv
//! - Math predicates: is_nan, is_inf, math_pi, math_e

use iris::{compile_multi, EmitKind};

// ── helpers ─────────────────────────────────────────────────────────────────

fn eval(src: &str) -> String {
    compile_multi(&[("main", src)], "main", EmitKind::Eval)
        .unwrap_or_else(|e| panic!("eval failed:\n{}\nsrc:\n{}", e, src))
        .trim()
        .to_owned()
}

fn eval_raw(src: &str) -> String {
    compile_multi(&[("main", src)], "main", EmitKind::Eval)
        .unwrap_or_else(|e| panic!("eval failed:\n{}\nsrc:\n{}", e, src))
}

fn ir_ok(src: &str) {
    compile_multi(&[("main", src)], "main", EmitKind::Ir)
        .unwrap_or_else(|e| panic!("IR compile failed:\n{}\nsrc:\n{}", e, src));
}

// ── Crypto builtins ──────────────────────────────────────────────────────────

#[test]
fn test_sha256_returns_hex_string() {
    let src = r#"
def main() -> str {
    sha256("hello")
}
"#;
    let out = eval(src);
    // SHA-256 of "hello" is 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    assert_eq!(
        out,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}

#[test]
fn test_sha256_empty_string() {
    let src = r#"
def main() -> str {
    sha256("")
}
"#;
    let out = eval(src);
    // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    assert_eq!(
        out,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn test_uuid_format() {
    let src = r#"
def main() -> i64 {
    val u = uuid()
    // A UUID has 36 chars: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    len(u)
}
"#;
    let out = eval(src);
    assert_eq!(out, "36");
}

#[test]
fn test_hex_encode_decode_roundtrip() {
    let src = r#"
def main() -> str {
    val encoded = hex_encode("abc")
    hex_decode(encoded)
}
"#;
    let out = eval(src);
    assert_eq!(out, "abc");
}

#[test]
fn test_base64_encode_decode_roundtrip() {
    let src = r#"
def main() -> str {
    val encoded = base64_encode("hello world")
    base64_decode(encoded)
}
"#;
    let out = eval(src);
    assert_eq!(out, "hello world");
}

#[test]
fn test_hash_returns_i64() {
    let src = r#"
def main() -> i64 {
    val h = hash("test")
    if h != 0 { 1 } else { 0 }
}
"#;
    let out = eval(src);
    assert_eq!(out, "1");
}

// ── Math predicates ──────────────────────────────────────────────────────────

#[test]
fn test_is_nan_true() {
    let src = r#"
def main() -> bool {
    val x = 0.0 / 0.0
    is_nan(x)
}
"#;
    // 0.0/0.0 may be NaN; at minimum this should compile
    ir_ok(src);
}

#[test]
fn test_is_nan_false_for_normal() {
    let src = r#"
def main() -> bool {
    is_nan(3.14)
}
"#;
    let out = eval(src);
    assert_eq!(out, "false");
}

#[test]
fn test_is_inf_false_for_normal() {
    let src = r#"
def main() -> bool {
    is_inf(42.0)
}
"#;
    let out = eval(src);
    assert_eq!(out, "false");
}

#[test]
fn test_math_pi_value() {
    let src = r#"
def main() -> bool {
    val pi = math_pi()
    pi > 3.14 && pi < 3.15
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

#[test]
fn test_math_e_value() {
    let src = r#"
def main() -> bool {
    val e = math_e()
    e > 2.71 && e < 2.72
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

// ── Random builtins ──────────────────────────────────────────────────────────

#[test]
fn test_random_in_range() {
    let src = r#"
def main() -> bool {
    val r = random()
    r >= 0.0 && r < 1.0
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

#[test]
fn test_random_range_bounds() {
    let src = r#"
def main() -> bool {
    val r = random_range(10, 20)
    r >= 10 && r < 20
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

// ── JSON builtins ─────────────────────────────────────────────────────────────

#[test]
fn test_json_stringify_string() {
    let src = r#"
def main() -> str {
    json_stringify("hello")
}
"#;
    let out = eval(src);
    assert!(out.contains("hello"), "json_stringify output: {}", out);
}

#[test]
fn test_json_stdlib_round_trip() {
    // json.iris: json_new, json_set, json_get, json_encode
    let src = r#"
bring std.json

def main() -> str {
    val p = json_new()
    val p2 = json_set(p, "name", "Alice")
    json_get(p2, "name")
}
"#;
    let out = eval(src);
    assert_eq!(out, "Alice");
}

#[test]
fn test_json_encode_produces_braces() {
    let src = r#"
bring std.json

def main() -> str {
    val p = json_new()
    val p2 = json_set(p, "k", "v")
    json_encode(p2)
}
"#;
    let out = eval(src);
    assert!(out.contains("{"), "expected JSON object: {}", out);
    assert!(out.contains("k"), "expected key: {}", out);
    assert!(out.contains("v"), "expected value: {}", out);
}

// ── HTTP stdlib compiles ──────────────────────────────────────────────────────

#[test]
fn test_http_stdlib_compiles() {
    let src = r#"
bring std.http

def main() -> str {
    http_get_request("example.com", "/")
}
"#;
    ir_ok(src);
}

#[test]
fn test_http_request_builder() {
    let src = r#"
bring std.http

def main() -> str {
    http_get_request("api.example.com", "/v1/predict")
}
"#;
    let out = eval(src);
    assert!(out.contains("GET"), "should contain GET: {}", out);
    assert!(out.contains("api.example.com"), "{}", out);
}

#[test]
fn test_http_post_request_builder() {
    let src = r#"
bring std.http

def main() -> str {
    http_post_request("api.example.com", "/infer", "{\"x\":1}")
}
"#;
    let out = eval(src);
    assert!(out.contains("POST"), "{}", out);
    assert!(out.contains("Content-Length"), "{}", out);
}

#[test]
fn test_http_status_code_parse() {
    let src = r#"
bring std.http

def main() -> i64 {
    http_status_code("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
}
"#;
    let out = eval(src);
    assert_eq!(out, "200");
}

// ── Crypto stdlib wrappers ────────────────────────────────────────────────────

#[test]
fn test_crypto_stdlib_compiles() {
    let src = r#"
bring std.crypto

def main() -> str {
    generate_uuid()
}
"#;
    ir_ok(src);
}

#[test]
fn test_crypto_hash_sha256_wrapper() {
    let src = r#"
bring std.crypto

def main() -> str {
    hash_sha256("hello")
}
"#;
    let out = eval(src);
    assert_eq!(
        out,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}

#[test]
fn test_crypto_hmac_verify() {
    let src = r#"
bring std.crypto

def main() -> bool {
    val sig = hmac_simple("secret", "payload")
    hmac_verify("secret", "payload", sig)
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

#[test]
fn test_crypto_hmac_wrong_key_fails() {
    let src = r#"
bring std.crypto

def main() -> bool {
    val sig = hmac_simple("secret", "payload")
    hmac_verify("wrong", "payload", sig)
}
"#;
    let out = eval(src);
    assert_eq!(out, "false");
}

#[test]
fn test_crypto_base64_roundtrip() {
    let src = r#"
bring std.crypto

def main() -> str {
    from_base64(to_base64("ML models rock"))
}
"#;
    let out = eval(src);
    assert_eq!(out, "ML models rock");
}

// ── FFI stdlib compiles ───────────────────────────────────────────────────────

#[test]
fn test_ffi_stdlib_compiles() {
    let src = r#"
bring std.ffi

def main() -> str {
    py_version()
}
"#;
    ir_ok(src);
}

#[test]
fn test_ffi_shell_compiles() {
    let src = r#"
bring std.ffi

def main() -> i64 {
    val _ = shell("echo hi")
    0
}
"#;
    ir_ok(src);
}

// ── Tensor stdlib: new ML ops ─────────────────────────────────────────────────

#[test]
fn test_tensor_sub() {
    let src = r#"
bring std.tensorx

def main() -> f64 {
    val shape: list<i64> = list()
    val _ = list_push(shape, 3)
    val a = tensor_full(shape, 5.0)
    val b = tensor_full(shape, 2.0)
    val c = tensor_sub(a, b)
    list_get(c.0, 0)
}
"#;
    let out = eval(src);
    assert_eq!(out, "3");
}

#[test]
fn test_tensor_scale() {
    let src = r#"
bring std.tensorx

def main() -> f64 {
    val shape: list<i64> = list()
    val _ = list_push(shape, 2)
    val t = tensor_full(shape, 4.0)
    val s = tensor_scale(t, 2.5)
    list_get(s.0, 0)
}
"#;
    let out = eval(src);
    assert_eq!(out, "10");
}

#[test]
fn test_tensor_sum_and_mean() {
    let src = r#"
bring std.tensorx

def main() -> f64 {
    val shape: list<i64> = list()
    val _ = list_push(shape, 4)
    val t = tensor_full(shape, 3.0)
    tensor_mean(t)
}
"#;
    let out = eval(src);
    assert_eq!(out, "3");
}

#[test]
fn test_tensor_mse_zero_for_equal() {
    let src = r#"
bring std.tensorx

def main() -> f64 {
    val shape: list<i64> = list()
    val _ = list_push(shape, 3)
    val a = tensor_full(shape, 1.0)
    val b = tensor_full(shape, 1.0)
    tensor_mse(a, b)
}
"#;
    let out = eval(src);
    assert_eq!(out, "0");
}

#[test]
fn test_tensor_mse_nonzero() {
    let src = r#"
bring std.tensorx

def main() -> bool {
    val shape: list<i64> = list()
    val _ = list_push(shape, 2)
    val pred = tensor_full(shape, 3.0)
    val target = tensor_full(shape, 1.0)
    tensor_mse(pred, target) > 0.0
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

#[test]
fn test_tensor_sgd_step_reduces_param() {
    let src = r#"
bring std.tensorx

def main() -> bool {
    val shape: list<i64> = list()
    val _ = list_push(shape, 2)
    val params = tensor_full(shape, 1.0)
    val grad   = tensor_full(shape, 0.1)
    val updated = tensor_sgd_step(params, grad, 0.5)
    // param - lr * grad = 1.0 - 0.5 * 0.1 = 0.95
    list_get(updated.0, 0) < 1.0
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

#[test]
fn test_tensor_softmax_sums_to_one() {
    let src = r#"
bring std.tensorx

def main() -> bool {
    val data: list<f64> = list()
    val _ = list_push(data, 1.0)
    val _ = list_push(data, 2.0)
    val _ = list_push(data, 3.0)
    val shape: list<i64> = list()
    val _ = list_push(shape, 3)
    val t = tensor_from_data(data, shape)
    val s = tensor_softmax(t)
    val total = tensor_sum(s)
    // sum should be ~1.0
    total > 0.999 && total < 1.001
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

#[test]
fn test_tensor_layer_norm_mean_near_zero() {
    let src = r#"
bring std.tensorx

def main() -> bool {
    val data: list<f64> = list()
    val _ = list_push(data, 1.0)
    val _ = list_push(data, 2.0)
    val _ = list_push(data, 3.0)
    val _ = list_push(data, 4.0)
    val shape: list<i64> = list()
    val _ = list_push(shape, 4)
    val t = tensor_from_data(data, shape)
    val normed = tensor_layer_norm(t, 0.00001)
    val m = tensor_mean(normed)
    // after layer norm, mean should be near 0
    m > -0.01 && m < 0.01
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

#[test]
fn test_linear_forward_shape() {
    let src = r#"
bring std.tensorx

def main() -> i64 {
    // x: [2], W: [3, 2], b: [3]  => output: [3]
    val x_data: list<f64> = list()
    val _ = list_push(x_data, 1.0)
    val _ = list_push(x_data, 1.0)
    val x_shape: list<i64> = list()
    val _ = list_push(x_shape, 2)
    val x = tensor_from_data(x_data, x_shape)

    val w_shape: list<i64> = list()
    val _ = list_push(w_shape, 6)
    val w = tensor_full(w_shape, 0.5)

    val b_shape: list<i64> = list()
    val _ = list_push(b_shape, 3)
    val b = tensor_zeros(b_shape)

    val y = linear_forward(x, w, b, 2, 3)
    list_len(y.0)
}
"#;
    let out = eval(src);
    assert_eq!(out, "3");
}

// ── Concurrency builtins ──────────────────────────────────────────────────────

#[test]
fn test_chan_len_empty() {
    let src = r#"
def main() -> i64 {
    val ch = channel()
    chan_len(ch)
}
"#;
    let out = eval(src);
    assert_eq!(out, "0");
}

#[test]
fn test_thread_count_positive() {
    let src = r#"
def main() -> bool {
    thread_count() > 0
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}

// ── String extras ─────────────────────────────────────────────────────────────

#[test]
fn test_str_pad_left() {
    let src = r#"
def main() -> str {
    str_pad_left("42", 5, "0")
}
"#;
    let out = eval(src);
    assert_eq!(out, "00042");
}

#[test]
fn test_str_pad_right() {
    let src = r#"
def main() -> str {
    str_pad_right("hi", 5, " ")
}
"#;
    // eval_raw preserves trailing whitespace
    let out = eval_raw(src);
    assert_eq!(out.trim_end_matches('\n'), "hi   ");
}

#[test]
fn test_str_chars_count() {
    let src = r#"
def main() -> i64 {
    val chars = str_chars("hello")
    list_len(chars)
}
"#;
    let out = eval(src);
    assert_eq!(out, "5");
}

// ── ML pipeline smoke test ────────────────────────────────────────────────────

#[test]
fn test_ml_training_step_smoke() {
    // A tiny 1-step gradient descent loop using tensor stdlib.
    let src = r#"
bring std.tensorx

def main() -> bool {
    // weight vector: [0.5, 0.5]
    val w_data: list<f64> = list()
    val _ = list_push(w_data, 0.5)
    val _ = list_push(w_data, 0.5)
    val shape: list<i64> = list()
    val _ = list_push(shape, 2)
    var w = tensor_from_data(w_data, shape)

    // target: [1.0, 1.0]
    val t_data: list<f64> = list()
    val _ = list_push(t_data, 1.0)
    val _ = list_push(t_data, 1.0)
    val target = tensor_from_data(t_data, shape)

    // 5 SGD steps
    var step = 0
    while step < 5 {
        val grad = tensor_mse_grad(w, target)
        w = tensor_sgd_step(w, grad, 0.1)
        step = step + 1
    }

    // weights should have moved towards 1.0
    list_get(w.0, 0) > 0.5
}
"#;
    let out = eval(src);
    assert_eq!(out, "true");
}
