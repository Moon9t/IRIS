//! Phase 134 integration tests: native-only execution, HTTP server stdlib, regex.
//!
//! All tests run through LLVM compile+execute (EmitKind::Eval now compiles to
//! a native binary and captures stdout — the interpreter is not used).
//!
//! Validates:
//! - Native execution path works for all prior test categories
//! - http_server.iris: router_new/add/match, response helpers
//! - Regex builtins: regex_match, regex_find_all, regex_replace
//! - Large-system patterns: HTTP routing, concurrent data structures

use iris::{compile_multi, EmitKind};

// ── helpers ──────────────────────────────────────────────────────────────────

fn eval(src: &str) -> String {
    compile_multi(&[("main", src)], "main", EmitKind::Eval)
        .unwrap_or_else(|e| panic!("eval failed:\n{}\nsrc:\n{}", e, src))
        .trim()
        .to_owned()
}

fn ir_ok(src: &str) {
    compile_multi(&[("main", src)], "main", EmitKind::Ir)
        .unwrap_or_else(|e| panic!("IR compile failed:\n{}\nsrc:\n{}", e, src));
}

// ── Native execution smoke tests ─────────────────────────────────────────────
// These verify the LLVM compile+run path works end-to-end.

#[test]
fn native_hello_world() {
    let src = r#"
def main() -> str {
    "hello from native"
}
"#;
    assert_eq!(eval(src), "hello from native");
}

#[test]
fn native_arithmetic() {
    let src = r#"
def main() -> i64 {
    val x = 6 * 7
    x
}
"#;
    assert_eq!(eval(src), "42");
}

#[test]
fn native_string_ops() {
    let src = r#"
def main() -> str {
    val s = concat("foo", "bar")
    to_upper(s)
}
"#;
    assert_eq!(eval(src), "FOOBAR");
}

#[test]
fn native_list_operations() {
    let src = r#"
def main() -> i64 {
    val xs: list<i64> = list()
    val _ = list_push(xs, 10)
    val _ = list_push(xs, 20)
    val _ = list_push(xs, 30)
    list_get(xs, 1)
}
"#;
    assert_eq!(eval(src), "20");
}

#[test]
fn native_closures() {
    let src = r#"
def apply(f: |i64| -> i64, x: i64) -> i64 {
    f(x)
}

def main() -> i64 {
    apply(|n: i64| n * n, 7)
}
"#;
    assert_eq!(eval(src), "49");
}

#[test]
fn native_recursion_fibonacci() {
    let src = r#"
def fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}

def main() -> i64 {
    fib(10)
}
"#;
    assert_eq!(eval(src), "55");
}

#[test]
fn native_option_some_none() {
    let src = r#"
def maybe_double(x: i64) -> option<i64> {
    if x > 0 { some(x * 2) } else { none }
}

def main() -> i64 {
    val r = maybe_double(5)
    if is_some(r) { unwrap(r) } else { -1 }
}
"#;
    assert_eq!(eval(src), "10");
}

#[test]
fn native_print_goes_to_stdout() {
    // Verify that print() output is captured through the native path.
    let src = r#"
def main() -> i64 {
    print("line one");
    print("line two");
    0
}
"#;
    let out = eval(src);
    assert!(out.contains("line one"), "got: {}", out);
    assert!(out.contains("line two"), "got: {}", out);
}

#[test]
fn native_print_inlined_string_result() {
    let src = r#"
def greet(name: str) -> str {
    concat("Hello, ", name)
}

def main() -> i64 {
    print(greet("World"));
    0
}
"#;
    let out = eval(src);
    assert!(out.contains("Hello, World"), "got: {}", out);
}

#[test]
fn native_inline_struct_field_access() {
    let src = r#"
record Pair {
    left: i64,
    right: i64,
}

def make_pair(a: i64, b: i64) -> Pair {
    Pair { left: a, right: b }
}

def bump(p: Pair) -> Pair {
    Pair { left: p.left + 1, right: p.right + 2 }
}

def main() -> i64 {
    val p = make_pair(10, 20)
    val q = bump(p)
    q.left + q.right
}
"#;
    assert_eq!(eval(src), "33");
}

#[test]
fn native_returned_list_survives_call_boundary() {
    let src = r#"
def make_values() -> list<i64> {
    val xs: list<i64> = list()
    val _ = list_push(xs, 7)
    val _ = list_push(xs, 9)
    xs
}

def main() -> i64 {
    list_get(make_values(), 1)
}
"#;
    assert_eq!(eval(src), "9");
}

#[test]
fn native_returned_list_alias_survives_call_boundary() {
    let src = r#"
def choose(flag: bool) -> list<i64> {
    val xs: list<i64> = list()
    val _ = list_push(xs, 4)
    val _ = list_push(xs, 8)
    val alias = if flag { xs } else { xs }
    alias
}

def main() -> i64 {
    list_get(choose(true), 1)
}
"#;
    assert_eq!(eval(src), "8");
}

#[test]
fn native_returned_struct_with_heap_field_survives_call_boundary() {
    let src = r#"
record Boxed {
    xs: list<i64>,
}

def build() -> Boxed {
    val xs: list<i64> = list()
    val _ = list_push(xs, 5)
    val _ = list_push(xs, 12)
    Boxed { xs: xs }
}

def main() -> i64 {
    list_get(build().xs, 1)
}
"#;
    assert_eq!(eval(src), "12");
}

// ── HTTP server stdlib ────────────────────────────────────────────────────────

#[test]
fn http_server_stdlib_compiles() {
    let src = r#"
bring std.http_server
bring std.http

def main() -> str {
    json_ok("{\"status\":\"ok\"}")
}
"#;
    ir_ok(src);
}

#[test]
fn http_server_json_ok_response() {
    let src = r#"
bring std.http_server

def main() -> bool {
    val resp = json_ok("{\"x\":1}")
    starts_with(resp, "HTTP/1.1 200")
}
"#;
    assert_eq!(eval(src), "true");
}

#[test]
fn http_server_text_ok_response() {
    let src = r#"
bring std.http_server

def main() -> bool {
    val resp = text_ok("pong")
    starts_with(resp, "HTTP/1.1 200") && contains(resp, "pong")
}
"#;
    assert_eq!(eval(src), "true");
}

#[test]
fn http_server_bad_request_response() {
    let src = r#"
bring std.http_server

def main() -> bool {
    val resp = bad_request("invalid input")
    starts_with(resp, "HTTP/1.1 400")
}
"#;
    assert_eq!(eval(src), "true");
}

#[test]
fn http_server_internal_error_response() {
    let src = r#"
bring std.http_server

def main() -> bool {
    val resp = internal_error("something broke")
    starts_with(resp, "HTTP/1.1 500")
}
"#;
    assert_eq!(eval(src), "true");
}

#[test]
fn http_server_router_match_hit() {
    let src = r#"
bring std.http_server

def main() -> str {
    var r = router_new()
    r = router_add(r, "GET",  "/health",       "health_handler")
    r = router_add(r, "POST", "/api/predict",  "predict_handler")
    r = router_add(r, "GET",  "/api/models",   "list_models_handler")
    router_match(r, "POST", "/api/predict")
}
"#;
    assert_eq!(eval(src), "predict_handler");
}

#[test]
fn http_server_router_match_miss() {
    let src = r#"
bring std.http_server

def main() -> str {
    var r = router_new()
    r = router_add(r, "GET", "/health", "health_handler")
    router_match(r, "GET", "/not_found")
}
"#;
    assert_eq!(eval(src), "");
}

#[test]
fn http_server_router_method_mismatch() {
    let src = r#"
bring std.http_server

def main() -> str {
    var r = router_new()
    r = router_add(r, "GET", "/data", "get_data")
    r = router_add(r, "POST", "/data", "post_data")
    router_match(r, "DELETE", "/data")
}
"#;
    assert_eq!(eval(src), "");
}

#[test]
fn http_server_complete_routing_flow() {
    // Simulate a full request dispatch cycle.
    let src = r#"
bring std.http_server
bring std.http

def dispatch(r: list<str>, req: str) -> str {
    val method = http_request_method(req)
    val path   = http_request_path(req)
    val tag    = router_match(r, method, path)
    if tag == "health" {
        json_ok("{\"status\":\"ok\"}")
    } else if tag == "predict" {
        json_ok("{\"prediction\":0.9}")
    } else {
        bad_request("unknown route")
    }
}

def main() -> bool {
    var r = router_new()
    r = router_add(r, "GET",  "/health",  "health")
    r = router_add(r, "POST", "/predict", "predict")

    val req1 = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
    val req2 = "POST /predict HTTP/1.1\r\nHost: localhost\r\n\r\n{}"
    val req3 = "DELETE /health HTTP/1.1\r\nHost: localhost\r\n\r\n"

    val r1 = dispatch(r, req1)
    val r2 = dispatch(r, req2)
    val r3 = dispatch(r, req3)

    starts_with(r1, "HTTP/1.1 200") &&
    starts_with(r2, "HTTP/1.1 200") &&
    starts_with(r3, "HTTP/1.1 400")
}
"#;
    assert_eq!(eval(src), "true");
}

// ── Regex builtins ────────────────────────────────────────────────────────────

#[test]
fn regex_match_literal() {
    let src = r#"
def main() -> bool {
    regex_match("hello", "hello world")
}
"#;
    assert_eq!(eval(src), "true");
}

#[test]
fn regex_match_no_match() {
    let src = r#"
def main() -> bool {
    regex_match("xyz", "hello world")
}
"#;
    assert_eq!(eval(src), "false");
}

#[test]
fn regex_match_wildcard() {
    let src = r#"
def main() -> bool {
    regex_match("h.llo", "hello")
}
"#;
    assert_eq!(eval(src), "true");
}

#[test]
fn regex_replace_basic() {
    let src = r#"
def main() -> str {
    regex_replace("world", "hello world", "IRIS")
}
"#;
    let out = eval(src);
    assert!(out.contains("IRIS"), "expected IRIS in: {}", out);
}

#[test]
fn regex_find_all_returns_list() {
    let src = r#"
def main() -> i64 {
    val matches = regex_find_all("o", "foo boo too")
    list_len(matches)
}
"#;
    let out = eval(src);
    // should find at least one 'o'
    let n: i64 = out.parse().unwrap_or(0);
    assert!(n >= 1, "expected at least 1 match, got {}", out);
}

// ── Large-scale pattern: ML inference server skeleton ────────────────────────

#[test]
fn ml_inference_server_compiles() {
    // An inference server that routes /predict → runs a linear model,
    // /health → 200 OK. Compile-only test (no actual TCP binding).
    let src = r#"
bring std.http_server
bring std.http
bring std.json
bring std.tensorx

def linear_predict(x: f64) -> f64 {
    val shape: list<i64> = list()
    val _ = list_push(shape, 1)
    val w = tensor_full(shape, 0.5)
    val b = tensor_full(shape, 0.1)
    val x_t = tensor_full(shape, x)
    val out = tensor_add(tensor_mul(w, x_t), b)
    list_get(out.0, 0)
}

def handle_request(req: str) -> str {
    val method = http_request_method(req)
    val path   = http_request_path(req)
    if method == "GET" && path == "/health" {
        json_ok("{\"status\":\"ok\"}")
    } else if method == "POST" && path == "/predict" {
        val body = http_body(req)
        val params = json_parse(body)
        val x_str = json_get(params, "x")
        // For this smoke test just return a static prediction
        json_ok("{\"prediction\":0.9}")
    } else {
        bad_request("unknown route")
    }
}

def main() -> i64 {
    // Smoke-test the handler without starting a server
    val req = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
    val resp = handle_request(req)
    if starts_with(resp, "HTTP/1.1 200") { 0 } else { 1 }
}
"#;
    assert_eq!(eval(src), "0");
}

// ── Map / KV data structure ───────────────────────────────────────────────────

#[test]
fn map_builtin_works_natively() {
    let src = r#"
def main() -> i64 {
    val m = map()
    val _ = map_set(m, "a", 1)
    val _ = map_set(m, "b", 2)
    val _ = map_set(m, "c", 3)
    map_len(m)
}
"#;
    assert_eq!(eval(src), "3");
}

#[test]
fn concurrent_channel_send_recv() {
    let src = r#"
def main() -> i64 {
    val ch = channel()
    spawn {
        send(ch, 42);
    };
    val v = recv(ch)
    v
}
"#;
    assert_eq!(eval(src), "42");
}
