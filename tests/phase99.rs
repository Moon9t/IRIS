//! Phase 99 integration tests: stdlib HTTP message parsing and building.

use iris::{compile_multi, EmitKind};

// ── 1. http_request_method parses "GET" ──────────────────────────────────────
#[test]
fn test_http_request_method() {
    let src = r#"
bring std.http
def f() -> str {
    http_request_method("GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "GET");
}

// ── 2. http_status_code extracts 200 ─────────────────────────────────────────
#[test]
fn test_http_status_code() {
    let src = r#"
bring std.http
def f() -> i64 {
    http_status_code("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "200");
}

// ── 3. http_header extracts Content-Type value ───────────────────────────────
#[test]
fn test_http_header_content_type() {
    let src = r#"
bring std.http
def f() -> str {
    val raw = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 5\r\n\r\nhello"
    http_header(raw, "Content-Type")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "text/html");
}

// ── 4. http_body extracts body after \r\n\r\n ─────────────────────────────────
#[test]
fn test_http_body() {
    let src = r#"
bring std.http
def f() -> str {
    http_body("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "hello");
}

// ── 5. http_get_request builds correct request string ────────────────────────
#[test]
fn test_http_get_request() {
    let src = r#"
bring std.http
def f() -> str {
    http_get_request("example.com", "/")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    // trim() strips trailing \r\n\r\n, so only the request line + host header survive
    assert_eq!(result.trim(), "GET / HTTP/1.1\r\nHost: example.com");
}

// ── 6. http_post_request includes Content-Length header ──────────────────────
#[test]
fn test_http_post_request() {
    let src = r#"
bring std.http
def f() -> str {
    val req = http_post_request("example.com", "/api", "hello")
    http_header(req, "Content-Length")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "5");
}

// ── 7. http_response builds response with correct status line ─────────────────
#[test]
fn test_http_response_build() {
    let src = r#"
bring std.http
def f() -> str {
    val resp = http_response(200, "OK", "hello")
    http_status_code(resp)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "200");
}

// ── 8. http_request_path extracts path ───────────────────────────────────────
#[test]
fn test_http_request_path() {
    let src = r#"
bring std.http
def f() -> str {
    http_request_path("POST /api/data HTTP/1.1\r\nHost: example.com\r\n\r\n")
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "/api/data");
}
