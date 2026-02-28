/// Phase 88: TCP network I/O — tcp_connect, tcp_listen, tcp_accept,
/// tcp_read, tcp_write, tcp_close intrinsics.
///
/// Interpreter stubs return sentinel values (fd=-1, str="").
/// Tests verify compile success, IR structure, and stub evaluation.

use iris::{compile, EmitKind};

fn eval(src: &str) -> String {
    compile(src, "phase88", EmitKind::Eval).expect("eval failed")
}

fn ir(src: &str) -> String {
    compile(src, "phase88", EmitKind::Ir).expect("ir failed")
}

fn llvm(src: &str) -> String {
    compile(src, "phase88", EmitKind::Llvm).expect("llvm failed")
}

// ------------------------------------------------------------------
// 1. tcp_connect compiles and returns sentinel fd
// ------------------------------------------------------------------
#[test]
fn test_tcp_connect_compiles() {
    let src = r#"
def main() -> i64 {
    val fd = tcp_connect("localhost", 8080)
    fd
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, -1, "expected sentinel fd -1, got {v}");
}

// ------------------------------------------------------------------
// 2. tcp_listen compiles and returns sentinel fd
// ------------------------------------------------------------------
#[test]
fn test_tcp_listen_compiles() {
    let src = r#"
def main() -> i64 {
    val listener = tcp_listen(9090)
    listener
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, -1);
}

// ------------------------------------------------------------------
// 3. tcp_accept compiles and returns sentinel fd
// ------------------------------------------------------------------
#[test]
fn test_tcp_accept_compiles() {
    let src = r#"
def main() -> i64 {
    val listener = tcp_listen(9091)
    val conn = tcp_accept(listener)
    conn
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, -1);
}

// ------------------------------------------------------------------
// 4. tcp_read returns empty string stub
// ------------------------------------------------------------------
#[test]
fn test_tcp_read_returns_empty_str() {
    let src = r#"
def main() -> i64 {
    val conn = tcp_connect("host", 1234)
    val s = tcp_read(conn)
    len(s)
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 0, "expected empty string (len=0), got {v}");
}

// ------------------------------------------------------------------
// 5. tcp_write and tcp_close are side-effecting (no crash)
// ------------------------------------------------------------------
#[test]
fn test_tcp_write_close_no_crash() {
    let src = r#"
def main() -> i64 {
    val conn = tcp_connect("host", 80)
    val _ = tcp_write(conn, "GET / HTTP/1.0")
    val _ = tcp_close(conn)
    42
}
"#;
    let v: i64 = eval(src).trim().parse().unwrap();
    assert_eq!(v, 42);
}

// ------------------------------------------------------------------
// 6. IR contains tcp_connect instruction
// ------------------------------------------------------------------
#[test]
fn test_ir_contains_tcp_connect() {
    let src = r#"
def main() -> i64 {
    val fd = tcp_connect("x", 80)
    fd
}
"#;
    let ir_text = ir(src);
    assert!(ir_text.contains("tcp_connect"),
        "expected tcp_connect in IR:\n{}", ir_text);
}

// ------------------------------------------------------------------
// 7. LLVM IR contains iris_tcp_connect declare/call
// ------------------------------------------------------------------
#[test]
fn test_llvm_contains_tcp_connect() {
    let src = r#"
def main() -> i64 {
    val fd = tcp_connect("host", 8080)
    fd
}
"#;
    let llvm_text = llvm(src);
    assert!(llvm_text.contains("iris_tcp_connect"),
        "expected iris_tcp_connect in LLVM IR:\n{}", llvm_text);
}

// ------------------------------------------------------------------
// 8. Full pipeline: listen → accept → read → write → close in IR
// ------------------------------------------------------------------
#[test]
fn test_ir_contains_all_tcp_ops() {
    let src = r#"
def main() -> i64 {
    val listener = tcp_listen(7777)
    val conn = tcp_accept(listener)
    val msg = tcp_read(conn)
    val _ = tcp_write(conn, msg)
    val _ = tcp_close(conn)
    val _ = tcp_close(listener)
    0
}
"#;
    let ir_text = ir(src);
    assert!(ir_text.contains("tcp_listen"), "missing tcp_listen");
    assert!(ir_text.contains("tcp_accept"), "missing tcp_accept");
    assert!(ir_text.contains("tcp_read"), "missing tcp_read");
    assert!(ir_text.contains("tcp_write"), "missing tcp_write");
    assert!(ir_text.contains("tcp_close"), "missing tcp_close");
}
