//! Phase 26 integration tests: Channels (`channel()`, `send()`, `recv()`) and `spawn`.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. channel() compiles to IR containing chan_new
// ---------------------------------------------------------------------------
#[test]
fn test_channel_new_ir() {
    let src = r#"
def f() -> i64 {
    val ch = channel()
    send(ch, 42);
    recv(ch)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("chan_new"),
        "IR should contain chan_new, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 2. send then recv gets the value back
// ---------------------------------------------------------------------------
#[test]
fn test_channel_send_recv_eval() {
    let src = r#"
def f() -> i64 {
    val ch = channel()
    send(ch, 42);
    recv(ch)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "recv after send(42) should be 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 3. channel of i64 values
// ---------------------------------------------------------------------------
#[test]
fn test_channel_i64() {
    let src = r#"
def f() -> i64 {
    val ch = channel()
    send(ch, 100);
    recv(ch)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "100", "channel i64 should return 100, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 4. channel value used in arithmetic
// ---------------------------------------------------------------------------
#[test]
fn test_channel_str() {
    let src = r#"
def f() -> i64 {
    val ch = channel()
    send(ch, 7);
    val v = recv(ch)
    v * 6
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "7 * 6 from channel should be 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 5. spawn block compiles to IR containing spawn
// ---------------------------------------------------------------------------
#[test]
fn test_spawn_ir() {
    let src = r#"
def f() -> i64 {
    val ch = channel()
    spawn {
        send(ch, 99)
    }
    recv(ch)
}
"#;
    let out = compile(src, "test", EmitKind::Ir).expect("should compile to IR");
    assert!(
        out.contains("spawn"),
        "IR should contain spawn, got:\n{}",
        out
    );
}

// ---------------------------------------------------------------------------
// 6. spawned body executes (sends to channel, we recv it)
// ---------------------------------------------------------------------------
#[test]
fn test_spawn_runs_body() {
    let src = r#"
def f() -> i64 {
    val ch = channel()
    spawn {
        send(ch, 42)
    }
    recv(ch)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "42", "spawned body should send 42, recv should get 42, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 7. chan value sent and received (validates channel works end-to-end)
// ---------------------------------------------------------------------------
#[test]
fn test_chan_type_in_sig() {
    let src = r#"
def f() -> i64 {
    val ch = channel()
    send(ch, 55);
    recv(ch)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "55", "channel should return 55, got: {}", out.trim());
}

// ---------------------------------------------------------------------------
// 8. send 3 values, recv 3 values in FIFO order
// ---------------------------------------------------------------------------
#[test]
fn test_channel_multiple_values() {
    let src = r#"
def f() -> i64 {
    val ch = channel()
    send(ch, 1);
    send(ch, 2);
    send(ch, 3);
    val a = recv(ch)
    val b = recv(ch)
    val c = recv(ch)
    a + b * 10 + c * 100
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(out.trim(), "321", "recv order: 1 + 2*10 + 3*100 = 321, got: {}", out.trim());
}
