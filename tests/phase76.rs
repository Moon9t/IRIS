//! Phase 76 integration tests: IR binary serialization / deserialization.
//!
//! Tests cover:
//! - Non-empty serialized output with magic header
//! - Deterministic (identical bytes for same module)
//! - Function count encoded in header
//! - Round-trip: function names survive serialize→deserialize
//! - Round-trip: eval result matches after serialize→deserialize
//! - Different modules produce different bytes
//! - Module name encoded in bytes
//! - Larger module produces larger serialized output

use iris::{compile_to_module, deserialize_module, eval_ir_module, serialize_module};

// ---------------------------------------------------------------------------
// 1. Serialized output is non-empty and starts with IRIS magic
// ---------------------------------------------------------------------------
#[test]
fn test_magic_header() {
    let src = "def f() -> i64 { 42 }";
    let module = compile_to_module(src, "test").unwrap();
    let bytes = serialize_module(&module);
    assert!(!bytes.is_empty(), "serialized bytes should not be empty");
    assert_eq!(&bytes[0..4], b"IRIS", "first 4 bytes must be magic 'IRIS'");
}

// ---------------------------------------------------------------------------
// 2. Version byte is 1
// ---------------------------------------------------------------------------
#[test]
fn test_version_byte() {
    let src = "def f() -> i64 { 42 }";
    let module = compile_to_module(src, "test").unwrap();
    let bytes = serialize_module(&module);
    assert_eq!(bytes[4], 1, "version byte must be 1");
}

// ---------------------------------------------------------------------------
// 3. Serialization is deterministic
// ---------------------------------------------------------------------------
#[test]
fn test_deterministic() {
    let src = "def f() -> i64 { 42 }";
    let module = compile_to_module(src, "test").unwrap();
    let bytes1 = serialize_module(&module);
    let bytes2 = serialize_module(&module);
    assert_eq!(bytes1, bytes2, "serialize must be deterministic");
}

// ---------------------------------------------------------------------------
// 4. Module name survives round-trip
// ---------------------------------------------------------------------------
#[test]
fn test_module_name_round_trip() {
    let src = "def f() -> i64 { 42 }";
    let module = compile_to_module(src, "my_module").unwrap();
    let bytes = serialize_module(&module);
    let module2 = deserialize_module(&bytes).expect("deserialization failed");
    assert_eq!(module2.name, "my_module");
}

// ---------------------------------------------------------------------------
// 5. Function names survive round-trip
// ---------------------------------------------------------------------------
#[test]
fn test_function_name_round_trip() {
    let src = r#"
def compute() -> i64 { 99 }
def f() -> i64 { 99 }
"#;
    let module = compile_to_module(src, "test").unwrap();
    let bytes = serialize_module(&module);
    let module2 = deserialize_module(&bytes).expect("deserialization failed");
    let names: Vec<&str> = module2.functions().iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"compute"), "function 'compute' should survive round-trip");
    assert!(names.contains(&"f"), "function 'f' should survive round-trip");
}

// ---------------------------------------------------------------------------
// 6. Eval result matches after round-trip
// ---------------------------------------------------------------------------
#[test]
fn test_eval_round_trip() {
    let src = "def f() -> i64 { 42 }";
    let module = compile_to_module(src, "test").unwrap();
    let orig_out = eval_ir_module(&module).unwrap();

    let bytes = serialize_module(&module);
    let module2 = deserialize_module(&bytes).expect("deserialization failed");
    let rt_out = eval_ir_module(&module2).unwrap();

    assert_eq!(orig_out.trim(), "42");
    assert_eq!(rt_out.trim(), orig_out.trim(), "eval result must match after round-trip");
}

// ---------------------------------------------------------------------------
// 7. Two different modules produce different bytes
// ---------------------------------------------------------------------------
#[test]
fn test_different_modules_different_bytes() {
    let src1 = "def f() -> i64 { 1 }";
    let src2 = "def f() -> i64 { 2 }";
    let m1 = compile_to_module(src1, "test").unwrap();
    let m2 = compile_to_module(src2, "test").unwrap();
    assert_ne!(serialize_module(&m1), serialize_module(&m2));
}

// ---------------------------------------------------------------------------
// 8. Larger module produces more bytes
// ---------------------------------------------------------------------------
#[test]
fn test_larger_module_more_bytes() {
    let src_small = "def f() -> i64 { 42 }";
    let src_large = r#"
def helper(x: i64) -> i64 { x * 2 }
def f() -> i64 {
    val a = 10
    val b = 20
    helper(a) + helper(b)
}
"#;
    let m_small = compile_to_module(src_small, "test").unwrap();
    let m_large = compile_to_module(src_large, "test").unwrap();
    let small_bytes = serialize_module(&m_small).len();
    let large_bytes = serialize_module(&m_large).len();
    assert!(large_bytes > small_bytes,
        "larger module ({} bytes) should produce more bytes than small ({} bytes)",
        large_bytes, small_bytes);
}
