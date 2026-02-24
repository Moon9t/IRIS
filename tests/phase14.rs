//! Phase 14 integration tests: binary ONNX protobuf output.

use iris::proto::{encode_string_field, encode_varint};
use iris::{compile, EmitKind};

// Helper: decode hex string back to bytes.
fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

// Minimal model DSL source with one Dense (Gemm) layer.
const MATMUL_MODEL: &str = r#"
model MatMulNet {
    input x: tensor<f32, [1, 4]>
    layer y Dense(units=8)
    output y
}
"#;

// Two-layer model for size check.
const TWO_LAYER_MODEL: &str = r#"
model Net {
    input x: tensor<f32, [1, 8]>
    layer h1 Dense(units=16)
    layer h2 ReLU(h1)
    output h2
}
"#;

// ---------------------------------------------------------------------------
// 1. encode_varint(1) == [0x01]
// ---------------------------------------------------------------------------
#[test]
fn test_proto_varint_1() {
    assert_eq!(encode_varint(1), vec![0x01]);
}

// ---------------------------------------------------------------------------
// 2. encode_varint(300) == [0xAC, 0x02]
// ---------------------------------------------------------------------------
#[test]
fn test_proto_varint_300() {
    // 300 = 0b1_0010_1100
    // LEB-128: low 7 bits = 0101100 (0x2C), set high bit → 0xAC
    //          remaining = 0b10 = 2 → 0x02
    assert_eq!(encode_varint(300), vec![0xAC, 0x02]);
}

// ---------------------------------------------------------------------------
// 3. encode_string_field(1, "hello") has correct tag and content
// ---------------------------------------------------------------------------
#[test]
fn test_proto_string_field() {
    // Field 1, wire type 2 → tag byte = (1 << 3) | 2 = 0x0A
    // Length = 5 → 0x05
    // UTF-8: b'h' b'e' b'l' b'l' b'o'
    let expected = vec![0x0A, 0x05, b'h', b'e', b'l', b'l', b'o'];
    assert_eq!(encode_string_field(1, "hello"), expected);
}

// ---------------------------------------------------------------------------
// 4. Output starts with ModelProto ir_version field (field 1, varint, value 7)
// ---------------------------------------------------------------------------
#[test]
fn test_onnx_binary_starts_with_magic() {
    let hex = compile(MATMUL_MODEL, "test", EmitKind::OnnxBinary)
        .expect("should compile to ONNX binary");
    let bytes = hex_to_bytes(&hex);
    assert!(!bytes.is_empty(), "output should not be empty");
    // ModelProto field 1 (ir_version), wire type 0: tag = 0x08
    // ir_version = 7: encoded as 0x07
    assert_eq!(bytes[0], 0x08, "first byte should be field-1 varint tag (0x08)");
    assert_eq!(bytes[1], 0x07, "second byte should be ir_version = 7");
}

// ---------------------------------------------------------------------------
// 5. MatMul model contains "Gemm" as bytes in NodeProto op_type
// ---------------------------------------------------------------------------
#[test]
fn test_onnx_binary_matmul_node() {
    let hex = compile(MATMUL_MODEL, "test", EmitKind::OnnxBinary)
        .expect("should compile");
    let bytes = hex_to_bytes(&hex);
    // Look for the ASCII bytes of "Gemm" in the output.
    let gemm = b"Gemm";
    let found = bytes.windows(gemm.len()).any(|w| w == gemm);
    assert!(found, "expected 'Gemm' bytes in ONNX output (Linear → Gemm mapping)");
}

// ---------------------------------------------------------------------------
// 6. Output contains the graph name "Linear" or "Net" as bytes
// ---------------------------------------------------------------------------
#[test]
fn test_onnx_binary_has_graph() {
    let hex = compile(MATMUL_MODEL, "test", EmitKind::OnnxBinary)
        .expect("should compile");
    let bytes = hex_to_bytes(&hex);
    // Graph name "MatMulNet" is encoded as a string field in GraphProto (field 2).
    let name = b"MatMulNet";
    let found = bytes.windows(name.len()).any(|w| w == name);
    assert!(found, "expected graph name 'MatMulNet' in ONNX output");
}

// ---------------------------------------------------------------------------
// 7. opset_import has domain="" and version=17
// ---------------------------------------------------------------------------
#[test]
fn test_onnx_binary_opset() {
    let hex = compile(MATMUL_MODEL, "test", EmitKind::OnnxBinary)
        .expect("should compile");
    let bytes = hex_to_bytes(&hex);
    // OperatorSetIdProto.version = 2, varint 17 = 0x11
    // field 2 tag (varint) = (2 << 3) | 0 = 0x10
    // So we expect the sequence [0x10, 0x11] somewhere in the output.
    let opset_version_bytes: &[u8] = &[0x10, 0x11];
    let found = bytes.windows(2).any(|w| w == opset_version_bytes);
    assert!(found, "expected opset version=17 encoded as [0x10, 0x11]");
}

// ---------------------------------------------------------------------------
// 8. Two-layer model produces > 20 bytes
// ---------------------------------------------------------------------------
#[test]
fn test_onnx_binary_roundtrip_size() {
    let hex = compile(TWO_LAYER_MODEL, "test", EmitKind::OnnxBinary)
        .expect("should compile");
    let bytes = hex_to_bytes(&hex);
    assert!(
        bytes.len() > 20,
        "expected > 20 bytes for a 2-node graph, got {}",
        bytes.len()
    );
}
