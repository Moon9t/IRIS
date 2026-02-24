//! Minimal hand-rolled Protocol Buffers binary encoder.
//!
//! Supports only the subset of proto3 needed to emit ONNX ModelProto:
//! - Wire type 0 (varint): i32, i64, u64 as LEB-128
//! - Wire type 2 (length-delimited): strings and embedded messages
//!
//! Wire format: tag = (field_number << 3) | wire_type

/// Encode a 64-bit unsigned integer as a LEB-128 varint.
pub fn encode_varint(mut v: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        } else {
            out.push(byte | 0x80);
        }
    }
    out
}

/// Encode a proto field tag: (field_number << 3) | wire_type.
fn encode_tag(field: u32, wire_type: u8) -> Vec<u8> {
    encode_varint(((field as u64) << 3) | (wire_type as u64))
}

/// Encode a varint field (wire type 0).
pub fn encode_varint_field(field: u32, v: u64) -> Vec<u8> {
    let mut out = encode_tag(field, 0);
    out.extend(encode_varint(v));
    out
}

/// Encode a length-delimited field (wire type 2) with raw bytes.
pub fn encode_len_field(field: u32, data: &[u8]) -> Vec<u8> {
    let mut out = encode_tag(field, 2);
    out.extend(encode_varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

/// Encode a string field (wire type 2, UTF-8).
pub fn encode_string_field(field: u32, s: &str) -> Vec<u8> {
    encode_len_field(field, s.as_bytes())
}

/// Encode an embedded message field (wire type 2).
pub fn encode_message_field(field: u32, msg: &[u8]) -> Vec<u8> {
    encode_len_field(field, msg)
}
