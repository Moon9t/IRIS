//! Integration tests for TensorRT NPU Acceleration Backend.

use iris::compile_multi;
use iris::EmitKind;

const MODEL_SRC: &str = r#"
model TensorRtNet {
    input x: tensor<f32, [1, 4]>
    layer y Dense(units=8)
    output y
}
"#;

#[test]
fn test_tensorrt_engine_compilation() {
    let res = compile_multi(&[("main", MODEL_SRC)], "main", EmitKind::TensorRt);
    assert!(res.is_ok(), "TensorRT engine compilation failed: {:?}", res.err());
    let trt_out = res.unwrap();
    
    // Assert the output contains structural elements of TensorRT lowering plan
    assert!(trt_out.contains("NVIDIA TensorRT"), "Expected TensorRT header reference");
    assert!(trt_out.contains("tensorrt_add_gemm"), "Expected Gemm layer lowering");
    assert!(trt_out.contains("deserialize_cuda_engine"), "Expected engine deserialization FFI call");
    assert!(trt_out.contains("execution_context_enqueue_v3"), "Expected async JIT execution FFI call");
}
