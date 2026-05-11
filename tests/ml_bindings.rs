use iris::stdlib::stdlib_source;
use iris::runtime_bindings::tensor_pair_from_slices;
use iris::runtime_bindings::tensor_pair_to_slices;

#[test]
fn stdlib_ml_module_exports_backend_wrappers() {
    let src = stdlib_source("ml").expect("ml stdlib source");
    assert!(src.contains("iris_ml_onnx_run"));
    assert!(src.contains("iris_ml_pytorch_run"));
    assert!(src.contains("iris_ml_tf_run"));
}

#[test]
fn tensor_pair_bridge_roundtrip() {
    let pair = tensor_pair_from_slices(&[0.0, 1.5, 2.0, 3.25], &[2, 2]).expect("pair");
    let (data, shape) = tensor_pair_to_slices(pair).expect("roundtrip");
    assert_eq!(data, vec![0.0, 1.5, 2.0, 3.25]);
    assert_eq!(shape, vec![2, 2]);
}
