use iris::runtime_bindings::tensor_pair_from_slices;
use iris::runtime_bindings::tensor_pair_to_slices;
use iris::stdlib::stdlib_source;

#[test]
fn stdlib_sql_module_exports_connector_helpers() {
    let src = stdlib_source("sql").expect("sql stdlib source");
    assert!(src.contains("sql_open"));
    assert!(src.contains("sql_exec"));
    assert!(src.contains("sql_query"));
    assert!(src.contains("sql_close"));
}

#[test]
fn tensor_pair_bridge_roundtrip() {
    let pair = tensor_pair_from_slices(&[0.0, 1.5, 2.0, 3.25], &[2, 2]).expect("pair");
    let (data, shape) = tensor_pair_to_slices(pair).expect("roundtrip");
    assert_eq!(data, vec![0.0, 1.5, 2.0, 3.25]);
    assert_eq!(shape, vec![2, 2]);
}
