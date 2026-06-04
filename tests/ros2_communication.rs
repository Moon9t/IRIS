//! Integration tests for ROS2 Client Library and DDS wrappers.

use iris::compile_multi;
use iris::EmitKind;

#[test]
fn test_ros2_stdlib_bring_and_compilation() {
    let src = r#"
bring std.ros2

def main() -> i64 {
    // Assert we can parse and call basic setup
    val node = ros2.create_node("test_iris_node", "sandbox");
    val pub_handle = ros2.create_publisher(node, "sensor_readings", "Float64");
    
    // Publish mock sensor feedback loop data
    val sent = ros2.publish_float64(pub_handle, 42.15);
    
    0
}
"#;
    // Compiling to LLVM IR to verify that the AST parses, merges, lowers,
    // and successfully completes optimization passes without compile errors.
    let res = compile_multi(&[("main", src)], "main", EmitKind::Llvm);
    assert!(
        res.is_ok(),
        "ROS2 client library wrapper failed to compile: {:?}",
        res.err()
    );
}
