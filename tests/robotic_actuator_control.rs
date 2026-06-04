//! Integration tests for the 'robotic_actuator_control' project.

use iris::{compile_file, EmitKind};
use std::path::{Path, PathBuf};

fn project_main_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("projects")
        .join("robotic_actuator_control")
        .join("src")
        .join("main.iris")
}

#[test]
fn test_robotic_actuator_control_project() {
    if std::env::var("AMENT_PREFIX_PATH").is_err() {
        std::env::set_var("AMENT_PREFIX_PATH", "C:\\dev\\ros2_humble\\ros2-windows");
    }
    if let Ok(old_path) = std::env::var("PATH") {
        let new_path = format!(
            "C:\\dev\\ros2_humble\\ros2-windows\\bin;\
             C:\\onnxruntime\\lib;\
             C:\\tensorflow\\lib;\
             C:\\openblas\\bin;{}",
            old_path
        );
        std::env::set_var("PATH", new_path);
    }

    let out = compile_file(&project_main_path(), EmitKind::Eval).unwrap();

    // Verify execution output matches either the JIT captured output or the interpreter return value "0"
    if out.trim() != "0" {
        assert!(
            out.contains("Robotic Joint Actuator Autonomous Learning Loop"),
            "output was:\n{}",
            out
        );
        assert!(out.contains("Step: 1"), "output was:\n{}", out);
        assert!(out.contains("Step: 2"), "output was:\n{}", out);
        assert!(out.contains("Step: 3"), "output was:\n{}", out);
        assert!(
            out.contains("Robotic Autonomous Joint learning loop completed successfully"),
            "output was:\n{}",
            out
        );
    }
}
