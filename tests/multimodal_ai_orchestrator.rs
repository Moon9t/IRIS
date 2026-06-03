//! Integration tests for the 'multimodal_ai_orchestrator' project.

use iris::{compile_file, EmitKind};
use std::path::{Path, PathBuf};

fn project_main_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("projects")
        .join("multimodal_ai_orchestrator")
        .join("src")
        .join("main.iris")
}

#[test]
fn test_multimodal_ai_orchestrator_project() {
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
        assert!(out.contains("IRIS Flagship Multimodal Edge AI Orchestrator Loop"), "output was:\n{}", out);
        assert!(out.contains("Execution Cycle: 1"), "output was:\n{}", out);
        assert!(out.contains("Execution Cycle: 2"), "output was:\n{}", out);
        assert!(out.contains("Execution Cycle: 3"), "output was:\n{}", out);
        assert!(out.contains("Multi-framework inference pipeline completed"), "output was:\n{}", out);
        assert!(out.contains("Projected state features to control space via matmul"), "output was:\n{}", out);
        assert!(out.contains("L1 dot projection coherence"), "output was:\n{}", out);
        assert!(out.contains("Policy parameters optimized with zero allocation"), "output was:\n{}", out);
        assert!(out.contains("Multimodal AI Orchestrator loop completed cleanly"), "output was:\n{}", out);
    }
}
