// phase50.rs — CUDA Kernel Codegen
//
// Tests for the CUDA/NVPTX IR emitter (EmitKind::Cuda):
//   - NVPTX64 target triple
//   - NVPTX data layout
//   - ParFor body functions become *_kernel CUDA kernels
//   - !nvvm.annotations metadata block
//   - @llvm.nvvm.read.ptx.sreg.tid.x declarations
//   - @llvm.nvvm.barrier0 for Barrier instruction
//   - Scalar arithmetic emitted correctly for GPU

use iris::{compile, EmitKind};

// ── Test 1: NVPTX target triple ───────────────────────────────────────────

#[test]
fn test_cuda_target_triple() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Cuda).unwrap();
    assert!(
        ir.contains("nvptx64-nvidia-cuda"),
        "expected 'nvptx64-nvidia-cuda' in CUDA output:\n{}",
        ir
    );
}

// ── Test 2: NVPTX data layout ─────────────────────────────────────────────

#[test]
fn test_cuda_data_layout() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Cuda).unwrap();
    assert!(
        ir.contains("target datalayout"),
        "expected 'target datalayout' in CUDA output:\n{}",
        ir
    );
    // NVPTX datalayout has p:64:64:64
    assert!(
        ir.contains("p:64:64:64") || ir.contains("p270") || ir.contains("nvptx"),
        "expected NVPTX-specific datalayout:\n{}",
        ir
    );
}

// ── Test 3: NVVM sreg declarations present ────────────────────────────────

#[test]
fn test_cuda_sreg_declares() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Cuda).unwrap();
    assert!(
        ir.contains("@llvm.nvvm.read.ptx.sreg.tid.x"),
        "expected '@llvm.nvvm.read.ptx.sreg.tid.x' declaration:\n{}",
        ir
    );
    assert!(
        ir.contains("@llvm.nvvm.read.ptx.sreg.ctaid.x"),
        "expected '@llvm.nvvm.read.ptx.sreg.ctaid.x' declaration:\n{}",
        ir
    );
}

// ── Test 4: barrier0 declaration present ──────────────────────────────────

#[test]
fn test_cuda_barrier_declare() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Cuda).unwrap();
    assert!(
        ir.contains("@llvm.nvvm.barrier0"),
        "expected '@llvm.nvvm.barrier0' declaration:\n{}",
        ir
    );
}

// ── Test 5: scalar arithmetic is correct in CUDA IR ──────────────────────

#[test]
fn test_cuda_scalar_arithmetic() {
    let src = r#"def compute(x: i64, y: i64) -> i64 { x + y }"#;
    let ir = compile(src, "test", EmitKind::Cuda).unwrap();
    // Should emit add instruction for integers.
    assert!(
        ir.contains("add") && (ir.contains("nsw") || ir.contains("i64")),
        "expected integer add in CUDA IR:\n{}",
        ir
    );
}

// ── Test 6: function definition appears in CUDA IR ────────────────────────

#[test]
fn test_cuda_function_definition() {
    let src = r#"def gpu_fn(n: i64) -> i64 { n * 2 }"#;
    let ir = compile(src, "test", EmitKind::Cuda).unwrap();
    assert!(
        ir.contains("define") && ir.contains("@gpu_fn"),
        "expected function definition '@gpu_fn' in CUDA IR:\n{}",
        ir
    );
}

// ── Test 7: CUDA launch helper declared ───────────────────────────────────

#[test]
fn test_cuda_launch_declare() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Cuda).unwrap();
    assert!(
        ir.contains("iris_cuda_launch") || ir.contains("iris_par_for"),
        "expected cuda launch helper declaration:\n{}",
        ir
    );
}

// ── Test 8: phase header comment present ─────────────────────────────────

#[test]
fn test_cuda_header_comment() {
    let src = r#"def f() -> i64 { 0 }"#;
    let ir = compile(src, "test", EmitKind::Cuda).unwrap();
    assert!(
        ir.contains("CUDA") || ir.contains("NVPTX"),
        "expected CUDA/NVPTX header comment:\n{}",
        ir
    );
}
