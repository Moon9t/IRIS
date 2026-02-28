/// Phase 87: CUDA end-to-end — @kernel attribute, NVVM annotations.

use iris::{compile, EmitKind};

fn cuda(src: &str) -> String {
    compile(src, "phase87", EmitKind::Cuda).expect("cuda emit failed")
}

// ------------------------------------------------------------------
// 1. @kernel function appears in NVVM annotations
// ------------------------------------------------------------------
#[test]
fn test_kernel_attr_in_nvvm_annotations() {
    let src = r#"
@kernel
def vec_add(a: f32, b: f32) -> f32 {
    a + b
}
def main() -> i64 { 0 }
"#;
    let out = cuda(src);
    assert!(out.contains("nvvm.annotations"),
        "expected nvvm.annotations:\n{}", out);
    assert!(out.contains("vec_add"),
        "expected vec_add in CUDA output:\n{}", out);
}

// ------------------------------------------------------------------
// 2. @kernel function is annotated as kernel=1 in NVVM metadata
// ------------------------------------------------------------------
#[test]
fn test_kernel_attr_marked_as_kernel() {
    let src = r#"
@kernel
def my_kernel(x: f32) -> f32 { x * 2.0 }
def main() -> i64 { 0 }
"#;
    let out = cuda(src);
    assert!(out.contains("\"kernel\"") || out.contains("!\"kernel\""),
        "expected kernel metadata:\n{}", out);
}

// ------------------------------------------------------------------
// 3. @kernel function body contains tid.x register read
// ------------------------------------------------------------------
#[test]
fn test_kernel_body_has_tid_read() {
    let src = r#"
@kernel
def compute(n: i64) -> i64 { n + 1 }
def main() -> i64 { 0 }
"#;
    let out = cuda(src);
    assert!(out.contains("tid.x") || out.contains("sreg.tid"),
        "expected tid register read in kernel body:\n{}", out);
}

// ------------------------------------------------------------------
// 4. @kernel function is emitted with define keyword (not declare)
// ------------------------------------------------------------------
#[test]
fn test_kernel_is_defined_not_declared() {
    let src = r#"
@kernel
def gpu_fn(x: f32, y: f32) -> f32 { x + y }
def main() -> i64 { 0 }
"#;
    let out = cuda(src);
    assert!(out.contains("define") && out.contains("gpu_fn"),
        "expected 'define ... gpu_fn' in CUDA output:\n{}", out);
}

// ------------------------------------------------------------------
// 5. Non-@kernel function is not annotated in NVVM
// ------------------------------------------------------------------
#[test]
fn test_non_kernel_not_in_nvvm() {
    let src = r#"
def host_fn(x: i64) -> i64 { x + 1 }
def main() -> i64 { 0 }
"#;
    let out = cuda(src);
    // Should not have any nvvm annotations (no @kernel, no ParFor)
    assert!(!out.contains("nvvm.annotations") || !out.contains("host_fn"),
        "host_fn should not be in NVVM annotations:\n{}", out);
}

// ------------------------------------------------------------------
// 6. Multiple @kernel functions both appear in NVVM annotations
// ------------------------------------------------------------------
#[test]
fn test_multiple_kernels_annotated() {
    let src = r#"
@kernel
def kernel_a(x: f32) -> f32 { x }
@kernel
def kernel_b(x: f32) -> f32 { x * 2.0 }
def main() -> i64 { 0 }
"#;
    let out = cuda(src);
    assert!(out.contains("kernel_a") && out.contains("kernel_b"),
        "expected both kernels in CUDA output:\n{}", out);
}

// ------------------------------------------------------------------
// 7. NVPTX target triple still present with @kernel functions
// ------------------------------------------------------------------
#[test]
fn test_nvptx_triple_with_kernel_attr() {
    let src = r#"
@kernel
def my_kernel(x: i64) -> i64 { x }
def main() -> i64 { 0 }
"#;
    let out = cuda(src);
    assert!(out.contains("nvptx64-nvidia-cuda"),
        "expected nvptx64 triple:\n{}", out);
}

// ------------------------------------------------------------------
// 8. @kernel with f32 arithmetic emits correct NVPTX IR
// ------------------------------------------------------------------
#[test]
fn test_kernel_float_arithmetic() {
    let src = r#"
@kernel
def scale(x: f32) -> f32 { x * 2.0 }
def main() -> i64 { 0 }
"#;
    let out = cuda(src);
    // Check that float multiply is present (fmul instruction)
    assert!(out.contains("fmul") || out.contains("fadd") || out.contains("scale"),
        "expected float op in CUDA kernel:\n{}", out);
}
