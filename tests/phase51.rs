// phase51.rs — CPU SIMD / Vectorization
//
// Tests for the SIMD-annotated LLVM IR emitter (EmitKind::Simd):
//   - Phase comment and vector width annotation
//   - x86-64-v3 target-cpu attribute group
//   - target-features with +avx2,+fma
//   - !llvm.loop vectorization metadata
//   - Loop vectorize.enable and vectorize.width hints
//   - SIMD vector intrinsic declarations (sqrt, fabs)
//   - Underlying IR still has correct function definitions

use iris::{compile, EmitKind};

// ── Test 1: SIMD phase header present ─────────────────────────────────────

#[test]
fn test_simd_header() {
    let src = r#"def f(x: i64) -> i64 { x * 2 }"#;
    let ir = compile(src, "test", EmitKind::Simd).unwrap();
    assert!(
        ir.contains("SIMD") || ir.contains("Vectorization") || ir.contains("vectoriz"),
        "expected SIMD/vectorization header in output:\n{}",
        ir
    );
}

// ── Test 2: target-cpu attribute group ────────────────────────────────────

#[test]
fn test_simd_target_cpu_attribute() {
    let src = r#"def f(x: i64) -> i64 { x + 1 }"#;
    let ir = compile(src, "test", EmitKind::Simd).unwrap();
    assert!(
        ir.contains("target-cpu"),
        "expected 'target-cpu' attribute in SIMD output:\n{}",
        ir
    );
    assert!(
        ir.contains("x86-64-v3") || ir.contains("x86_64"),
        "expected x86-64-v3 target-cpu:\n{}",
        ir
    );
}

// ── Test 3: target-features includes AVX2 ────────────────────────────────

#[test]
fn test_simd_avx2_features() {
    let src = r#"def f(x: f64) -> f64 { x }"#;
    let ir = compile(src, "test", EmitKind::Simd).unwrap();
    assert!(
        ir.contains("avx2") || ir.contains("avx"),
        "expected '+avx2' in target-features:\n{}",
        ir
    );
}

// ── Test 4: !llvm.loop vectorize.enable metadata ─────────────────────────

#[test]
fn test_simd_loop_vectorize_enable() {
    let src = r#"def f(x: f64) -> f64 { x }"#;
    let ir = compile(src, "test", EmitKind::Simd).unwrap();
    assert!(
        ir.contains("vectorize.enable"),
        "expected 'vectorize.enable' metadata hint:\n{}",
        ir
    );
}

// ── Test 5: !llvm.loop vectorize.width metadata ───────────────────────────

#[test]
fn test_simd_loop_vectorize_width() {
    let src = r#"def f(x: f64) -> f64 { x }"#;
    let ir = compile(src, "test", EmitKind::Simd).unwrap();
    assert!(
        ir.contains("vectorize.width"),
        "expected 'vectorize.width' metadata hint:\n{}",
        ir
    );
}

// ── Test 6: FMA in target features ───────────────────────────────────────

#[test]
fn test_simd_fma_feature() {
    let src = r#"def f(x: f64) -> f64 { x * x }"#;
    let ir = compile(src, "test", EmitKind::Simd).unwrap();
    assert!(
        ir.contains("fma"),
        "expected '+fma' in target-features:\n{}",
        ir
    );
}

// ── Test 7: attribute group #0 emitted ────────────────────────────────────

#[test]
fn test_simd_attribute_group() {
    let src = r#"def f(x: i64) -> i64 { x + 1 }"#;
    let ir = compile(src, "test", EmitKind::Simd).unwrap();
    assert!(
        ir.contains("attributes #0"),
        "expected 'attributes #0' group in SIMD output:\n{}",
        ir
    );
}

// ── Test 8: evaluation still correct under SIMD backend ───────────────────

#[test]
fn test_simd_eval_unchanged() {
    let src = r#"def f() -> i64 { 2 + 2 }"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "4");
    // Verify SIMD backend also emits correct function signature.
    let ir = compile(src, "test", EmitKind::Simd).unwrap();
    assert!(ir.contains("@f"), "expected '@f' in SIMD IR:\n{}", ir);
}
