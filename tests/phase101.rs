//! Phase 101 integration tests: cross-platform binary compilation target support.

use iris::codegen::{emit_llvm_ir_with_target, target_data_layout, target_preset_to_triple};
use iris::compile_to_module;

// ── 1. data_layout for x86_64 returns non-empty string ───────────────────────
#[test]
fn test_data_layout_x86_64() {
    let layout = target_data_layout("x86_64-unknown-linux-gnu");
    assert!(!layout.is_empty());
    assert!(layout.contains("i64:64"), "x86_64 layout should specify i64:64, got: {}", layout);
}

// ── 2. data_layout for macos-arm64 is different from x86_64 ─────────────────
#[test]
fn test_data_layout_macos_arm64_differs() {
    let x64 = target_data_layout("x86_64-unknown-linux-gnu");
    let arm = target_data_layout("aarch64-apple-macosx14.0");
    assert_ne!(x64, arm, "aarch64-apple and x86_64 should have different data layouts");
}

// ── 3. LLVM IR for linux-x64 contains x86_64 triple ─────────────────────────
#[test]
fn test_llvm_ir_linux_x64_triple() {
    let module = compile_to_module("def f() -> i64 { 42 }", "m").unwrap();
    let ir = emit_llvm_ir_with_target(&module, Some("linux-x64")).unwrap();
    assert!(
        ir.contains("x86_64-unknown-linux-gnu"),
        "linux-x64 IR should contain x86_64 triple"
    );
}

// ── 4. LLVM IR for macos-arm64 contains aarch64-apple triple ────────────────
#[test]
fn test_llvm_ir_macos_arm64_triple() {
    let module = compile_to_module("def f() -> i64 { 0 }", "m").unwrap();
    let ir = emit_llvm_ir_with_target(&module, Some("macos-arm64")).unwrap();
    assert!(
        ir.contains("aarch64-apple-macosx14.0"),
        "macos-arm64 IR should contain aarch64-apple triple"
    );
}

// ── 5. LLVM IR for riscv64-linux contains riscv64gc triple ───────────────────
#[test]
fn test_llvm_ir_riscv64_triple() {
    let module = compile_to_module("def f() -> i64 { 0 }", "m").unwrap();
    let ir = emit_llvm_ir_with_target(&module, Some("riscv64-linux")).unwrap();
    assert!(
        ir.contains("riscv64gc-unknown-linux-gnu"),
        "riscv64-linux IR should contain riscv64gc triple"
    );
}

// ── 6. emit_llvm_ir_with_target(None) succeeds (native fallback) ─────────────
#[test]
fn test_emit_with_no_target() {
    let module = compile_to_module("def f() -> i64 { 1 }", "m").unwrap();
    let ir = emit_llvm_ir_with_target(&module, None).unwrap();
    assert!(!ir.is_empty());
    assert!(ir.contains("target triple"));
}

// ── 7. Preset "macos-arm64" resolves to the correct LLVM triple ──────────────
#[test]
fn test_preset_macos_arm64_triple() {
    let triple = target_preset_to_triple("macos-arm64");
    assert_eq!(triple, Some("aarch64-apple-macosx14.0"));
}

// ── 8. All 7 presets produce distinct target triples ─────────────────────────
#[test]
fn test_all_presets_distinct() {
    let presets = [
        "linux-x64",
        "linux-arm64",
        "macos-x64",
        "macos-arm64",
        "windows-x64",
        "windows-arm64",
        "riscv64-linux",
    ];
    let triples: Vec<&str> = presets.iter()
        .map(|p| target_preset_to_triple(p).expect(&format!("preset '{}' should resolve", p)))
        .collect();
    // All 7 triples must be distinct.
    let mut seen = std::collections::HashSet::new();
    for t in &triples {
        assert!(seen.insert(*t), "duplicate triple: {}", t);
    }
    assert_eq!(seen.len(), 7);
}
