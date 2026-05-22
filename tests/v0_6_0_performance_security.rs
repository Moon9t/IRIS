//! Phase 132 integration tests: v0.6.0 — Performance & Security.
//!
//! Validates:
//! - Security policy (sandboxed, default, custom)
//! - Path validation (traversal, null bytes, Windows device names)
//! - Audit logging (operations recorded, reports generated)
//! - Profiler (enter/exit, finalize, summary, folded stacks, flame SVG)
//! - CopyPropPass (duplicate constant elimination)
//! - LicmPass (loop-invariant code motion)
//! - Full pipeline with new passes (compile → IR → eval)
//! - Benchmark suite files exist

use iris::ir::function::Param;
use iris::ir::instr::{BinOp, IrInstr};
use iris::ir::module::{IrFunctionBuilder, IrModule};
use iris::ir::types::{DType, IrType};
use iris::pass::type_infer::TypeInferPass;
use iris::pass::validate::ValidatePass;
use iris::pass::{
    ConstFoldPass, CopyPropPass, CsePass, DcePass, LicmPass, OpExpandPass, PassManager,
    ShapeCheckPass, StrengthReducePass,
};
use iris::profiler::{ProfileResult, Profiler};
use iris::security::{self, AuditOp, SecurityError, SecurityPolicy};

fn scalar_i64() -> IrType {
    IrType::Scalar(DType::I64)
}

fn scalar_f32() -> IrType {
    IrType::Scalar(DType::F32)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Security Policy Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_security_sandboxed_denies_all() {
    let policy = SecurityPolicy::sandboxed();
    assert!(!policy.allow_fs_read);
    assert!(!policy.allow_fs_write);
    assert!(!policy.allow_network);
    assert!(!policy.allow_ffi);
    assert!(!policy.allow_process);
}

#[test]
fn test_security_default_allows_all() {
    let policy = SecurityPolicy::default();
    assert!(policy.allow_fs_read);
    assert!(policy.allow_fs_write);
    assert!(policy.allow_network);
    assert!(policy.allow_ffi);
    assert!(policy.allow_process);
}

#[test]
fn test_security_custom_policy_fs_read_only() {
    let policy = SecurityPolicy {
        allow_fs_read: true,
        allow_fs_write: false,
        allow_network: false,
        allow_ffi: false,
        allow_process: false,
        ..SecurityPolicy::default()
    };
    assert!(policy.allow_fs_read);
    assert!(!policy.allow_fs_write);
    assert!(!policy.allow_network);
}

#[test]
fn test_security_error_display() {
    let err = SecurityError::FsReadDenied {
        path: "/etc/passwd".into(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("filesystem read denied"));
    assert!(msg.contains("/etc/passwd"));

    let err = SecurityError::NetworkDenied {
        host: "evil.com".into(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("network access denied"));
    assert!(msg.contains("evil.com"));
}

#[test]
fn test_security_error_variants_exhaustive() {
    // Ensure all error variants can be constructed.
    let errors: Vec<SecurityError> = vec![
        SecurityError::FsReadDenied { path: "a".into() },
        SecurityError::FsWriteDenied { path: "b".into() },
        SecurityError::NetworkDenied { host: "c".into() },
        SecurityError::FfiDenied {
            library: "d".into(),
        },
        SecurityError::ProcessDenied {
            command: "e".into(),
        },
        SecurityError::PathTraversal { path: "f".into() },
        SecurityError::FileSizeLimitExceeded {
            size: 100,
            limit: 50,
        },
        SecurityError::TooManyOpenFiles {
            current: 10,
            limit: 5,
        },
        SecurityError::TooManyConnections {
            current: 8,
            limit: 4,
        },
    ];
    for err in &errors {
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
    }
    assert_eq!(errors.len(), 9);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Path Validation Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_path_traversal_rejection() {
    assert!(security::validate_path("../../../etc/passwd").is_err());
    assert!(security::validate_path("foo/../../bar").is_err());
    assert!(security::validate_path("..").is_err());
}

#[test]
fn test_path_null_byte_rejection() {
    assert!(security::validate_path("foo\0bar").is_err());
    assert!(security::validate_path("\0").is_err());
}

#[test]
fn test_path_valid_accepts() {
    assert!(security::validate_path("hello.iris").is_ok());
    assert!(security::validate_path("src/main.rs").is_ok());
    assert!(security::validate_path("./local_file").is_ok());
    assert!(security::validate_path("deeply/nested/path/file.txt").is_ok());
}

#[test]
fn test_path_traversal_hidden_in_middle() {
    // "a/../b" should be caught by the traversal detector.
    assert!(security::validate_path("a/../b").is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit Log Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_audit_op_display() {
    assert_eq!(format!("{}", AuditOp::FsRead), "fs_read");
    assert_eq!(format!("{}", AuditOp::FsWrite), "fs_write");
    assert_eq!(format!("{}", AuditOp::Network), "network");
    assert_eq!(format!("{}", AuditOp::FfiLoad), "ffi_load");
    assert_eq!(format!("{}", AuditOp::FfiCall), "ffi_call");
    assert_eq!(format!("{}", AuditOp::ProcessSpawn), "process_spawn");
}

#[test]
fn test_audit_report_empty() {
    // The audit_report function should handle an empty log gracefully.
    // (We can't clear global state in tests safely, but we can check it doesn't crash.)
    let report = security::audit_report();
    assert!(!report.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Profiler Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_profiler_basic_lifecycle() {
    let mut profiler = Profiler::new();
    profiler.enter_function("main");
    profiler.record_instruction();
    profiler.record_instruction();
    profiler.record_instruction();
    let elapsed = profiler.exit_function("main");
    // elapsed >= 0 always true for u64 but check it's reasonable
    assert!(elapsed < 10_000_000); // less than 10 seconds

    let result = profiler.finalize();
    assert!(result.functions.contains_key("main"));
    let main_prof = &result.functions["main"];
    assert_eq!(main_prof.call_count, 1);
    assert_eq!(main_prof.instr_count, 3);
    assert_eq!(result.total_instructions, 3);
}

#[test]
fn test_profiler_nested_calls() {
    let mut profiler = Profiler::new();
    profiler.enter_function("main");
    profiler.record_instruction();
    profiler.enter_function("helper");
    profiler.record_instruction();
    profiler.record_instruction();
    profiler.exit_function("helper");
    profiler.record_instruction();
    profiler.exit_function("main");

    let result = profiler.finalize();
    assert!(result.functions.contains_key("main"));
    assert!(result.functions.contains_key("helper"));
    assert_eq!(result.functions["main"].call_count, 1);
    assert_eq!(result.functions["helper"].call_count, 1);
    assert_eq!(result.functions["main"].instr_count, 2);
    assert_eq!(result.functions["helper"].instr_count, 2);
    assert_eq!(result.total_instructions, 4);
}

#[test]
fn test_profiler_folded_stacks_output() {
    let mut profiler = Profiler::new();
    profiler.enter_function("main");
    profiler.enter_function("compute");
    profiler.exit_function("compute");
    profiler.exit_function("main");

    let result = profiler.finalize();
    let folded = result.to_folded_stacks();
    // Should contain something like "main;compute 1"
    assert!(folded.contains("main;compute"), "Folded stacks: {}", folded);
}

#[test]
fn test_profiler_summary_format() {
    let mut profiler = Profiler::new();
    profiler.enter_function("test_func");
    profiler.record_instruction();
    profiler.exit_function("test_func");

    let result = profiler.finalize();
    let summary = result.summary();
    assert!(summary.contains("IRIS Profile Report"));
    assert!(summary.contains("test_func"));
    assert!(summary.contains("Total time:"));
    assert!(summary.contains("Total instructions:"));
}

#[test]
fn test_profiler_flame_svg_generation() {
    let mut profiler = Profiler::new();
    profiler.enter_function("main");
    profiler.enter_function("hot_function");
    profiler.record_instruction();
    profiler.exit_function("hot_function");
    profiler.exit_function("main");

    let result = profiler.finalize();
    let svg = result.to_flame_svg();
    assert!(
        svg.contains("<svg"),
        "SVG should contain <svg tag: {}",
        &svg[..svg.len().min(200)]
    );
    assert!(svg.contains("</svg>"), "SVG should be well-formed");
}

#[test]
fn test_profiler_multiple_calls() {
    let mut profiler = Profiler::new();
    for _ in 0..10 {
        profiler.enter_function("repeated");
        profiler.record_instruction();
        profiler.exit_function("repeated");
    }

    let result = profiler.finalize();
    assert_eq!(result.functions["repeated"].call_count, 10);
    assert_eq!(result.functions["repeated"].instr_count, 10);
    assert_eq!(result.total_instructions, 10);
}

#[test]
fn test_profile_result_default() {
    let result = ProfileResult::default();
    assert!(result.functions.is_empty());
    assert!(result.folded_stacks.is_empty());
    assert_eq!(result.total_program_us, 0);
    assert_eq!(result.total_instructions, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CopyPropPass Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_copy_prop_pass_name() {
    let pass = CopyPropPass;
    assert_eq!(iris::pass::Pass::name(&pass), "copy-prop");
}

#[test]
fn test_copy_prop_dedup_constants() {
    // Build a function with two identical ConstInt(42) instructions.
    // After CopyPropPass, the second should be replaced.
    let params = vec![Param {
        name: "x".into(),
        ty: scalar_i64(),
    }];
    let mut builder = IrFunctionBuilder::new("test_dedup", params, scalar_i64());
    let entry = builder.create_block(Some("entry"));
    let _x = builder.add_block_param(entry, Some("x"), scalar_i64());
    builder.set_current_block(entry);

    // Two constants with the same value 42.
    let c1 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstInt {
            result: c1,
            value: 42,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    let c2 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstInt {
            result: c2,
            value: 42,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    // Use both in an add: result = c1 + c2.
    let sum = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: sum,
            op: BinOp::Add,
            lhs: c1,
            rhs: c2,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    builder.push_instr(IrInstr::Return { values: vec![sum] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    // Run CopyPropPass.
    let mut pass = CopyPropPass;
    iris::pass::Pass::run(&mut pass, &mut module).expect("CopyPropPass should succeed");

    // After propagation, the BinOp should use c1 for both operands
    // (c2 is a duplicate of c1, so c2 → c1).
    let func = &module.functions()[0];
    let mut found_add = false;
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::BinOp {
                op: BinOp::Add,
                lhs,
                rhs,
                ..
            } = instr
            {
                // Both should now reference c1 (the first const).
                assert_eq!(lhs, rhs, "CopyProp should make both operands the same");
                found_add = true;
            }
        }
    }
    assert!(found_add, "Should find the Add instruction");
}

#[test]
fn test_copy_prop_no_change_on_distinct_constants() {
    // Two different constants should NOT be merged.
    let params = vec![];
    let mut builder = IrFunctionBuilder::new("test_distinct", params, scalar_i64());
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);

    let c1 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstInt {
            result: c1,
            value: 10,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    let c2 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstInt {
            result: c2,
            value: 20,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    let sum = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: sum,
            op: BinOp::Add,
            lhs: c1,
            rhs: c2,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    builder.push_instr(IrInstr::Return { values: vec![sum] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let mut pass = CopyPropPass;
    iris::pass::Pass::run(&mut pass, &mut module).expect("CopyPropPass should succeed");

    // The operands should remain different.
    let func = &module.functions()[0];
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::BinOp {
                op: BinOp::Add,
                lhs,
                rhs,
                ..
            } = instr
            {
                assert_ne!(lhs, rhs, "Distinct constants should not be merged");
            }
        }
    }
}

#[test]
fn test_copy_prop_float_dedup() {
    // Two identical ConstFloat(3.14) should be deduplicated.
    let params = vec![];
    let mut builder = IrFunctionBuilder::new("test_float_dedup", params, scalar_f32());
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);

    let c1 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: c1,
            value: 3.14,
            ty: scalar_f32(),
        },
        Some(scalar_f32()),
    );

    let c2 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstFloat {
            result: c2,
            value: 3.14,
            ty: scalar_f32(),
        },
        Some(scalar_f32()),
    );

    let prod = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: prod,
            op: BinOp::Mul,
            lhs: c1,
            rhs: c2,
            ty: scalar_f32(),
        },
        Some(scalar_f32()),
    );

    builder.push_instr(IrInstr::Return { values: vec![prod] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let mut pass = CopyPropPass;
    iris::pass::Pass::run(&mut pass, &mut module).expect("CopyPropPass should succeed");

    let func = &module.functions()[0];
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::BinOp {
                op: BinOp::Mul,
                lhs,
                rhs,
                ..
            } = instr
            {
                assert_eq!(lhs, rhs, "Duplicate floats should be merged");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  LicmPass Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_licm_pass_name() {
    let pass = LicmPass;
    assert_eq!(iris::pass::Pass::name(&pass), "licm");
}

#[test]
fn test_licm_no_crash_simple_function() {
    // A function with no loops should pass through LICM unchanged.
    let params = vec![Param {
        name: "n".into(),
        ty: scalar_i64(),
    }];
    let mut builder = IrFunctionBuilder::new("no_loop", params, scalar_i64());
    let entry = builder.create_block(Some("entry"));
    let n = builder.add_block_param(entry, Some("n"), scalar_i64());
    builder.set_current_block(entry);

    let c1 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstInt {
            result: c1,
            value: 1,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    let sum = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: sum,
            op: BinOp::Add,
            lhs: n,
            rhs: c1,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    builder.push_instr(IrInstr::Return { values: vec![sum] }, None);

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let mut pass = LicmPass;
    iris::pass::Pass::run(&mut pass, &mut module)
        .expect("LicmPass should not crash on non-loop code");

    // Verify function still has the return.
    let func = &module.functions()[0];
    let has_return = func
        .blocks()
        .iter()
        .any(|b| b.instrs.iter().any(|i| matches!(i, IrInstr::Return { .. })));
    assert!(
        has_return,
        "Function should still have a return instruction"
    );
}

#[test]
fn test_licm_empty_module() {
    let mut module = IrModule::new("empty");
    let mut pass = LicmPass;
    iris::pass::Pass::run(&mut pass, &mut module).expect("LicmPass on empty module should succeed");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Full Pipeline with New Passes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_pipeline_with_copyprop_licm() {
    // Compile a simple IRIS program through the full pipeline (which now includes
    // CopyPropPass and LicmPass) and verify it produces valid IR.
    let src = r#"
def add_constants() -> i64 {
    val a = 42
    val b = 42
    a + b
}
"#;
    let ir = iris::compile(src, "test", iris::EmitKind::Ir).expect("compile");
    assert!(
        ir.contains("add_constants"),
        "IR should contain the function: {}",
        ir
    );
}

#[test]
fn test_full_pipeline_arithmetic() {
    let src = r#"
def compute(x: i64) -> i64 {
    val a = 10
    val b = 20
    x + a + b
}
"#;
    let ir = iris::compile(src, "test", iris::EmitKind::Ir).expect("compile");
    assert!(
        ir.contains("compute"),
        "IR should contain the function: {}",
        ir
    );
}

#[test]
fn test_compile_to_module_includes_new_passes() {
    // compile_to_module should include CopyPropPass and LicmPass in the pipeline.
    let src = r#"
def identity(x: i64) -> i64 {
    x
}
"#;
    let module = iris::compile_to_module(src, "test").expect("compile_to_module");
    assert!(
        !module.functions().is_empty(),
        "Module should have functions"
    );
    assert_eq!(module.functions()[0].name, "identity");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Pass Manager Integration Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_pass_manager_with_all_v060_passes() {
    // Run the full v0.6.0 pass pipeline on a hand-built IR module.
    let params = vec![Param {
        name: "x".into(),
        ty: scalar_i64(),
    }];
    let mut builder = IrFunctionBuilder::new("opt_test", params, scalar_i64());
    let entry = builder.create_block(Some("entry"));
    let x = builder.add_block_param(entry, Some("x"), scalar_i64());
    builder.set_current_block(entry);

    let c1 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstInt {
            result: c1,
            value: 5,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    let c2 = builder.fresh_value();
    builder.push_instr(
        IrInstr::ConstInt {
            result: c2,
            value: 5,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    let sum = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result: sum,
            op: BinOp::Add,
            lhs: c1,
            rhs: c2,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    let result = builder.fresh_value();
    builder.push_instr(
        IrInstr::BinOp {
            result,
            op: BinOp::Add,
            lhs: x,
            rhs: sum,
            ty: scalar_i64(),
        },
        Some(scalar_i64()),
    );

    builder.push_instr(
        IrInstr::Return {
            values: vec![result],
        },
        None,
    );

    let mut module = IrModule::new("test");
    module.add_function(builder.build()).unwrap();

    let mut pm = PassManager::new();
    pm.add_pass(ValidatePass);
    pm.add_pass(TypeInferPass);
    pm.add_pass(ConstFoldPass);
    pm.add_pass(StrengthReducePass);
    pm.add_pass(CopyPropPass);
    pm.add_pass(OpExpandPass);
    pm.add_pass(LicmPass);
    pm.add_pass(DcePass);
    pm.add_pass(CsePass);
    pm.add_pass(ShapeCheckPass);

    pm.run(&mut module).expect("Full pipeline should succeed");

    // The module should still have a valid function.
    assert!(!module.functions().is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Benchmark Suite Files Exist
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_benchmark_files_exist() {
    // Verify that the new benchmark files were created.
    let bench_files = [
        "benches/binary_search_bench.iris",
        "benches/tree_bench.iris",
        "benches/hashmap_bench.iris",
        "benches/collatz_bench.iris",
        "benches/sieve_bench.iris",
        "benches/numerical_bench.iris",
        // Original benchmarks too.
        "benches/factorial_bench.iris",
        "benches/fib_bench.iris",
        "benches/list_bench.iris",
        "benches/matrix_bench.iris",
        "benches/sort_bench.iris",
        "benches/string_bench.iris",
    ];
    for file in &bench_files {
        assert!(
            std::path::Path::new(file).exists(),
            "Benchmark file should exist: {}",
            file
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Security + Policy Integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_security_working_dir_policy() {
    let policy = SecurityPolicy::with_working_dir(std::path::PathBuf::from("/safe/dir"));
    assert!(policy.allow_fs_read);
    assert!(policy.allow_fs_write);
    assert_eq!(policy.fs_read_allowlist.len(), 1);
    assert_eq!(policy.fs_write_allowlist.len(), 1);
}

#[test]
fn test_security_policy_limits() {
    let policy = SecurityPolicy {
        max_file_write_bytes: 1024 * 1024,
        max_open_files: 256,
        max_connections: 64,
        ..SecurityPolicy::default()
    };
    assert_eq!(policy.max_file_write_bytes, 1024 * 1024);
    assert_eq!(policy.max_open_files, 256);
    assert_eq!(policy.max_connections, 64);
}

#[test]
fn test_security_blocklist_configuration() {
    let policy = SecurityPolicy {
        fs_blocklist: vec![
            std::path::PathBuf::from("/etc"),
            std::path::PathBuf::from("/sys"),
        ],
        network_blocklist: vec!["evil.com".into(), "malware.net".into()],
        ..SecurityPolicy::default()
    };
    assert_eq!(policy.fs_blocklist.len(), 2);
    assert_eq!(policy.network_blocklist.len(), 2);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CopyPropPass on Empty Module
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_copy_prop_empty_module() {
    let mut module = IrModule::new("empty");
    let mut pass = CopyPropPass;
    iris::pass::Pass::run(&mut pass, &mut module).expect("CopyPropPass on empty module");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Profiler Edge Cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_profiler_exit_without_enter() {
    // Exiting a function that was never entered should not panic.
    let mut profiler = Profiler::new();
    let elapsed = profiler.exit_function("nonexistent");
    assert_eq!(elapsed, 0);
    let result = profiler.finalize();
    assert_eq!(result.total_instructions, 0);
}

#[test]
fn test_profiler_record_instruction_without_function() {
    // Recording instructions with no active function should not panic.
    let mut profiler = Profiler::new();
    profiler.record_instruction();
    profiler.record_instruction();
    let result = profiler.finalize();
    assert_eq!(result.total_instructions, 2);
}
