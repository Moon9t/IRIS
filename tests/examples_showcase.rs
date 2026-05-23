//! End-to-end example coverage for the IRIS examples folder.

use iris::{compile_file, EmitKind};
use std::path::{Path, PathBuf};

fn example_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

#[test]
fn arrays_example_runs() {
    let out = compile_file(&example_path("01_basics/arrays.iris"), EmitKind::Eval).unwrap();
    assert!(out.contains("sum = 150"), "output was:\n{}", out);
    assert!(out.contains("last = 9"), "output was:\n{}", out);
}

#[test]
fn concurrency_example_runs() {
    let out = compile_file(&example_path("05_systems/concurrency.iris"), EmitKind::Eval).unwrap();
    assert!(
        out.contains("messages received = 5"),
        "output was:\n{}",
        out
    );
    assert!(out.contains("message total = 15"), "output was:\n{}", out);
    assert!(out.contains("atomic counter = 5"), "output was:\n{}", out);
}

#[test]
fn ffi_example_runs() {
    let out = compile_file(&example_path("05_systems/ffi.iris"), EmitKind::Eval).unwrap();
    assert!(out.contains("ffi = ffi"), "output was:\n{}", out);
}

#[test]
fn loops_example_runs() {
    let out = compile_file(&example_path("01_basics/loops.iris"), EmitKind::Eval).unwrap();
    assert!(out.contains("sum_to(10) = 55"), "output was:\n{}", out);
    assert!(out.contains("sum_odds(10) = 25"), "output was:\n{}", out);
    assert!(
        out.contains("first_even_after(7) = 8"),
        "output was:\n{}",
        out
    );
    assert!(
        out.contains("count_pairs(3, 2) = 6"),
        "output was:\n{}",
        out
    );
}

#[test]
fn neural_net_example_runs() {
    let out = compile_file(
        &example_path("06_machine_learning/neural_net.iris"),
        EmitKind::Eval,
    )
    .unwrap();
    assert!(out.contains("pred(0,0) = "), "output was:\n{}", out);
    assert!(out.contains("pred(1,1) = "), "output was:\n{}", out);
    assert!(out.contains("xor loss = "), "output was:\n{}", out);
}

#[test]
fn ml_full_pipeline_example_runs() {
    let out = compile_file(
        &example_path("06_machine_learning/ml_full_pipeline.iris"),
        EmitKind::Eval,
    )
    .unwrap();
    assert!(out.contains("ingested rows = 8"), "output was:\n{}", out);
    assert!(out.contains("updated accuracy = "), "output was:\n{}", out);
    assert!(
        out.contains("expected output = approve"),
        "output was:\n{}",
        out
    );
    assert!(out.contains("torch hook = ready"), "output was:\n{}", out);
    assert!(
        out.contains("tensorflow hook = ready"),
        "output was:\n{}",
        out
    );
}

#[test]
fn networking_example_runs() {
    let out = compile_file(&example_path("05_systems/networking.iris"), EmitKind::Eval).unwrap();
    assert!(out.contains("preview status = 200"), "output was:\n{}", out);
    assert!(
        out.contains("preview body = echo: hello"),
        "output was:\n{}",
        out
    );
    assert!(out.contains("status = 200"), "output was:\n{}", out);
    assert!(out.contains("body = ready"), "output was:\n{}", out);
}

#[test]
fn backend_example_runs() {
    let out = compile_file(&example_path("05_systems/backend.iris"), EmitKind::Eval).unwrap();
    assert!(out.contains("routes = 6"), "output was:\n{}", out);
    assert!(
        out.contains("health = {\"status\":\"ok\"}"),
        "output was:\n{}",
        out
    );
    assert!(
        out.contains("echo = {\"echo\":\"hello\"}"),
        "output was:\n{}",
        out
    );
}

#[test]
fn database_example_runs() {
    let out = compile_file(&example_path("05_systems/database.iris"), EmitKind::Eval).unwrap();
    assert!(out.contains("rows = 2"), "output was:\n{}", out);
    assert!(out.contains("first = Write docs"), "output was:\n{}", out);
    assert!(
        out.contains("second = Ship release"),
        "output was:\n{}",
        out
    );
}

#[test]
fn sql_params_example_runs() {
    let out = compile_file(&example_path("05_systems/sql_params.iris"), EmitKind::Eval).unwrap();
    assert!(out.contains("Query result:"), "output was:\n{}", out);
    assert!(out.contains("Bob, 25"), "output was:\n{}", out);
    assert!(out.contains("matched rows = 1"), "output was:\n{}", out);
}
