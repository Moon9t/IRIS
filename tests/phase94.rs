//! Phase 94: Module System — file brings, stdlib brings, visibility, transitive, circular.

use iris::{compile_multi, EmitKind};

// ── 1. bring "file.iris" — public function callable ─────────────────────────

#[test]
fn test_bring_file_pub_function() {
    let utils_src = r#"pub def add_one(x: i64) -> i64 { x + 1 }"#;
    let main_src = r#"
bring "utils.iris"
def f() -> i64 { add_one(41) }
"#;
    let result = compile_multi(
        &[("utils", utils_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 2. Private function from dependency NOT callable ─────────────────────────

#[test]
fn test_bring_file_private_not_visible() {
    let utils_src = r#"def secret(x: i64) -> i64 { x + 100 }"#;
    let main_src = r#"
bring "utils.iris"
def f() -> i64 { secret(0) }
"#;
    let result = compile_multi(
        &[("utils", utils_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    );
    assert!(result.is_err(), "expected error: private fn should not be visible");
}

// ── 3. pub record from dependency usable as a type ──────────────────────────

#[test]
fn test_bring_file_pub_record() {
    let types_src = r#"pub record Point { x: i64, y: i64 }"#;
    let main_src = r#"
bring "types.iris"
def f() -> i64 {
    val p = Point { x: 3, y: 4 }
    p.x + p.y
}
"#;
    let result = compile_multi(
        &[("types", types_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "7");
}

// ── 4. pub const accessible ─────────────────────────────────────────────────

#[test]
fn test_bring_file_pub_const() {
    let config_src = r#"pub const ANSWER: i64 = 42"#;
    let main_src = r#"
bring "config.iris"
def f() -> i64 { ANSWER }
"#;
    let result = compile_multi(
        &[("config", config_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 5. Transitive bring: A brings B, B brings C — C's pub items in A ────────

#[test]
fn test_bring_transitive() {
    // C defines a helper.
    let c_src = r#"pub def triple(x: i64) -> i64 { x * 3 }"#;
    // B brings C, re-exports via its own pub fn (note: B's bring of C resolves in BFS).
    let b_src = r#"
bring "c.iris"
pub def six_times(x: i64) -> i64 { triple(x) * 2 }
"#;
    // A brings B; should be able to call six_times (and triple if it's pub in C).
    let main_src = r#"
bring "b.iris"
def f() -> i64 { six_times(7) }
"#;
    let result = compile_multi(
        &[("c", c_src), ("b", b_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "42");
}

// ── 6. Circular dependency: file A brings B, B brings A — error or safe stop ─

#[test]
fn test_bring_circular_no_infinite_loop() {
    // compile_multi uses a visited set so circular deps don't loop forever.
    // With circular deps, at least one module just won't see the other's items,
    // but the compilation should either succeed or fail — never loop.
    let a_src = r#"
bring "b.iris"
pub def a_fn() -> i64 { 1 }
"#;
    let b_src = r#"
bring "a.iris"
pub def b_fn() -> i64 { 2 }
"#;
    let main_src = r#"
bring "a.iris"
def f() -> i64 { a_fn() }
"#;
    // Should not hang. May succeed or fail, but must return.
    let _result = compile_multi(
        &[("a", a_src), ("b", b_src), ("main", main_src)],
        "main",
        EmitKind::Eval,
    );
    // No assertion on result — just that it terminates.
}

// ── 7. bring std.math — stdlib math module loads and a function is callable ──

#[test]
fn test_bring_stdlib_math() {
    let main_src = r#"
bring std.math
def f() -> i64 { gcd(12, 8) }
"#;
    let result = compile_multi(
        &[("main", main_src)],
        "main",
        EmitKind::Eval,
    ).unwrap();
    assert_eq!(result.trim(), "4");
}

// ── 8. bring via actual disk files using FileCompiler ───────────────────────

#[test]
fn test_bring_file_from_disk() {
    use std::io::Write;

    let dir = std::env::temp_dir().join("iris_phase94_test");
    std::fs::create_dir_all(&dir).unwrap();

    // Write a utility file.
    let util_path = dir.join("myutil.iris");
    {
        let mut f = std::fs::File::create(&util_path).unwrap();
        writeln!(f, "pub def mul_ten(x: i64) -> i64 {{ x * 10 }}").unwrap();
    }

    // Write the main file that brings the utility.
    let main_path = dir.join("main94.iris");
    {
        let mut f = std::fs::File::create(&main_path).unwrap();
        writeln!(f, "bring \"myutil.iris\"").unwrap();
        writeln!(f, "def f() -> i64 {{ mul_ten(7) }}").unwrap();
    }

    let result = iris::compile_file(&main_path, EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "70");
}
