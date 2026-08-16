//! Reverse-mode autodiff across the chunked tape.
//!
//! The tape was a fixed thread-local array of 131072 nodes — 12.6 MB reserved
//! per thread whether or not the thread ever called `grad`, measured at 808 MB
//! for 64 sleeping threads (known-issues #47). It is now allocated lazily in
//! 4096-node chunks.
//!
//! Chunks rather than one growable block because nodes reference their parents
//! by **pointer**: reallocating a single block would invalidate every parent
//! pointer already recorded, corrupting the graph silently rather than failing.
//! The test that matters is therefore a chain long enough to span more than one
//! chunk — if growth ever moves a node, the gradient comes back wrong.

use iris::{compile, EmitKind};

/// Builds `v1 = x + x; v2 = v1 + x; ...` so that `v_n = (n + 1) * x` and
/// `d/dx = n + 1`. Straight-line because a tape handle does not survive being
/// passed through a function (#49).
fn chain_program(n: usize) -> String {
    let mut s = String::from("def f() -> f64 {\n    val x = tape(2.0);\n    val v1 = x + x;\n");
    for k in 2..=n {
        s.push_str(&format!("    val v{} = v{} + x;\n", k, k - 1));
    }
    s.push_str(&format!("    val _ = backward(v{n});\n"));
    s.push_str(&format!("    grad(x)\n}}\n"));
    s
}

/// Well inside the first chunk — the case that worked before and must still.
#[test]
fn gradient_is_correct_within_the_first_chunk() {
    let result = compile(&chain_program(100), "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "101");
}

/// 5000 nodes spans two 4096-node chunks. This is the assertion that the old
/// fixed array made impossible to get wrong and the chunked one could: if a
/// grown chunk ever relocated an existing node, the parent pointers recorded
/// before the growth would dangle and the gradient would be wrong.
#[test]
fn gradient_is_correct_across_a_chunk_boundary() {
    let result = compile(&chain_program(5000), "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "5001");
}

/// Straddles the boundary as closely as possible from both sides.
#[test]
fn gradient_is_correct_either_side_of_the_boundary() {
    for n in [4095usize, 4096, 4097] {
        let result = compile(&chain_program(n), "test", EmitKind::Eval).unwrap();
        assert_eq!(
            result.trim(),
            (n + 1).to_string(),
            "wrong gradient for a {n}-node chain"
        );
    }
}

/// The value carried alongside the gradient must also survive growth.
#[test]
fn primal_value_is_correct_across_a_chunk_boundary() {
    let n = 5000;
    let mut s = String::from("def f() -> f64 {\n    val x = tape(2.0);\n    val v1 = x + x;\n");
    for k in 2..=n {
        s.push_str(&format!("    val v{} = v{} + x;\n", k, k - 1));
    }
    s.push_str(&format!("    val _ = backward(v{n});\n    v{n}\n}}\n"));
    let result = compile(&s, "test", EmitKind::Eval).unwrap();
    // v_n = (n + 1) * 2.0
    assert_eq!(result.trim(), ((n + 1) * 2).to_string());
}
