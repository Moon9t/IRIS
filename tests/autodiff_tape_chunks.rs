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

// ── #48: leaves must not be recycled out from under a live handle ──────────

/// `iris_backward` recycles the arena when it finishes, so the next `tape(...)`
/// used to return an address the previous leaf still held. `grad(x)` then read
/// the *newer* node and reported its gradient as x's:
///
/// ```text
/// dx=6 dy=6        <- dx should not be 6; that is y's gradient
/// ```
///
/// Leaves are now pinned in their own region. x and y are distinct nodes, so x
/// reports 0 for a pass it took no part in rather than borrowing y's answer.
#[test]
fn a_later_tape_does_not_alias_an_earlier_leaf() {
    let src = r#"
def f() -> f64 {
    val x = tape(2.0);
    val fx = x * x;
    val _ = backward(fx);
    val y = tape(3.0);
    val y2 = y * y;
    val _2 = backward(y2);
    // d(y2)/dx is 0 -- x is not in y2's graph. Before the fix this was 6,
    // which is d(y2)/dy: x and y were literally the same node.
    grad(x)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0", "x aliased y's tape node");
}

/// The same program's *other* gradient must still be right, so the fix cannot
/// pass by simply zeroing everything.
#[test]
fn the_second_pass_gradient_is_still_correct() {
    let src = r#"
def f() -> f64 {
    val x = tape(2.0);
    val fx = x * x;
    val _ = backward(fx);
    val y = tape(3.0);
    val y2 = y * y;
    val _2 = backward(y2);
    grad(y)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

/// Reading each gradient immediately after its own pass -- the ordinary usage.
#[test]
fn sequential_passes_each_report_their_own_gradient() {
    let src = r#"
def f() -> f64 {
    val x = tape(2.0);
    val fx = x * x;
    val _ = backward(fx);
    val dx = grad(x);
    val y = tape(3.0);
    val y2 = y * y;
    val _2 = backward(y2);
    val dy = grad(y);
    dx * 100.0 + dy
}
"#;
    // dx = 4, dy = 6
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "406");
}

/// Several leaves live in one graph, each with a distinct partial derivative.
#[test]
fn multiple_live_leaves_get_distinct_gradients() {
    let src = r#"
def f() -> f64 {
    val a = tape(2.0);
    val b = tape(5.0);
    val p = a * b;
    val _ = backward(p);
    // d/da = b = 5, d/db = a = 2
    grad(a) * 10.0 + grad(b)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "52");
}

// ── #49: a tape handle must survive a loop back-edge ───────────────────────

/// A loop-carried taped value used to lose its handle at the back-edge: only
/// the primal was threaded through the block parameters, so on the second
/// iteration `acc` was an ordinary `f64` and `backward` was rejected outright.
/// The handle now rides along as an extra block parameter.
///
/// `acc = sum of 10 * z*z` at z = 1.5, so `d/dz = 20z = 30`.
#[test]
fn a_while_loop_can_accumulate_a_loss() {
    let src = r#"
def f() -> f64 {
    val z = tape(1.5);
    var acc = tape(0.0);
    var i = 0;
    while i < 10 { acc = acc + z * z; i = i + 1; };
    val _ = backward(acc);
    grad(z)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "30");
}

/// The same for `for`, which is the shape a training loop actually takes.
/// `loss = 5 * w*w` at w = 2, so `d/dw = 10w = 20`.
#[test]
fn a_for_loop_can_accumulate_a_loss() {
    let src = r#"
def f() -> f64 {
    val w = tape(2.0);
    var loss = tape(0.0);
    for k in 0..5 { loss = loss + w * w; };
    val _ = backward(loss);
    grad(w)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "20");
}

/// The accumulated *value* must be right too, not only its gradient.
#[test]
fn the_accumulated_loss_value_is_correct() {
    let src = r#"
def f() -> f64 {
    val w = tape(2.0);
    var loss = tape(0.0);
    for k in 0..5 { loss = loss + w * w; };
    val _ = backward(loss);
    loss
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "20");
}

/// A loss that varies per sample, so the gradient depends on the loop index
/// rather than being a constant multiple. sum over k=1..4 of (m*k)^2 with m = 1
/// is 30, and d/dm = sum 2*k*k*m = 60.
#[test]
fn a_per_sample_loss_accumulates_the_right_gradient() {
    let src = r#"
def f() -> f64 {
    val m = tape(1.0);
    var total = tape(0.0);
    for s in 1..5 {
        val x = to_f64(s);
        val pred = m * x;
        total = total + pred * pred;
    };
    val _ = backward(total);
    grad(m)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

/// A loop that records nothing taped must be completely unaffected — this is
/// the overwhelmingly common case, and the threading must not perturb it.
#[test]
fn an_ordinary_loop_is_unchanged() {
    let src = r#"
def f() -> i64 {
    var total = 0;
    for k in 1..101 { total = total + k; };
    total
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "5050");
}
