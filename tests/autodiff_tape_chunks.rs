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

// ── #49: a tape handle must survive a function call ────────────────────────

/// A tape handle cannot be passed in an `f64` parameter, and multi-value
/// returns are unsupported, so it cannot come back out either. A call carrying
/// a taped argument is therefore lowered inline: `tape_nodes` is keyed by
/// `ValueId`, so binding the callee's parameters to the caller's argument
/// values carries the mapping across.
#[test]
fn a_gradient_flows_through_a_function_call() {
    let src = r#"
def sq(v: f64) -> f64 { v * v }
def f() -> f64 {
    val a = tape(3.0);
    val y = sq(a);
    val _ = backward(y);
    grad(a)
}
"#;
    // d/dv v^2 = 2v = 6
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

/// A helper with a local binding, which is the usual shape of a loss function.
#[test]
fn a_gradient_flows_through_a_helper_with_a_local() {
    let src = r#"
def mse(pred: f64, target: f64) -> f64 { val d = pred - target; d * d }
def f() -> f64 {
    val p = tape(5.0);
    val l = mse(p, 3.0);
    val _ = backward(l);
    grad(p)
}
"#;
    // 2(p - t) = 4
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "4");
}

/// Calls nested two deep, each carrying the tape through.
#[test]
fn a_gradient_flows_through_nested_calls() {
    let src = r#"
def sq(v: f64) -> f64 { v * v }
def cube(v: f64) -> f64 { v * v * v }
def outer(v: f64) -> f64 { sq(v) + cube(v) }
def f() -> f64 {
    val n = tape(2.0);
    val y = outer(n);
    val _ = backward(y);
    grad(n)
}
"#;
    // 2v + 3v^2 = 4 + 12
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "16");
}

/// A loop and a helper together -- what a training step actually looks like.
#[test]
fn a_training_step_shape_works_end_to_end() {
    let src = r#"
def mse(pred: f64, target: f64) -> f64 { val d = pred - target; d * d }
def f() -> f64 {
    val w = tape(1.0);
    var total = tape(0.0);
    for s in 1..5 { total = total + mse(w * to_f64(s), 0.0); };
    val _ = backward(total);
    grad(w)
}
"#;
    // sum 2*k*k*w for k=1..4 = 2*30 = 60
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

/// A taped `if` merges two arms through a block parameter, which dropped the
/// handle exactly as a loop back-edge did. This failed with no helper and no
/// inlining involved, so it was its own defect.
#[test]
fn a_gradient_flows_through_an_if_expression() {
    let src = r#"
def f() -> f64 {
    val v = tape(4.0);
    val y = if v > 0.0 { v * 2.0 } else { v * 0.5 };
    val _ = backward(y);
    grad(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

/// The other arm, so the merge is not passing one side's handle for both.
#[test]
fn the_other_if_arm_reports_its_own_gradient() {
    let src = r#"
def f() -> f64 {
    val v = tape(0.0 - 4.0);
    val y = if v > 0.0 { v * 2.0 } else { v * 0.5 };
    val _ = backward(y);
    grad(v)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0.5");
}

// ── The declines, which matter more than the successes ────────────────────

/// A helper containing `return` must NOT be inlined: its `Return` would
/// terminate the *caller*. The whitelist declines it, so the call stays a call.
/// The property under test is that `f` runs to completion and returns 99 --
/// a miscompile would return 6 (the helper's value) instead.
#[test]
fn a_helper_containing_return_is_not_inlined_into_its_caller() {
    let src = r#"
def with_return(v: f64) -> f64 { if v > 0.0 { return v * 3.0; }; v }
def f() -> i64 {
    val a = tape(2.0);
    val y = with_return(a);
    val ignored = y;
    99
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "99", "the helper's return escaped into its caller");
}

/// A recursive taped call must not expand forever at lowering time.
#[test]
fn a_recursive_taped_call_terminates() {
    let src = r#"
def countdown(v: f64, n: i64) -> f64 {
    if n == 0 { v } else { countdown(v * 1.0, n - 1) }
}
def f() -> f64 {
    val a = tape(2.0);
    countdown(a, 3)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

/// An ordinary untaped call must be completely unaffected -- the inline path
/// is only entered when an argument carries a tape handle.
#[test]
fn an_untaped_call_is_unchanged() {
    let src = r#"
def add3(a: i64, b: i64, c: i64) -> i64 { a + b + c }
def f() -> i64 { add3(1, 2, 3) * add3(10, 20, 30) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "360");
}
