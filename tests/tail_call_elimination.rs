//! Tail-call elimination: self-recursion (loop header) and mutual recursion
//! (trampoline).
//!
//! These live in the Rust suite because nothing globs `tests/*.iris`, so the
//! `.iris` corpus is never executed by `cargo test`. `tests/test_tco.iris` and
//! `tests/test_tco_mutual.iris` cover the same ground for a human running them
//! directly; these are what CI actually enforces.
//!
//! `EmitKind::Eval` builds natively and falls back to the interpreter, so these
//! exercise codegen rather than only the interpreter.

use iris::{compile, EmitKind};

// ── Self recursion ─────────────────────────────────────────────────────────

/// The depth that used to fail. Before `TailCallPass`, this stopped with
/// "call depth exceeded 250" -- it genuinely built 50,000 frames.
#[test]
fn self_tail_call_runs_at_default_depth() {
    let src = r#"
def sum_to(n: i64, acc: i64) -> i64 {
    if n == 0 { acc } else { sum_to(n - 1, acc + n) }
}
def f() -> i64 { sum_to(50000, 0) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1250025000");
}

#[test]
fn self_tail_call_base_cases_still_terminate() {
    let src = r#"
def gcd_tco(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd_tco(b, a % b) }
}
def f() -> i64 { gcd_tco(48, 18) + gcd_tco(17, 5) + gcd_tco(9, 0) }
"#;
    // 6 + 1 + 9
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "16");
}

/// A recursive call whose result is *used* before being returned is not a tail
/// call and must be left alone. If the pass rewrote this into a jump, the
/// multiply would be skipped and the answer would come back as 1.
#[test]
fn non_tail_recursion_is_not_rewritten() {
    let src = r#"
def fact(n: i64) -> i64 {
    if n <= 1 { 1 } else { n * fact(n - 1) }
}
def f() -> i64 { fact(10) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3628800");
}

// ── Mutual recursion ───────────────────────────────────────────────────────

/// Neither call is a self-call, so the loop-header transform does nothing here.
/// This depth previously failed interpreted while passing natively, because
/// clang applied its own TCO to the emitted IR -- the backends disagreed.
#[test]
fn mutual_tail_call_runs_deeper_than_any_stack() {
    let src = r#"
def is_even(n: i64) -> i64 {
    if n == 0 { 1 } else { is_odd(n - 1) }
}
def is_odd(n: i64) -> i64 {
    if n == 0 { 0 } else { is_even(n - 1) }
}
def f() -> i64 { is_even(100000) + is_odd(100000) }
"#;
    // 1 + 0
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

/// Arguments must survive the hand-off, not just control flow. A merge that
/// dropped the accumulator would still terminate -- with 0.
#[test]
fn mutual_tail_call_carries_its_accumulator() {
    let src = r#"
def sum_even(n: i64, acc: i64) -> i64 {
    if n == 0 { acc } else { sum_odd(n - 1, acc + n) }
}
def sum_odd(n: i64, acc: i64) -> i64 {
    if n == 0 { acc } else { sum_even(n - 1, acc + n) }
}
def f() -> i64 { sum_even(100000, 0) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "5000050000");
}

/// Three members need two `sel ==` tests rather than one, so this is where a
/// wrong `else` target in the dispatch chain would surface.
#[test]
fn three_way_mutual_recursion_dispatches_correctly() {
    let src = r#"
def m3_a(n: i64) -> i64 { if n == 0 { 10 } else { m3_b(n - 1) } }
def m3_b(n: i64) -> i64 { if n == 0 { 20 } else { m3_c(n - 1) } }
def m3_c(n: i64) -> i64 { if n == 0 { 30 } else { m3_a(n - 1) } }
def f() -> i64 {
    // Entering at each member, and a depth that rotates all the way round.
    m3_a(0) + m3_b(0) + m3_c(0) + m3_a(1) + m3_a(2) + m3_a(30000)
}
"#;
    // 10 + 20 + 30 + 20 + 30 + 10
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "120");
}

/// The merged signature must not be assumed integral.
#[test]
fn mutual_tail_call_with_f64_accumulator() {
    let src = r#"
def fadd_a(n: i64, acc: f64) -> f64 {
    if n == 0 { acc } else { fadd_b(n - 1, acc + 1.5) }
}
def fadd_b(n: i64, acc: f64) -> f64 {
    if n == 0 { acc } else { fadd_a(n - 1, acc + 1.5) }
}
def f() -> f64 { fadd_a(10000, 0.0) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "15000");
}

/// Members with different signatures cannot share one merged signature, so the
/// SCC is declined. They must still compute the right answer -- the pass has to
/// bow out cleanly rather than transform them wrongly. Kept shallow because
/// these still build real frames.
#[test]
fn mismatched_signatures_are_declined_but_still_correct() {
    let src = r#"
def diff_a(n: i64) -> i64 { if n == 0 { 7 } else { diff_b(n - 1, 1) } }
def diff_b(n: i64, extra: i64) -> i64 { if n == 0 { 8 + extra } else { diff_a(n - 1) } }
def f() -> i64 { diff_a(0) + diff_a(1) + diff_a(2) + diff_b(0, 5) }
"#;
    // 7 + 9 + 7 + 13
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "36");
}

/// The originals survive as forwarders, so a *non-tail* call to a merged member
/// must keep working. Here the call to `is_even` is multiplied, not returned.
#[test]
fn merged_members_remain_callable_from_non_tail_positions() {
    let src = r#"
def is_even(n: i64) -> i64 {
    if n == 0 { 1 } else { is_odd(n - 1) }
}
def is_odd(n: i64) -> i64 {
    if n == 0 { 0 } else { is_even(n - 1) }
}
def f() -> i64 { 100 * is_even(4) + 10 * is_odd(4) + is_even(3) }
"#;
    // 100 * 1 + 10 * 0 + 0
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "100");
}

// ── The renumbering hole the trampoline exposed ────────────────────────────

/// `set_result` handled 100 of 134 `IrInstr` variants behind a `_ => {}`, so a
/// callee containing one of the other five kept the *callee's* ValueId when
/// inlined. `nnz` in a one-line helper was enough to produce
/// "[after inline] main: block0 instr17 operand[0] = %28 not defined".
#[test]
fn inlining_a_callee_that_uses_nnz_renumbers_its_result() {
    let src = r#"
def count_nz(a: [i64; 6]) -> i64 {
    val s = sparsify(a);
    nnz(s)
}
def f() -> i64 {
    val arr = [1, 0, 3, 0, 0, 6];
    count_nz(arr)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}
