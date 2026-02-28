//! Phase 91: REPL — interactive evaluation of IRIS source lines.

use iris::ReplState;

// ── 1. Basic expression ──────────────────────────────────────────────────────

#[test]
fn test_repl_basic_expr() {
    let mut repl = ReplState::new();
    let result = repl.eval("1 + 2").unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 2. Val binding then use ──────────────────────────────────────────────────

#[test]
fn test_repl_val_binding_and_use() {
    let mut repl = ReplState::new();
    repl.eval("val x = 10").unwrap();
    let result = repl.eval("x + 5").unwrap();
    assert_eq!(result.trim(), "15");
}

// ── 3. Function definition then call ────────────────────────────────────────

#[test]
fn test_repl_function_def_and_call() {
    let mut repl = ReplState::new();
    repl.eval("def sq(n: i64) -> i64 { n * n }").unwrap();
    let result = repl.eval("sq(5)").unwrap();
    assert_eq!(result.trim(), "25");
}

// ── 4. Record definition and field access ────────────────────────────────────

#[test]
fn test_repl_record_and_field_access() {
    let mut repl = ReplState::new();
    repl.eval("record Point { x: i64, y: i64 }").unwrap();
    repl.eval("val p = Point { x: 10, y: 20 }").unwrap();
    let result = repl.eval("p.x").unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 5. Reset clears all state ────────────────────────────────────────────────

#[test]
fn test_repl_reset_clears_state() {
    let mut repl = ReplState::new();
    repl.eval("def sq(n: i64) -> i64 { n * n }").unwrap();
    repl.reset();
    // After reset, sq is no longer defined — should error.
    let result = repl.eval("sq(5)");
    assert!(result.is_err(), "expected error after reset but got Ok");
}

// ── 6. Error recovery: bad input leaves state intact ────────────────────────

#[test]
fn test_repl_error_recovery() {
    let mut repl = ReplState::new();
    // Malformed input should return an error.
    let bad = repl.eval("@@@not iris@@@");
    assert!(bad.is_err());
    // Valid input still works afterwards.
    let good = repl.eval("1 + 1").unwrap();
    assert_eq!(good.trim(), "2");
}

// ── 7. Multiple val bindings accumulate ─────────────────────────────────────

#[test]
fn test_repl_multiple_val_bindings() {
    let mut repl = ReplState::new();
    repl.eval("val a = 3").unwrap();
    repl.eval("val b = 7").unwrap();
    let result = repl.eval("a + b").unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 8. choice / enum type usable in REPL session ────────────────────────────

#[test]
fn test_repl_choice_enum() {
    let mut repl = ReplState::new();
    // Defining a choice type returns a "defined: ..." acknowledgement.
    let ack = repl.eval("choice Color { Red, Green, Blue }").unwrap();
    assert!(ack.contains("Color"), "expected 'Color' in ack, got: {}", ack);
    // Subsequent arithmetic still works.
    let result = repl.eval("2 * 3").unwrap();
    assert_eq!(result.trim(), "6");
}
