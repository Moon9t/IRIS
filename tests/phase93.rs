//! Phase 93: Debugger — span table, trace-based step/breakpoint debugging.

use iris::{DebugSession, compile_to_module};

// Source with two let statements and a tail expression.
// Line 1: def main() -> i64 {
// Line 2:     val x = 10
// Line 3:     val y = 20
// Line 4:     x + y
// Line 5: }
const SRC_TWO_STMTS: &str = "def main() -> i64 {\n    val x = 10\n    val y = 20\n    x + y\n}";

// Source with three statements for two-breakpoint test.
// Line 1: def main() -> i64 {
// Line 2:     val a = 1
// Line 3:     val b = 2
// Line 4:     val c = 3
// Line 5:     a + b + c
// Line 6: }
const SRC_THREE_STMTS: &str = "def main() -> i64 {\n    val a = 1\n    val b = 2\n    val c = 3\n    a + b + c\n}";

// ── 1. Span table populated after compile_to_module ─────────────────────────

#[test]
fn test_span_table_populated() {
    let module = compile_to_module(SRC_TWO_STMTS, "debug").unwrap();
    let main_fn = module.functions().iter().find(|f| f.name == "main").unwrap();
    assert!(!main_fn.span_table.is_empty(), "span_table should be non-empty after compile_to_module");
}

// ── 2. DebugSession compiles and starts without error ───────────────────────

#[test]
fn test_debug_session_start() {
    let mut session = DebugSession::new();
    session.set_source(SRC_TWO_STMTS);
    session.start().expect("DebugSession::start should not error on valid source");
    assert!(!session.is_finished(), "session should have trace entries");
}

// ── 3. Breakpoint at line 3 returns frame with line == 3 ───────────────────

#[test]
fn test_breakpoint_at_line_3() {
    let mut session = DebugSession::new();
    session.set_source(SRC_TWO_STMTS);
    session.set_breakpoint(3);
    session.start().unwrap();
    let frame = session.continue_to_breakpoint()
        .expect("should hit breakpoint at line 3");
    assert_eq!(frame.line, 3, "expected frame at line 3, got line {}", frame.line);
}

// ── 4. Frame has correct function name ───────────────────────────────────────

#[test]
fn test_frame_func_name() {
    let mut session = DebugSession::new();
    session.set_source(SRC_TWO_STMTS);
    session.start().unwrap();
    let frame = session.current_frame().expect("should have at least one frame");
    assert_eq!(frame.func_name, "main", "frame func_name should be 'main'");
}

// ── 5. step() advances cursor; all_frames().len() matches trace count ────────

#[test]
fn test_step_advances_cursor() {
    let mut session = DebugSession::new();
    session.set_source(SRC_TWO_STMTS);
    session.start().unwrap();
    let total = session.all_frames().len();
    assert!(total >= 2, "should have at least 2 trace entries (one per val stmt)");
    // Step once and confirm we advance.
    let advanced = session.step();
    assert!(advanced, "step() should return true when not at end");
}

// ── 6. Two breakpoints hit in order ─────────────────────────────────────────

#[test]
fn test_two_breakpoints_in_order() {
    let mut session = DebugSession::new();
    session.set_source(SRC_THREE_STMTS);
    session.set_breakpoint(2);
    session.set_breakpoint(3);
    session.start().unwrap();

    // Initial frame should be at line 2 (first trace entry = first let stmt).
    let first = session.current_frame().expect("should have initial frame");
    assert_eq!(first.line, 2, "initial frame should be at line 2");

    // Advance to next breakpoint: should hit line 3.
    let second = session.continue_to_breakpoint()
        .expect("should hit second breakpoint at line 3");
    assert_eq!(second.line, 3, "second breakpoint frame should be at line 3");
}

// ── 7. is_finished() is true after continue_to_breakpoint() runs off the end ─

#[test]
fn test_is_finished_after_all_frames() {
    let mut session = DebugSession::new();
    session.set_source(SRC_TWO_STMTS);
    // No breakpoints set — continue_to_breakpoint() will scan past all frames.
    session.start().unwrap();
    let result = session.continue_to_breakpoint();
    assert!(result.is_none(), "expected None when no breakpoints are set");
    assert!(session.is_finished(), "is_finished() should be true after running off end");
}

// ── 8. Trace has ≥ 2 entries for source with 2 let statements ───────────────

#[test]
fn test_trace_entry_count() {
    let mut session = DebugSession::new();
    session.set_source(SRC_TWO_STMTS);
    session.start().unwrap();
    let frames = session.all_frames();
    assert!(
        frames.len() >= 2,
        "expected at least 2 trace entries (one per val stmt), got {}",
        frames.len()
    );
    // Both frames should reference "main".
    for frame in frames {
        assert_eq!(frame.func_name, "main");
    }
}
