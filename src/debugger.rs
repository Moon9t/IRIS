//! Trace-based debugger for IRIS programs.
//!
//! [`DebugSession`] compiles and executes a program, recording a trace of
//! executed instructions with their source positions and in-scope variable
//! snapshots. The trace can then be replayed step-by-step or advanced to the
//! next breakpoint, providing offline (post-mortem) debugging without requiring
//! coroutines or unsafe threading.
//!
//! ## Features
//!
//! - Source breakpoints with optional **hit-count** and **condition** expressions
//! - **Log-point** messages (evaluated and printed without stopping)
//! - Step forward / backward (time-travel)
//! - Step-over, step-into, step-out with function-level granularity
//! - Watch-expression evaluation against the current frame
//! - Variable mutation via `set_variable`
//! - Debug-console completions drawn from in-scope variables

use std::collections::HashMap;

use crate::error::Error;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single recorded execution step with source position and variable state.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Name of the function being executed.
    pub func_name: String,
    /// 1-based source line number.
    pub line: u32,
    /// 1-based source column number.
    pub column: u32,
    /// Snapshot of named variables at this point: (display_name, display_value).
    pub variables: Vec<(String, String)>,
    /// Call depth of this frame (0 = top-level / main).
    pub depth: u32,
}

/// Metadata attached to a breakpoint.
#[derive(Debug, Clone, Default)]
pub struct BreakpointInfo {
    /// Optional condition expression — breakpoint only fires when this evaluates to `true`.
    pub condition: Option<String>,
    /// Optional hit-count threshold (e.g. `">5"`). The breakpoint fires when
    /// the hit count satisfies the expression.
    pub hit_condition: Option<String>,
    /// Optional log message — if set, the breakpoint becomes a log-point:
    /// the message is emitted to the debug console and execution continues.
    pub log_message: Option<String>,
    /// Number of times execution has reached this breakpoint.
    pub hit_count: u64,
}

/// A debug session for a single IRIS source file.
///
/// # Usage
/// ```text
/// let mut session = DebugSession::new();
/// session.set_source(src);
/// session.set_breakpoint(3, None);
/// session.start().unwrap();
/// if let Some(frame) = session.continue_to_breakpoint() {
///     println!("stopped at line {}", frame.line);
/// }
/// ```
pub struct DebugSession {
    source: String,
    breakpoints: HashMap<u32, BreakpointInfo>, // line -> info
    trace: Vec<TraceEntry>,
    cursor: usize,
    /// Log messages emitted by log-points during the last `continue` / `step`.
    pub pending_logs: Vec<String>,
    /// Whether to break on panics / runtime errors.
    pub break_on_exception: bool,
    /// Cached exception message from the last `start()` if the program panicked.
    pub exception_message: Option<String>,
}

impl Default for DebugSession {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugSession {
    /// Creates an empty debug session.
    pub fn new() -> Self {
        Self {
            source: String::new(),
            breakpoints: HashMap::new(),
            trace: Vec::new(),
            cursor: 0,
            pending_logs: Vec::new(),
            break_on_exception: true,
            exception_message: None,
        }
    }

    /// Sets the IRIS source code to debug.
    pub fn set_source(&mut self, src: &str) {
        self.source = src.to_owned();
        self.trace.clear();
        self.cursor = 0;
        self.exception_message = None;
    }

    /// Registers a breakpoint at `line` (1-based) with optional metadata.
    pub fn set_breakpoint(&mut self, line: u32, info: Option<BreakpointInfo>) {
        self.breakpoints.insert(line, info.unwrap_or_default());
    }

    /// Removes a breakpoint.
    pub fn remove_breakpoint(&mut self, line: u32) {
        self.breakpoints.remove(&line);
    }

    /// Removes all breakpoints.
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Compiles the source and runs it, collecting a full execution trace.
    ///
    /// After this call, use `step()` / `continue_to_breakpoint()` to walk the trace.
    pub fn start(&mut self) -> Result<(), Error> {
        self.trace.clear();
        self.cursor = 0;
        self.exception_message = None;

        // Compile to module (runs all passes).
        let module = crate::compile_to_module(&self.source, "debug")?;

        // Collect the trace via the interpreter.
        let trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        {
            let t = std::rc::Rc::clone(&trace);
            crate::interp::collect_trace(&module, &self.source, t)?;
        }
        self.trace = std::rc::Rc::try_unwrap(trace)
            .map_err(|_| ())
            .unwrap_or_default()
            .into_inner();

        Ok(())
    }

    /// Returns the current trace frame (at `cursor`).
    pub fn current_frame(&self) -> Option<&TraceEntry> {
        self.trace.get(self.cursor)
    }

    /// Returns the total number of trace entries.
    pub fn trace_len(&self) -> usize {
        self.trace.len()
    }

    /// Returns current cursor position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    // ── Stepping ──────────────────────────────────────────────────────────

    /// Advances the cursor by one step. Returns `false` if already at the end.
    pub fn step(&mut self) -> bool {
        if self.cursor + 1 < self.trace.len() {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    /// Steps backwards by one step. Returns `false` if already at the beginning.
    pub fn step_back(&mut self) -> bool {
        if self.cursor > 0 {
            self.cursor -= 1;
            true
        } else {
            false
        }
    }

    /// Step over: advance to the next trace entry at the same or lower call depth,
    /// skipping entries that are deeper (i.e. inside called functions).
    pub fn step_over(&mut self) -> bool {
        if self.cursor + 1 >= self.trace.len() {
            return false;
        }
        let current_depth = self.trace[self.cursor].depth;
        self.cursor += 1;
        while self.cursor < self.trace.len() {
            if self.trace[self.cursor].depth <= current_depth {
                return true;
            }
            self.cursor += 1;
        }
        // Reached end
        self.cursor = self.trace.len().saturating_sub(1);
        true
    }

    /// Step into: simply advance one step (enters function calls).
    pub fn step_into(&mut self) -> bool {
        self.step()
    }

    /// Step out: advance until the call depth is strictly lower than the current depth
    /// (i.e. we've returned from the current function).
    pub fn step_out(&mut self) -> bool {
        if self.cursor + 1 >= self.trace.len() {
            return false;
        }
        let current_depth = self.trace[self.cursor].depth;
        self.cursor += 1;
        while self.cursor < self.trace.len() {
            if self.trace[self.cursor].depth < current_depth {
                return true;
            }
            self.cursor += 1;
        }
        self.cursor = self.trace.len().saturating_sub(1);
        true
    }

    // ── Breakpoints ───────────────────────────────────────────────────────

    /// Advances the cursor to the next frame that matches a registered breakpoint
    /// (respecting conditions, hit counts, and log-points).
    ///
    /// Returns a reference to that frame, or `None` if no breakpoint is hit.
    pub fn continue_to_breakpoint(&mut self) -> Option<&TraceEntry> {
        self.pending_logs.clear();
        self.cursor += 1;
        while self.cursor < self.trace.len() {
            let line = self.trace[self.cursor].line;
            if let Some(bp) = self.breakpoints.get_mut(&line) {
                bp.hit_count += 1;

                // Check hit-count condition (e.g. ">5", "==3", "10" meaning ==10).
                if let Some(ref hc) = bp.hit_condition {
                    if !check_hit_condition(bp.hit_count, hc) {
                        self.cursor += 1;
                        continue;
                    }
                }

                // Check expression condition.
                if let Some(ref cond) = bp.condition {
                    let vars = &self.trace[self.cursor].variables;
                    if !evaluate_condition(vars, cond) {
                        self.cursor += 1;
                        continue;
                    }
                }

                // Log-point: emit message, don't stop.
                if let Some(ref msg) = bp.log_message {
                    let vars = &self.trace[self.cursor].variables;
                    let rendered = interpolate_log_message(msg, vars);
                    self.pending_logs.push(rendered);
                    self.cursor += 1;
                    continue;
                }

                return self.trace.get(self.cursor);
            }
            self.cursor += 1;
        }
        None
    }

    // ── Variable mutation ─────────────────────────────────────────────────

    /// Sets a variable's display value in the current frame.
    /// Returns `true` if the variable was found and updated.
    pub fn set_variable(&mut self, name: &str, value: &str) -> bool {
        if self.cursor >= self.trace.len() {
            return false;
        }
        let frame = &mut self.trace[self.cursor];
        for (n, v) in &mut frame.variables {
            if n == name {
                *v = value.to_owned();
                return true;
            }
        }
        false
    }

    // ── Query helpers ─────────────────────────────────────────────────────

    /// Returns all recorded trace entries.
    pub fn all_frames(&self) -> &[TraceEntry] {
        &self.trace
    }

    /// Returns a simulated call stack from the current cursor position.
    /// Walks backwards through the trace to find distinct function frames.
    pub fn all_visible_frames(&self) -> Vec<&TraceEntry> {
        if self.cursor >= self.trace.len() {
            return Vec::new();
        }
        let current = &self.trace[self.cursor];
        let mut frames = vec![current];

        // Walk backward to find caller frames (different function names)
        let mut seen_funcs = std::collections::HashSet::new();
        seen_funcs.insert(&current.func_name);

        for i in (0..self.cursor).rev() {
            let entry = &self.trace[i];
            if !seen_funcs.contains(&entry.func_name) {
                seen_funcs.insert(&entry.func_name);
                frames.push(entry);
            }
        }
        frames
    }

    /// Returns a list of variable names in scope at the current frame.
    /// Useful for providing debug-console completions.
    pub fn completions(&self) -> Vec<String> {
        self.current_frame()
            .map(|f| f.variables.iter().map(|(n, _)| n.clone()).collect())
            .unwrap_or_default()
    }

    /// Returns `true` when the cursor is at or past the last trace entry.
    pub fn is_finished(&self) -> bool {
        self.cursor >= self.trace.len()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Checks whether `count` satisfies a hit-condition string.
/// Supported forms: `"5"` (== 5), `">5"`, `">=5"`, `"<5"`, `"<=5"`, `"==5"`, `"%3"` (every 3rd).
fn check_hit_condition(count: u64, cond: &str) -> bool {
    let cond = cond.trim();
    if cond.is_empty() {
        return true;
    }

    if let Some(rest) = cond.strip_prefix(">=") {
        rest.trim().parse::<u64>().map_or(true, |n| count >= n)
    } else if let Some(rest) = cond.strip_prefix(">") {
        rest.trim().parse::<u64>().map_or(true, |n| count > n)
    } else if let Some(rest) = cond.strip_prefix("<=") {
        rest.trim().parse::<u64>().map_or(true, |n| count <= n)
    } else if let Some(rest) = cond.strip_prefix("<") {
        rest.trim().parse::<u64>().map_or(true, |n| count < n)
    } else if let Some(rest) = cond.strip_prefix("==") {
        rest.trim().parse::<u64>().map_or(true, |n| count == n)
    } else if let Some(rest) = cond.strip_prefix('%') {
        rest.trim()
            .parse::<u64>()
            .map_or(true, |n| n > 0 && count % n == 0)
    } else {
        // Plain number means == n.
        cond.parse::<u64>().map_or(true, |n| count == n)
    }
}

/// Evaluates a condition expression against the current variable snapshot.
/// Returns `true` if the expression evaluates to `true` (or the evaluation fails).
fn evaluate_condition(vars: &[(String, String)], expr: &str) -> bool {
    // Try simple comparisons first: "x > 5", "name == \"foo\""
    for op in &["==", "!=", ">=", "<=", ">", "<"] {
        if let Some(idx) = expr.find(op) {
            let lhs = expr[..idx].trim();
            let rhs = expr[idx + op.len()..].trim();
            if let Some(lhs_val) = find_var(vars, lhs) {
                return compare_values(&lhs_val, op, rhs);
            }
        }
    }
    // Fall back: if the expression is a variable name that is "true", fire.
    if let Some(val) = find_var(vars, expr.trim()) {
        return val == "true";
    }
    // Can't evaluate — default to firing.
    true
}

fn find_var(vars: &[(String, String)], name: &str) -> Option<String> {
    vars.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone())
}

fn compare_values(lhs: &str, op: &str, rhs: &str) -> bool {
    // Try numeric comparison first.
    if let (Ok(l), Ok(r)) = (lhs.parse::<f64>(), rhs.parse::<f64>()) {
        return match op {
            "==" => (l - r).abs() < f64::EPSILON,
            "!=" => (l - r).abs() >= f64::EPSILON,
            ">" => l > r,
            ">=" => l >= r,
            "<" => l < r,
            "<=" => l <= r,
            _ => false,
        };
    }
    // String comparison (strip surrounding quotes from rhs if present).
    let rhs_clean = rhs.trim_matches('"');
    match op {
        "==" => lhs == rhs_clean,
        "!=" => lhs != rhs_clean,
        _ => false,
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    const SIMPLE_SRC: &str = "def main() -> i64 {\n    val x = 1\n    val y = 2\n    x + y\n}\n";

    fn started_session() -> DebugSession {
        let mut s = DebugSession::new();
        s.set_source(SIMPLE_SRC);
        s.start().expect("start failed");
        s
    }

    #[test]
    fn new_session_is_empty() {
        let s = DebugSession::new();
        assert_eq!(s.trace_len(), 0);
        assert!(s.current_frame().is_none());
        assert!(s.is_finished());
    }

    #[test]
    fn set_source_clears_trace() {
        let mut s = started_session();
        assert!(s.trace_len() > 0);
        s.set_source(SIMPLE_SRC);
        assert_eq!(s.trace_len(), 0);
    }

    #[test]
    fn start_populates_trace() {
        let s = started_session();
        assert!(s.trace_len() > 0);
        assert!(s.current_frame().is_some());
    }

    #[test]
    fn step_advances_cursor() {
        let mut s = started_session();
        let start = s.cursor();
        let advanced = s.step();
        assert!(advanced);
        assert_eq!(s.cursor(), start + 1);
    }

    #[test]
    fn step_back_returns_to_previous() {
        let mut s = started_session();
        s.step();
        let pos = s.cursor();
        let went_back = s.step_back();
        assert!(went_back);
        assert_eq!(s.cursor(), pos - 1);
    }

    #[test]
    fn step_back_at_start_returns_false() {
        let mut s = started_session();
        assert!(!s.step_back());
        assert_eq!(s.cursor(), 0);
    }

    #[test]
    fn continue_to_breakpoint_no_bp_returns_none() {
        let mut s = started_session();
        let result = s.continue_to_breakpoint();
        assert!(result.is_none());
        assert!(s.is_finished());
    }

    #[test]
    fn continue_to_breakpoint_hits_registered_bp() {
        let mut s = started_session();
        // Set breakpoints on every line — at least one should be hit.
        for line in 1..=5 {
            s.set_breakpoint(line, None);
        }
        // Restart cursor to beginning.
        s.set_source(SIMPLE_SRC);
        s.start().unwrap();
        let hit = s.continue_to_breakpoint();
        assert!(hit.is_some());
    }

    #[test]
    fn remove_breakpoint_prevents_stop() {
        let mut s = DebugSession::new();
        s.set_source(SIMPLE_SRC);
        for line in 1..=5 {
            s.set_breakpoint(line, None);
        }
        s.start().unwrap();
        // Remove all breakpoints, then continue should run to end.
        s.clear_breakpoints();
        let result = s.continue_to_breakpoint();
        assert!(result.is_none());
    }

    #[test]
    fn set_variable_updates_display_value() {
        let mut s = started_session();
        // Walk forward until we find a frame with variables.
        let has_vars = (0..s.trace_len()).any(|_| {
            s.step();
            s.current_frame()
                .map(|f| !f.variables.is_empty())
                .unwrap_or(false)
        });
        if !has_vars {
            return; // No named variables in this trace — skip.
        }
        if let Some(frame) = s.current_frame() {
            if let Some((name, _)) = frame.variables.first() {
                let name = name.clone();
                let ok = s.set_variable(&name, "999");
                assert!(ok);
                let updated = s.current_frame().unwrap();
                assert_eq!(
                    updated
                        .variables
                        .iter()
                        .find(|(n, _)| n == &name)
                        .unwrap()
                        .1,
                    "999"
                );
            }
        }
    }

    #[test]
    fn all_visible_frames_nonempty_when_running() {
        let s = started_session();
        let frames = s.all_visible_frames();
        assert!(!frames.is_empty());
    }

    #[test]
    fn step_over_skips_nested_frames() {
        let src = "def inner(x: i64) -> i64 { x + 1 }\ndef main() -> i64 { inner(5) }\n";
        let mut s = DebugSession::new();
        s.set_source(src);
        s.start().unwrap();
        let depth_before = s.current_frame().map(|f| f.depth).unwrap_or(0);
        let advanced = s.step_over();
        if advanced {
            let depth_after = s.current_frame().map(|f| f.depth).unwrap_or(0);
            assert!(depth_after <= depth_before);
        }
    }

    #[test]
    fn hit_condition_count_check() {
        assert!(super::check_hit_condition(5, ">4"));
        assert!(!super::check_hit_condition(3, ">4"));
        assert!(super::check_hit_condition(3, "==3"));
        assert!(super::check_hit_condition(6, "%3"));
        assert!(!super::check_hit_condition(5, "%3"));
        assert!(super::check_hit_condition(10, ">=10"));
        assert!(!super::check_hit_condition(9, ">=10"));
    }

    #[test]
    fn log_message_interpolation() {
        let vars = vec![("x".to_owned(), "42".to_owned())];
        assert_eq!(super::interpolate_log_message("x = {x}", &vars), "x = 42");
        assert_eq!(
            super::interpolate_log_message("no vars here", &vars),
            "no vars here"
        );
        assert_eq!(
            super::interpolate_log_message("{unknown}", &vars),
            "{unknown}"
        );
    }

    #[test]
    fn compare_values_numeric() {
        assert!(super::compare_values("10", ">", "5"));
        assert!(!super::compare_values("3", ">", "5"));
        assert!(super::compare_values("5", "==", "5"));
        assert!(super::compare_values("3.5", "<", "4.0"));
    }

    #[test]
    fn compare_values_string() {
        assert!(super::compare_values("hello", "==", "\"hello\""));
        assert!(!super::compare_values("hello", "==", "\"world\""));
        assert!(super::compare_values("abc", "!=", "\"xyz\""));
    }

    #[test]
    fn start_invalid_source_returns_error() {
        let mut s = DebugSession::new();
        s.set_source("def broken( -> { }");
        assert!(s.start().is_err());
        assert_eq!(s.trace_len(), 0);
    }
}

/// Interpolates `{varname}` placeholders in a log-point message.
fn interpolate_log_message(msg: &str, vars: &[(String, String)]) -> String {
    let mut result = String::new();
    let mut chars = msg.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            let mut name = String::new();
            for ch in chars.by_ref() {
                if ch == '}' {
                    break;
                }
                name.push(ch);
            }
            if let Some(val) = find_var(vars, name.trim()) {
                result.push_str(&val);
            } else {
                result.push('{');
                result.push_str(&name);
                result.push('}');
            }
        } else {
            result.push(c);
        }
    }
    result
}
