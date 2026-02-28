//! Trace-based debugger for IRIS programs.
//!
//! [`DebugSession`] compiles and executes a program, recording a trace of
//! executed instructions with their source positions and in-scope variable
//! snapshots. The trace can then be replayed step-by-step or advanced to the
//! next breakpoint, providing offline (post-mortem) debugging without requiring
//! coroutines or unsafe threading.

use std::collections::HashSet;

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
}

/// A debug session for a single IRIS source file.
///
/// # Usage
/// ```text
/// let mut session = DebugSession::new();
/// session.set_source(src);
/// session.set_breakpoint(3);      // break at source line 3
/// session.start().unwrap();       // compile + run, collecting trace
/// if let Some(frame) = session.continue_to_breakpoint() {
///     println!("stopped at line {}", frame.line);
/// }
/// ```
pub struct DebugSession {
    source: String,
    breakpoints: HashSet<u32>,  // 1-based line numbers
    trace: Vec<TraceEntry>,
    cursor: usize,
}

impl Default for DebugSession {
    fn default() -> Self { Self::new() }
}

impl DebugSession {
    /// Creates an empty debug session.
    pub fn new() -> Self {
        Self {
            source: String::new(),
            breakpoints: HashSet::new(),
            trace: Vec::new(),
            cursor: 0,
        }
    }

    /// Sets the IRIS source code to debug.
    pub fn set_source(&mut self, src: &str) {
        self.source = src.to_owned();
        self.trace.clear();
        self.cursor = 0;
    }

    /// Registers a breakpoint at `line` (1-based).
    pub fn set_breakpoint(&mut self, line: u32) {
        self.breakpoints.insert(line);
    }

    /// Removes a breakpoint.
    pub fn remove_breakpoint(&mut self, line: u32) {
        self.breakpoints.remove(&line);
    }

    /// Compiles the source and runs it, collecting a full execution trace.
    ///
    /// After this call, use `step()` / `continue_to_breakpoint()` to walk the trace.
    pub fn start(&mut self) -> Result<(), Error> {
        self.trace.clear();
        self.cursor = 0;

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

    /// Advances the cursor by one step. Returns `false` if already at the end.
    pub fn step(&mut self) -> bool {
        if self.cursor + 1 < self.trace.len() {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    /// Advances the cursor to the next frame that matches a registered breakpoint.
    ///
    /// Returns a reference to that frame, or `None` if no breakpoint is hit before the end.
    pub fn continue_to_breakpoint(&mut self) -> Option<&TraceEntry> {
        // Advance past the current position.
        self.cursor += 1;
        while self.cursor < self.trace.len() {
            if self.breakpoints.contains(&self.trace[self.cursor].line) {
                return self.trace.get(self.cursor);
            }
            self.cursor += 1;
        }
        None
    }

    /// Returns all recorded trace entries.
    pub fn all_frames(&self) -> &[TraceEntry] {
        &self.trace
    }

    /// Returns `true` when the cursor is at or past the last trace entry.
    pub fn is_finished(&self) -> bool {
        self.cursor >= self.trace.len()
    }
}
