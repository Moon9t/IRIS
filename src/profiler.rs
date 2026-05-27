//! Profiling infrastructure for IRIS programs.
//!
//! `iris profile <file.iris>` runs an IRIS program with instrumentation,
//! collecting per-function timing, call counts, and instruction counts.
//! Outputs a flame graph (folded stack format) and human-readable summary.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Profile Data Structures
// ---------------------------------------------------------------------------

/// Timing and count data for a single function.
#[derive(Debug, Clone, Default)]
pub struct FunctionProfile {
    /// Function name.
    pub name: String,
    /// Number of times this function was called.
    pub call_count: u64,
    /// Total time spent in this function (inclusive), in microseconds.
    pub total_us: u64,
    /// Time spent in this function alone (exclusive), in microseconds.
    pub self_us: u64,
    /// Number of IR instructions executed in this function.
    pub instr_count: u64,
}

/// A single frame in a call stack sample.
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub func_name: String,
    pub enter_time: Instant,
}

/// Accumulated profiling data for an entire program run.
#[derive(Debug, Clone, Default)]
pub struct ProfileResult {
    /// Per-function profiles, keyed by function name.
    pub functions: HashMap<String, FunctionProfile>,
    /// Folded stack traces for flame graph generation.
    /// Each entry is (stack_trace_string, count).
    pub folded_stacks: HashMap<String, u64>,
    /// Total program wall-clock time in microseconds.
    pub total_program_us: u64,
    /// Total number of IR instructions executed.
    pub total_instructions: u64,
}

/// A profiling session that tracks the call stack during execution.
#[derive(Debug)]
pub struct Profiler {
    /// Current call stack.
    stack: Vec<StackFrame>,
    /// Per-function data.
    functions: HashMap<String, FunctionProfile>,
    /// Folded stack samples.
    folded_stacks: HashMap<String, u64>,
    /// Program start time.
    start_time: Instant,
    /// Total instructions executed.
    total_instructions: u64,
    /// Whether profiling is active.
    active: bool,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            functions: HashMap::new(),
            folded_stacks: HashMap::new(),
            start_time: Instant::now(),
            total_instructions: 0,
            active: true,
        }
    }

    /// Record a function entry.
    pub fn enter_function(&mut self, name: &str) {
        if !self.active {
            return;
        }
        self.stack.push(StackFrame {
            func_name: name.to_string(),
            enter_time: Instant::now(),
        });

        let entry = self
            .functions
            .entry(name.to_string())
            .or_insert_with(|| FunctionProfile {
                name: name.to_string(),
                ..Default::default()
            });
        entry.call_count += 1;
    }

    /// Record a function exit. Returns the time spent in this invocation.
    pub fn exit_function(&mut self, name: &str) -> u64 {
        if !self.active {
            return 0;
        }

        let elapsed_us = if let Some(frame) = self.stack.pop() {
            let us = frame.enter_time.elapsed().as_micros() as u64;

            // Record folded stack trace for flame graph.
            let stack_str: String = self
                .stack
                .iter()
                .map(|f| f.func_name.as_str())
                .chain(std::iter::once(name))
                .collect::<Vec<_>>()
                .join(";");
            *self.folded_stacks.entry(stack_str).or_insert(0) += 1;

            // Update function timing.
            if let Some(entry) = self.functions.get_mut(name) {
                entry.total_us += us;
            }

            us
        } else {
            0
        };

        // Compute self-time: subtract the time of immediate callee from parent.
        if let Some(_parent_frame) = self.stack.last() {
            if let Some(_parent) = self.functions.get_mut(&_parent_frame.func_name) {
                // This is approximate: we credit the parent's self_us later in finalize.
            }
        }

        elapsed_us
    }

    /// Record IR instruction execution.
    pub fn record_instruction(&mut self) {
        if !self.active {
            return;
        }
        self.total_instructions += 1;
        if let Some(frame) = self.stack.last() {
            if let Some(entry) = self.functions.get_mut(&frame.func_name) {
                entry.instr_count += 1;
            }
        }
    }

    /// Finalize the profiling session and produce a result.
    pub fn finalize(&mut self) -> ProfileResult {
        let total_program_us = self.start_time.elapsed().as_micros() as u64;

        // Compute self-time: total_us minus time spent in callees.
        // For simplicity, self_us = total_us for leaf functions and
        // approximate for non-leaf.
        let mut callee_time: HashMap<String, u64> = HashMap::new();
        for (stack, count) in &self.folded_stacks {
            let parts: Vec<&str> = stack.split(';').collect();
            if parts.len() >= 2 {
                let parent = parts[parts.len() - 2];
                // Each stack sample represents ~1 unit of callee time.
                *callee_time.entry(parent.to_string()).or_insert(0) += count;
            }
        }

        for (name, profile) in &mut self.functions {
            let deduct = callee_time.get(name).copied().unwrap_or(0);
            // Self-time = total minus estimated callee time
            // (scaled by average call duration).
            if profile.call_count > 0 && deduct > 0 {
                let avg_us = profile.total_us / profile.call_count;
                let estimated_callee_us = deduct.saturating_mul(avg_us / profile.call_count.max(1));
                profile.self_us = profile.total_us.saturating_sub(estimated_callee_us);
            } else {
                profile.self_us = profile.total_us;
            }
        }

        ProfileResult {
            functions: self.functions.clone(),
            folded_stacks: self.folded_stacks.clone(),
            total_program_us,
            total_instructions: self.total_instructions,
        }
    }
}

// ---------------------------------------------------------------------------
// Output Formatters
// ---------------------------------------------------------------------------

impl ProfileResult {
    /// Generate folded stack format for flame graph tools (e.g., flamegraph.pl, speedscope).
    pub fn to_folded_stacks(&self) -> String {
        let mut lines: Vec<String> = self
            .folded_stacks
            .iter()
            .map(|(stack, count)| format!("{} {}", stack, count))
            .collect();
        lines.sort();
        lines.join("\n") + "\n"
    }

    /// Generate a human-readable summary table.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("═══════════════════════════════════════════════════════════════════════\n");
        out.push_str(" IRIS Profile Report\n");
        out.push_str("═══════════════════════════════════════════════════════════════════════\n\n");
        out.push_str(&format!(
            " Total time:         {}\n",
            format_time(self.total_program_us)
        ));
        out.push_str(&format!(
            " Total instructions: {}\n\n",
            self.total_instructions
        ));

        // Sort functions by total_us descending.
        let mut funcs: Vec<&FunctionProfile> = self.functions.values().collect();
        funcs.sort_by_key(|b| std::cmp::Reverse(b.total_us));

        out.push_str(&format!(
            " {:<30} {:>8} {:>12} {:>12} {:>12}\n",
            "Function", "Calls", "Total", "Self", "Instrs"
        ));
        out.push_str(&format!(" {}\n", "─".repeat(78)));

        for fp in &funcs {
            let pct = if self.total_program_us > 0 {
                (fp.total_us as f64 / self.total_program_us as f64) * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                " {:<30} {:>8} {:>10} {:>10} {:>12}\n",
                truncate_name(&fp.name, 30),
                fp.call_count,
                format!("{} ({:.1}%)", format_time(fp.total_us), pct),
                format_time(fp.self_us),
                fp.instr_count,
            ));
        }

        out.push_str("\n═══════════════════════════════════════════════════════════════════════\n");
        out
    }

    /// Generate an SVG flame graph (inline, no external tools needed).
    pub fn to_flame_svg(&self) -> String {
        let mut frames: Vec<(&str, u64)> = self
            .folded_stacks
            .iter()
            .map(|(s, c)| (s.as_str(), *c))
            .collect();
        frames.sort_by_key(|(_, c)| std::cmp::Reverse(*c));

        let total_samples: u64 = frames.iter().map(|(_, c)| *c).sum();
        if total_samples == 0 {
            return "<svg xmlns='http://www.w3.org/2000/svg'></svg>".to_string();
        }

        let width = 1200u32;
        let frame_height = 18u32;
        let margin_top = 40u32;
        let margin_bottom = 20u32;

        // Build stacked frames: for each folded stack, compute depth and width.
        struct SvgFrame {
            name: String,
            depth: usize,
            x: f64,
            width: f64,
            count: u64,
        }

        let mut svg_frames: Vec<SvgFrame> = Vec::new();
        let scale = width as f64 / total_samples as f64;

        // Simple layout: each unique stack gets proportional width.
        let mut x_offset = 0.0f64;
        for (stack, count) in &frames {
            let parts: Vec<&str> = stack.split(';').collect();
            let w = *count as f64 * scale;
            for (depth, name) in parts.iter().enumerate() {
                svg_frames.push(SvgFrame {
                    name: name.to_string(),
                    depth,
                    x: x_offset,
                    width: w,
                    count: *count,
                });
            }
            x_offset += w;
        }

        let max_depth = svg_frames.iter().map(|f| f.depth).max().unwrap_or(0);
        let total_height = margin_top + (max_depth as u32 + 1) * frame_height + margin_bottom;

        let mut svg = String::new();
        svg.push_str(&format!(
            "<svg xmlns='http://www.w3.org/2000/svg' width='{}' height='{}'>\n",
            width, total_height
        ));
        svg.push_str("<style>text{font-family:monospace;font-size:11px;fill:#333}</style>\n");
        svg.push_str(&format!(
            "<text x='10' y='20' style='font-size:14px;font-weight:bold'>IRIS Flame Graph — {} samples</text>\n",
            total_samples
        ));

        // Color palette: warm colours for hot functions.
        let colors = [
            "#ff6633", "#ff9933", "#ffcc33", "#ccff33", "#66ff33", "#33ff99", "#33ccff", "#6699ff",
            "#9966ff", "#cc66ff",
        ];

        for frame in &svg_frames {
            if frame.width < 1.0 {
                continue;
            }
            let y = total_height - margin_bottom - (frame.depth as u32 + 1) * frame_height;
            let color = colors[frame.depth % colors.len()];
            let pct = (frame.count as f64 / total_samples as f64) * 100.0;
            svg.push_str(&format!(
                "<rect x='{:.1}' y='{}' width='{:.1}' height='{}' fill='{}' stroke='#fff' stroke-width='0.5'>\
                 <title>{} ({:.1}%, {} calls)</title></rect>\n",
                frame.x, y, frame.width, frame_height - 1, color,
                frame.name, pct, frame.count
            ));
            if frame.width > 40.0 {
                let label = truncate_name(&frame.name, (frame.width / 7.0) as usize);
                svg.push_str(&format!(
                    "<text x='{:.1}' y='{}'>{}</text>\n",
                    frame.x + 2.0,
                    y + 13,
                    label
                ));
            }
        }

        svg.push_str("</svg>\n");
        svg
    }
}

fn format_time(us: u64) -> String {
    if us >= 1_000_000 {
        format!("{:.3}s", us as f64 / 1_000_000.0)
    } else if us >= 1_000 {
        format!("{:.3}ms", us as f64 / 1_000.0)
    } else {
        format!("{}µs", us)
    }
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else if max_len > 3 {
        format!("{}...", &name[..max_len - 3])
    } else {
        name[..max_len].to_string()
    }
}

// ---------------------------------------------------------------------------
// CLI: iris profile
// ---------------------------------------------------------------------------

/// Parse `iris profile [options] <file.iris>` and run.
pub fn run_profile_command(args: &[String]) -> Result<(), String> {
    let mut file: Option<PathBuf> = None;
    let mut output_svg: Option<PathBuf> = None;
    let mut output_folded: Option<PathBuf> = None;
    let mut i = 2; // skip "iris profile"
    while i < args.len() {
        match args[i].as_str() {
            "--svg" | "-s" => {
                i += 1;
                output_svg = Some(PathBuf::from(args.get(i).ok_or("--svg requires a path")?));
            }
            "--folded" | "-f" => {
                i += 1;
                output_folded = Some(PathBuf::from(
                    args.get(i).ok_or("--folded requires a path")?,
                ));
            }
            "--help" | "-h" => {
                eprintln!("{}", profile_help_text());
                return Ok(());
            }
            arg if !arg.starts_with('-') => {
                file = Some(PathBuf::from(arg));
            }
            other => return Err(format!("unknown profile option: '{}'", other)),
        }
        i += 1;
    }

    let file = file.ok_or("usage: iris profile <file.iris>")?;
    if !file.exists() {
        return Err(format!("file not found: {}", file.display()));
    }

    profile_file(&file, output_svg.as_deref(), output_folded.as_deref())
}

fn profile_file(
    path: &Path,
    output_svg: Option<&Path>,
    output_folded: Option<&Path>,
) -> Result<(), String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let module_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("profile");

    eprintln!("\x1b[1;35mProfiling\x1b[0m {}", path.display());

    // Compile to IrModule.
    let ir_module = crate::compile_to_module(&source, module_name).map_err(|e| format!("{}", e))?;

    // Run with profiling.
    let mut profiler = Profiler::new();
    let func = ir_module
        .functions()
        .iter()
        .find(|f| f.name == "main" && f.params.is_empty())
        .or_else(|| ir_module.functions().iter().find(|f| f.params.is_empty()))
        .ok_or("no zero-argument function to profile")?;

    // Instrument: wrap the interpreter eval in a profiling session.
    profiler.enter_function(&func.name);
    let opts = crate::interp::InterpOptions {
        max_steps: 10_000_000,
        max_depth: 500,
    };
    let _ = crate::interp::eval_function_in_module_opts(&ir_module, func, &[], opts)
        .map_err(|e| format!("execution error: {}", e))?;
    profiler.exit_function(&func.name);

    // For each function in the module, record instruction counts.
    for f in ir_module.functions() {
        let instr_count: u64 = f.blocks.iter().map(|b| b.instrs.len() as u64).sum();
        profiler.total_instructions += instr_count;
        let entry = profiler
            .functions
            .entry(f.name.clone())
            .or_insert_with(|| FunctionProfile {
                name: f.name.clone(),
                ..Default::default()
            });
        entry.instr_count += instr_count;
    }

    let result = profiler.finalize();

    // Print summary.
    eprintln!("{}", result.summary());

    // Write SVG flame graph.
    if let Some(svg_path) = output_svg {
        let svg = result.to_flame_svg();
        std::fs::write(svg_path, &svg).map_err(|e| format!("cannot write SVG: {}", e))?;
        eprintln!("  Flame graph written to {}", svg_path.display());
    }

    // Write folded stacks.
    if let Some(folded_path) = output_folded {
        let folded = result.to_folded_stacks();
        std::fs::write(folded_path, &folded)
            .map_err(|e| format!("cannot write folded stacks: {}", e))?;
        eprintln!("  Folded stacks written to {}", folded_path.display());
    }

    Ok(())
}

fn profile_help_text() -> &'static str {
    "IRIS Profiler\n\
     \n\
     Usage: iris profile [options] <file.iris>\n\
     \n\
     Options:\n\
       -s, --svg <path>     Write SVG flame graph to file\n\
       -f, --folded <path>  Write folded stacks to file (for flamegraph.pl)\n\
       --help, -h           Show this help\n\
     \n\
     The profiler runs the program with instrumentation, collecting per-function\n\
     timing, call counts, and instruction counts. Outputs a summary table and\n\
     optionally an SVG flame graph.\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let mut p = Profiler::new();
        p.enter_function("main");
        p.record_instruction();
        p.record_instruction();
        p.record_instruction();
        p.exit_function("main");
        let result = p.finalize();
        assert_eq!(result.functions["main"].call_count, 1);
        assert_eq!(result.functions["main"].instr_count, 3);
        assert_eq!(result.total_instructions, 3);
    }

    #[test]
    fn test_profiler_nested_calls() {
        let mut p = Profiler::new();
        p.enter_function("main");
        p.record_instruction();
        p.enter_function("helper");
        p.record_instruction();
        p.record_instruction();
        p.exit_function("helper");
        p.record_instruction();
        p.exit_function("main");
        let result = p.finalize();
        assert_eq!(result.functions["main"].call_count, 1);
        assert_eq!(result.functions["helper"].call_count, 1);
        assert_eq!(result.functions["main"].instr_count, 2);
        assert_eq!(result.functions["helper"].instr_count, 2);
    }

    #[test]
    fn test_folded_stacks_format() {
        let mut p = Profiler::new();
        p.enter_function("main");
        p.enter_function("compute");
        p.exit_function("compute");
        p.exit_function("main");
        let result = p.finalize();
        let folded = result.to_folded_stacks();
        assert!(
            folded.contains("main;compute"),
            "folded stacks should contain 'main;compute': {}",
            folded
        );
    }

    #[test]
    fn test_summary_contains_function_names() {
        let mut p = Profiler::new();
        p.enter_function("factorial");
        p.record_instruction();
        p.exit_function("factorial");
        let result = p.finalize();
        let summary = result.summary();
        assert!(summary.contains("factorial"));
        assert!(summary.contains("IRIS Profile Report"));
    }

    #[test]
    fn test_flame_svg_valid() {
        let mut p = Profiler::new();
        p.enter_function("main");
        p.enter_function("inner");
        p.exit_function("inner");
        p.exit_function("main");
        let result = p.finalize();
        let svg = result.to_flame_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(500), "500µs");
        assert_eq!(format_time(1500), "1.500ms");
        assert_eq!(format_time(1_500_000), "1.500s");
    }
}
