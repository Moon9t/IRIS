//! Interactive REPL for the IRIS DSL.
//!
//! [`ReplState`] accumulates top-level definitions and in-scope `val`/`var`
//! bindings across calls so the session feels like a live notebook.

use crate::error::Error;
use crate::EmitKind;

/// Persistent REPL session state.
///
/// Two accumulation buckets:
/// - `top_level` — `def`, `record`, `choice`, `const`, `type`, `extern`, `trait`, `impl`
/// - `context`   — `val x = expr` / `var x = expr` statements in the implicit scope
pub struct ReplState {
    top_level: Vec<String>,
    context: Vec<String>,
    eval_counter: usize,
}

impl Default for ReplState {
    fn default() -> Self { Self::new() }
}

impl ReplState {
    /// Creates an empty REPL session.
    pub fn new() -> Self {
        Self { top_level: Vec::new(), context: Vec::new(), eval_counter: 0 }
    }

    /// Clears all accumulated state, returning the session to its initial empty form.
    pub fn reset(&mut self) {
        self.top_level.clear();
        self.context.clear();
        self.eval_counter = 0;
    }

    /// Evaluates one line (or a multi-line block) of IRIS input.
    ///
    /// Returns the string result on success (for expressions) or a short
    /// acknowledgement string (for definitions/bindings/commands).
    pub fn eval(&mut self, input: &str) -> Result<String, Error> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(String::new());
        }

        // REPL meta-commands start with `:`.
        if let Some(cmd) = trimmed.strip_prefix(':') {
            return Ok(self.run_command(cmd.trim()));
        }

        let first_word = trimmed.split_whitespace().next().unwrap_or("");

        match first_word {
            "def" | "record" | "choice" | "const" | "type"
            | "extern" | "trait" | "impl" => {
                self.add_top_level(trimmed)
            }
            "val" | "var" => {
                self.add_context(trimmed)
            }
            _ => {
                self.eval_expression(trimmed)
            }
        }
    }

    /// Returns the list of active top-level definitions (for `:env`).
    pub fn top_level_defs(&self) -> &[String] { &self.top_level }

    /// Returns the list of active context bindings (for `:env`).
    pub fn context_bindings(&self) -> &[String] { &self.context }

    // ------------------------------------------------------------------
    // REPL meta-command dispatch
    // ------------------------------------------------------------------

    fn run_command(&mut self, cmd: &str) -> String {
        let (name, arg) = cmd.split_once(' ')
            .map(|(n, a)| (n, a.trim()))
            .unwrap_or((cmd, ""));

        match name {
            "help" => concat!(
                "IRIS REPL commands:\n",
                "  :help          — show this message\n",
                "  :env           — list all active bindings and definitions\n",
                "  :type <expr>   — show the inferred type of an expression\n",
                "  :bring <mod>   — load a stdlib module (e.g. :bring std.math)\n",
                "  :reset         — clear all session state\n",
                "  :quit          — exit the REPL",
            ).to_owned(),

            "env" => {
                let mut out = String::new();
                if self.top_level.is_empty() && self.context.is_empty() {
                    return "(empty session)".to_owned();
                }
                for def in &self.top_level {
                    let first_line = def.lines().next().unwrap_or(def);
                    out.push_str(&format!("  {}\n", first_line));
                }
                for binding in &self.context {
                    out.push_str(&format!("  {}\n", binding));
                }
                out.trim_end().to_owned()
            }

            "type" => {
                if arg.is_empty() {
                    return "usage: :type <expr>".to_owned();
                }
                let n = self.eval_counter;
                for (ret_ty, label) in &[
                    ("i64", "i64"), ("f64", "f64"), ("bool", "bool"), ("str", "str"),
                ] {
                    if self.try_eval_with_type(arg, ret_ty, n).is_some() {
                        return format!(": {}", label);
                    }
                }
                ": (unknown)".to_owned()
            }

            "bring" => {
                // Accept both `bring std.math` and `std.math` forms.
                let mod_spec = if arg.starts_with("std.") { arg.to_owned() }
                    else { format!("std.{}", arg) };
                let bring_line = format!("bring {}", mod_spec);
                // Validate by compiling with this bring statement.
                let test_src = format!(
                    "{}\n{}\ndef __repl_validate__() -> i64 {{ 0 }}",
                    bring_line, self.top_level.join("\n")
                );
                match crate::compile(&test_src, "repl", crate::EmitKind::Ir) {
                    Ok(_) => {
                        // Prepend to top-level so it's available in all future evals.
                        self.top_level.insert(0, bring_line.clone());
                        format!("loaded: {}", mod_spec)
                    }
                    Err(e) => format!("error: {}", e),
                }
            }

            "reset" => {
                self.reset();
                "session cleared".to_owned()
            }

            "quit" | "exit" => {
                std::process::exit(0);
            }

            _ => format!("unknown command: :{} (try :help)", name),
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn full_source_for_eval(&self, eval_fn: &str) -> String {
        let mut src = self.top_level.join("\n");
        src.push('\n');
        src.push_str(eval_fn);
        src
    }

    fn try_eval_with_type(&self, expr: &str, ret_ty: &str, n: usize) -> Option<String> {
        let ctx = self.context.join("\n    ");
        let eval_fn = if ctx.is_empty() {
            format!("def __eval_{n}() -> {ret_ty} {{\n    {expr}\n}}")
        } else {
            format!("def __eval_{n}() -> {ret_ty} {{\n    {ctx}\n    {expr}\n}}")
        };
        let src = self.full_source_for_eval(&eval_fn);
        crate::compile(&src, "repl", EmitKind::Eval).ok()
    }

    fn eval_expression(&mut self, expr: &str) -> Result<String, Error> {
        let n = self.eval_counter;
        self.eval_counter += 1;

        // Try candidate return types in order.
        for ret_ty in &["i64", "f64", "bool", "str"] {
            if let Some(result) = self.try_eval_with_type(expr, ret_ty, n) {
                return Ok(result.trim_end_matches('\n').to_owned());
            }
        }

        // All type candidates failed; run one more time to surface the real error.
        let ctx = self.context.join("\n    ");
        let eval_fn = if ctx.is_empty() {
            format!("def __eval_{n}() -> i64 {{\n    {expr}\n}}")
        } else {
            format!("def __eval_{n}() -> i64 {{\n    {ctx}\n    {expr}\n}}")
        };
        let src = self.full_source_for_eval(&eval_fn);
        // This will return the error.
        crate::compile(&src, "repl", EmitKind::Eval)?;
        // Unreachable — the line above always errors when we get here.
        Ok(String::new())
    }

    fn add_top_level(&mut self, item: &str) -> Result<String, Error> {
        // Extract a display name for the acknowledgement message.
        let display_name = extract_defined_name(item);

        self.top_level.push(item.to_owned());

        // Validate by trying to compile the accumulated source with a dummy main.
        let test_src = format!("{}\ndef __repl_validate__() -> i64 {{ 0 }}", self.top_level.join("\n"));
        if let Err(e) = crate::compile(&test_src, "repl", EmitKind::Ir) {
            // Roll back.
            self.top_level.pop();
            return Err(e);
        }

        Ok(format!("defined: {}", display_name))
    }

    fn add_context(&mut self, binding: &str) -> Result<String, Error> {
        // Extract the variable name (second token after val/var).
        let parts: Vec<&str> = binding.splitn(3, ' ').collect();
        let name = if parts.len() >= 2 {
            parts[1].trim_end_matches(':').trim_end_matches('=').trim().to_owned()
        } else {
            "?".to_owned()
        };

        self.context.push(binding.to_owned());

        // Validate: try to build a function that uses all the context.
        let ctx = self.context.join("\n    ");
        let test_src = format!(
            "{}\ndef __repl_ctx_validate__() -> i64 {{\n    {}\n    0\n}}",
            self.top_level.join("\n"),
            ctx
        );
        if let Err(e) = crate::compile(&test_src, "repl", EmitKind::Ir) {
            self.context.pop();
            return Err(e);
        }

        Ok(format!("defined: {}", name))
    }
}

/// Extracts a human-readable name from a top-level definition string.
fn extract_defined_name(item: &str) -> &str {
    let tokens: Vec<&str> = item.split_whitespace().collect();
    // For `def name(...)`, `record Name {`, `choice Name {`, etc.
    // the name is typically the second token.
    if tokens.len() >= 2 {
        // Strip trailing `(` or `{` if attached.
        tokens[1].trim_end_matches('(').trim_end_matches('{').trim()
    } else {
        tokens.first().copied().unwrap_or("?")
    }
}
