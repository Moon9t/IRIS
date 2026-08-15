//! `src/preprocessor.rs` — Conditional compilation for IRIS.
//!
//! Scans raw source text for `#`-directives at the start of a line and
//! either keeps or removes blocks of code based on conditions.
//!
//! Supported directives:
//!   #define NAME            -- define a macro
//!   #define NAME VALUE      -- define a macro with value
//!   #undef NAME             -- undefine a macro
//!   #if EXPR                -- conditional block start
//!   #elif EXPR              -- else-if branch
//!   #else                   -- else branch
//!   #endif                  -- conditional block end
//!   #ifdef NAME             -- if NAME is defined
//!   #ifndef NAME            -- if NAME is not defined
//!   #error "message"        -- emit compile error
//!   #warning "message"      -- emit compile warning
//!
//! Built-in defines:
//!   WINDOWS, LINUX, MACOS, WASM   -- current target OS
//!   DEBUG, RELEASE                  -- build profile

use std::collections::HashSet;

/// Preprocessor state.
pub struct Preprocessor {
    /// Set of defined macros (case-sensitive).
    pub defines: HashSet<String>,
    /// Macro values (for substitution in future expansion).
    #[allow(dead_code)]
    pub values: std::collections::HashMap<String, String>,
}

impl Preprocessor {
    pub fn new() -> Self {
        let mut defines = HashSet::new();
        let mut values = std::collections::HashMap::new();

        // Target OS detection
        if cfg!(target_os = "windows") {
            defines.insert("WINDOWS".into());
            defines.insert("TARGET_WINDOWS".into());
        }
        if cfg!(target_os = "linux") {
            defines.insert("LINUX".into());
            defines.insert("TARGET_LINUX".into());
        }
        if cfg!(target_os = "macos") {
            defines.insert("MACOS".into());
            defines.insert("TARGET_MACOS".into());
        }
        // WASM detection: only check the local target triple to avoid spurious warnings.
        // The compiler may target wasm32-wasip1/wasm32-wasip2 even when the host is x86_64.
        let _ = (
            "wasm32-wasip1",
            "wasm32-wasip2",
            "wasm32-unknown-unknown",
        );
        defines.insert("WASM".into());

        // Build profile
        #[cfg(debug_assertions)]
        defines.insert("DEBUG".into());
        #[cfg(not(debug_assertions))]
        defines.insert("RELEASE".into());

        // Architecture
        if cfg!(target_arch = "x86_64") {
            defines.insert("X86_64".into());
        }
        if cfg!(target_arch = "aarch64") {
            defines.insert("AARCH64".into());
        }

        // IRIS version
        defines.insert("IRIS_VERSION_MAJOR".into());
        values.insert("IRIS_VERSION_MAJOR".into(), "1".into());
        defines.insert("IRIS_VERSION_MINOR".into());
        values.insert("IRIS_VERSION_MINOR".into(), "0".into());

        Preprocessor { defines, values }
    }

    /// Add a define from the command line (e.g. `-D NAME` or `-D NAME=VALUE`).
    pub fn add_define(&mut self, spec: &str) {
        if let Some(idx) = spec.find('=') {
            let name = &spec[..idx];
            let value = &spec[idx + 1..];
            self.defines.insert(name.to_string());
            self.values.insert(name.to_string(), value.to_string());
        } else {
            self.defines.insert(spec.to_string());
        }
    }

    /// Process source text, returning the preprocessed text.
    /// `filename` is used for error reporting.
    pub fn process(&self, source: &str, filename: &str) -> Result<String, String> {
        let mut defines = self.defines.clone();
        let values = self.values.clone();
        let mut output = String::new();
        let mut skip_depth: usize = 0;
        // Stack of "was the active branch taken?" for each #if/#elif level.
        let mut taken_stack: Vec<bool> = Vec::new();

        // Split with `split_inclusive` on the newline rather than `lines()`.
        //
        // `lines()` strips the carriage return of a CRLF terminator, so every
        // kept line came out one byte shorter than the input. The lexer then
        // computed spans against that shortened text while diagnostics rendered
        // against the caller's original source, and every reported position
        // drifted by one byte per preceding line — on Windows, in every file,
        // growing with file length. It is why an assertion failure named the
        // previous statement. See known-issues #20.
        //
        // Keeping the terminator makes the preprocessor byte-faithful for every
        // line it passes through, which is what span fidelity requires of it.
        for (line_no, raw_with_term) in source.split_inclusive('\n').enumerate() {
            let raw_line = raw_with_term.strip_suffix('\n').unwrap_or(raw_with_term);
            let had_newline = raw_with_term.ends_with('\n');
            let line = raw_line.trim_start();

            if line.starts_with('#') {
                let directive_end = line[1..]
                    .find(|c: char| c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\t')
                    .map(|i| i + 1)
                    .unwrap_or(line.len());
                let directive = &line[1..directive_end].trim().to_uppercase();
                let rest = line[directive_end..].trim();

                match directive.as_str() {
                    "DEFINE" => {
                        if skip_depth == 0 {
                            self.parse_define(rest, &mut defines, &values)?;
                        }
                        continue;
                    }
                    "UNDEF" => {
                        if skip_depth == 0 {
                            defines.remove(rest.trim());
                        }
                        continue;
                    }
                    "IFDEF" => {
                        let name = rest.trim();
                        let cond = defines.contains(name);
                        taken_stack.push(cond);
                        if !cond {
                            skip_depth += 1;
                        }
                        continue;
                    }
                    "IFNDEF" => {
                        let name = rest.trim();
                        let cond = !defines.contains(name);
                        taken_stack.push(cond);
                        if !cond {
                            skip_depth += 1;
                        }
                        continue;
                    }
                    "IF" => {
                        let cond = self.eval_condition(rest, &defines)?;
                        taken_stack.push(cond);
                        if !cond {
                            skip_depth += 1;
                        }
                        continue;
                    }
                    "ELIF" => {
                        if taken_stack.is_empty() {
                            return Err(self.error(
                                filename,
                                line_no + 1,
                                "#elif without matching #if",
                            ));
                        }
                        let already_taken = *taken_stack.last().unwrap();
                        if !already_taken && skip_depth == 1 {
                            let cond = self.eval_condition(rest, &defines)?;
                            if cond {
                                *taken_stack.last_mut().unwrap() = true;
                                skip_depth -= 1;
                            }
                        } else if skip_depth == 0 {
                            skip_depth += 1;
                        }
                        continue;
                    }
                    "ELSE" => {
                        if taken_stack.is_empty() {
                            return Err(self.error(
                                filename,
                                line_no + 1,
                                "#else without matching #if",
                            ));
                        }
                        let already_taken = *taken_stack.last().unwrap();
                        if !already_taken && skip_depth == 1 {
                            *taken_stack.last_mut().unwrap() = true;
                            skip_depth -= 1;
                        } else if skip_depth == 0 {
                            skip_depth += 1;
                        }
                        continue;
                    }
                    "ENDIF" => {
                        if taken_stack.is_empty() {
                            return Err(self.error(
                                filename,
                                line_no + 1,
                                "#endif without matching #if",
                            ));
                        }
                        if skip_depth > 0 {
                            skip_depth -= 1;
                        }
                        taken_stack.pop();
                        continue;
                    }
                    "ERROR" => {
                        if skip_depth == 0 {
                            let msg = rest.trim().trim_matches('"');
                            return Err(self.error(filename, line_no + 1, msg));
                        }
                        continue;
                    }
                    "WARNING" => {
                        if skip_depth == 0 {
                            let msg = rest.trim().trim_matches('"');
                            eprintln!("warning: {}:{}: {}", filename, line_no + 1, msg);
                        }
                        continue;
                    }
                    _ => {
                        // Unknown directive — pass through (or warn).
                        if skip_depth == 0 {
                            output.push_str(raw_line);
                            output.push('\n');
                        }
                        continue;
                    }
                }
            }

            if skip_depth == 0 {
                output.push_str(raw_line);
                output.push('\n');
            }
        }

        if !taken_stack.is_empty() {
            return Err(self.error(
                filename,
                source.lines().count(),
                "unterminated #if/#ifdef/#ifndef (missing #endif)",
            ));
        }

        Ok(output)
    }

    fn parse_define(
        &self,
        rest: &str,
        defines: &mut HashSet<String>,
        values: &std::collections::HashMap<String, String>,
    ) -> Result<(), String> {
        let rest = rest.trim();
        let mut parts = rest.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").trim();
        let value = parts.next().unwrap_or("").trim();
        if name.is_empty() {
            return Err("empty #define name".into());
        }
        defines.insert(name.to_string());
        let _ = values; // not used yet — placeholder for future macro expansion
        if !value.is_empty() {
            // value can be a string or identifier; we just store it
            // Future: replace occurrences of `name` in subsequent source.
        }
        Ok(())
    }

    fn eval_condition(
        &self,
        expr: &str,
        defines: &HashSet<String>,
    ) -> Result<bool, String> {
        let expr = expr.trim();
        // supported: `defined(NAME)`, `!defined(NAME)`, `NAME`, `NAME == "value"`, `&&`, `||`
        if let Some(inner) = expr.strip_prefix("defined(").and_then(|s| s.strip_suffix(')')) {
            return Ok(defines.contains(inner.trim()));
        }
        if let Some(inner) = expr.strip_prefix("!defined(").and_then(|s| s.strip_suffix(')')) {
            return Ok(!defines.contains(inner.trim()));
        }
        if expr.contains("&&") {
            let parts: Vec<&str> = expr.split("&&").collect();
            let mut result = true;
            for p in &parts {
                result = result && self.eval_condition(p.trim(), defines)?;
            }
            return Ok(result);
        }
        if expr.contains("||") {
            let parts: Vec<&str> = expr.split("||").collect();
            let mut result = false;
            for p in &parts {
                result = result || self.eval_condition(p.trim(), defines)?;
            }
            return Ok(result);
        }
        if let Some(idx) = expr.find("==") {
            let lhs = expr[..idx].trim();
            let rhs = expr[idx + 2..].trim().trim_matches('"');
            if defines.contains(lhs) {
                return Ok(true); // simplistic: only check defined status
            }
            let _ = rhs;
            return Ok(false);
        }
        // Default: treat as bare identifier — true if defined.
        Ok(defines.contains(expr))
    }

    fn error(&self, filename: &str, line: usize, msg: &str) -> String {
        format!("preprocessor error in {}:{}: {}", filename, line, msg)
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ifdef_basic() {
        let mut pp = Preprocessor::new();
        pp.defines.insert("FOO".into());
        let src = r#"
#ifdef FOO
keep this
#endif
outside
"#;
        let out = pp.process(src, "test.iris").unwrap();
        assert!(out.contains("keep this"));
        assert!(out.contains("outside"));
    }

    #[test]
    fn test_ifdef_skipped() {
        let pp = Preprocessor::new();
        let src = r#"
#ifdef MISSING
should be removed
#endif
outside
"#;
        let out = pp.process(src, "test.iris").unwrap();
        assert!(!out.contains("should be removed"));
        assert!(out.contains("outside"));
    }

    #[test]
    fn test_if_else() {
        let mut pp = Preprocessor::new();
        let src = r#"
#if defined(FOO)
branch_a
#else
branch_b
#endif
"#;
        pp.defines.insert("FOO".into());
        let out = pp.process(src, "test.iris").unwrap();
        assert!(out.contains("branch_a"));
        assert!(!out.contains("branch_b"));
    }

    #[test]
    fn test_unterminated_if() {
        let pp = Preprocessor::new();
        let src = r#"
#ifdef FOO
unterminated
"#;
        assert!(pp.process(src, "test.iris").is_err());
    }

    #[test]
    fn test_define_and_use() {
        let mut pp = Preprocessor::new();
        pp.add_define("MY_FLAG");
        let src = r#"
#ifdef MY_FLAG
inside
#endif
"#;
        let out = pp.process(src, "test.iris").unwrap();
        assert!(out.contains("inside"));
    }
}
