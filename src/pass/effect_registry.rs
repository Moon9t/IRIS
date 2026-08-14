// Effect registry — maps builtin function names to their effect rows.
//
// Used by the effect checker to look up the effects of primitive / stdlib
// calls when inferring effect rows for user functions.

use std::collections::HashMap;

/// A row of effect tags (e.g. ["io", "alloc"]). Sorted and deduplicated.
/// Supports effect row variables for row polymorphism (e.g. `effect E`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct EffectRow {
    pub effects: Vec<String>,
    /// Effect row variables (e.g. `effect E` in a function signature).
    /// At call sites, these are instantiated with the caller's effect row.
    pub vars: Vec<String>,
}

impl EffectRow {
    pub fn pure() -> Self {
        Self { effects: vec![], vars: vec![] }
    }

    pub fn new(effects: Vec<String>) -> Self {
        let mut s = effects;
        s.sort();
        s.dedup();
        Self { effects: s, vars: vec![] }
    }

    pub fn from_strs(effects: &[&str]) -> Self {
        Self::new(effects.iter().map(|s| s.to_string()).collect())
    }

    /// Build from effect list + effect variable names.
    pub fn from_parts(effects: Vec<String>, vars: Vec<String>) -> Self {
        let mut s = effects;
        s.sort();
        s.dedup();
        Self { effects: s, vars }
    }

    pub fn is_pure(&self) -> bool {
        self.effects.is_empty() && self.vars.is_empty()
    }

    pub fn contains(&self, e: &str) -> bool {
        self.effects.iter().any(|x| x == e)
    }

    /// Returns true if every effect in `self` is also in `other` (subset relation).
    /// Effect variables are treated as always-satisfied (they'll be instantiated).
    pub fn subset(&self, other: &Self) -> bool {
        self.effects.iter().all(|e| other.effects.contains(e))
    }

    /// Set union of two effect rows.
    pub fn union(&self, other: &Self) -> Self {
        let mut combined = self.effects.clone();
        combined.extend(other.effects.clone());
        let mut vcombined = self.vars.clone();
        vcombined.extend(other.vars.clone());
        vcombined.sort();
        vcombined.dedup();
        Self { effects: Self::new(combined).effects, vars: vcombined }
    }

    /// Set intersection of two effect rows.
    pub fn intersect(&self, other: &Self) -> Self {
        let combined: Vec<String> = self
            .effects
            .iter()
            .filter(|e| other.effects.contains(e))
            .cloned()
            .collect();
        Self { effects: Self::new(combined).effects, vars: vec![] }
    }

    /// Instantiate effect variables by substituting them with a concrete row.
    /// Used at call sites: `fn_row.instantiate(&caller_row)` yields the concrete row.
    pub fn instantiate(&self, with: &Self) -> Self {
        let row = Self::from_parts(self.effects.clone(), vec![]);
        // Effect variables absorb the caller's effects.
        if self.vars.is_empty() {
            row
        } else {
            row.union(with)
        }
    }

    pub fn display(&self) -> String {
        let parts: Vec<String> = self.effects.iter().cloned().collect();
        let mut s = parts.join(", ");
        if !self.vars.is_empty() {
            if !s.is_empty() { s.push_str(", "); }
            s.push_str(&self.vars.join(", "));
        }
        if s.is_empty() { "pure".to_string() } else { s }
    }
}

pub struct EffectRegistry {
    /// Maps builtin / stdlib function name -> effect row.
    pub builtins: HashMap<String, EffectRow>,
}

impl EffectRegistry {
    pub fn new() -> Self {
        let mut r = Self {
            builtins: HashMap::new(),
        };
        r.register_stdlib();
        r
    }

    /// Look up the effect row for a function name. Returns None if not a known builtin.
    pub fn lookup(&self, name: &str) -> Option<&EffectRow> {
        self.builtins.get(name)
    }

    fn reg(&mut self, names: &[&str], row: EffectRow) {
        for n in names {
            self.builtins.insert(n.to_string(), row.clone());
        }
    }

    fn register_stdlib(&mut self) {
        // === io (terminal I/O) ===
        self.reg(
            &[
                "println",
                "print",
                "eprint",
                "eprintln",
                "std.fmt.println",
                "std.fmt.print",
                "std.io.read_line",
                "std.io.write",
                "std.io.flush",
                "std.io.getline",
                "std.term.read_key",
                "std.term.term_rows",
                "std.term.term_cols",
            ],
            EffectRow::from_strs(&["io"]),
        );

        // === io + alloc (formatted printing allocates) ===
        self.reg(
            &[
                "std.fmt.sprintf",
                "std.fmt.printf",
                "std.fmt.format",
                "std.io.putline",
                "std.string.concat",
                "sprintf",
                "format",
                "concat",
                "to_str",
            ],
            EffectRow::from_strs(&["io", "alloc"]),
        );

        // === alloc (heap allocation) ===
        //
        // Both spellings matter. The IR-level names (`list_new`, `list_push`)
        // are what the lowerer emits, but the effect checker runs on the AST
        // and therefore sees the *surface* names a programmer writes — `list`,
        // `push`, `pop`. Registering only the IR names left the single most
        // common allocation in any IRIS program invisible to the checker.
        self.reg(
            &[
                // Surface spellings (what the effect checker actually matches).
                "list",
                "push",
                "pop",
                "map",
                "list_new",
                "list_with_capacity",
                "list_push",
                "list_get",
                "list_set",
                "list_len",
                "list_pop",
                "list_insert",
                "list_remove",
                "map_new",
                "map_with_capacity",
                "map_insert",
                "map_get",
                "map_remove",
                "map_contains",
                "map_len",
                "map_keys",
                "map_values",
                "chan_new",
                "task_group_new",
                "string_concat",
                "vec_new",
                "vec_with_capacity",
                "option_new",
                "result_new",
                "some",
                "none",
                "ok",
                "err",
            ],
            EffectRow::from_strs(&["alloc"]),
        );

        // === fs (file system) ===
        self.reg(
            &[
                "file_read",
                "file_read_all",
                "file_read_lines",
                "file_read_bytes",
                "file_write",
                "file_write_all",
                "file_write_bytes",
                "file_append",
                "file_open",
                "file_close",
                "file_exists",
                "file_size",
                "file_delete",
                "file_rename",
                "dir_create",
                "dir_list",
                "dir_delete",
                "std.fs.read_file",
                "std.fs.write_file",
                "std.fs.exists",
                "std.fs.delete",
                "std.fs.list_dir",
                "std.fs.append",
                "std.fs.file_size",
                "std.fs.read_lines",
                "std.fs.read_bytes",
                "std.fs.write_bytes",
            ],
            EffectRow::from_strs(&["fs", "alloc"]),
        );

        // === net (networking) ===
        self.reg(
            &[
                "tcp_listen",
                "tcp_connect",
                "tcp_send",
                "tcp_recv",
                "tcp_close",
                "tcp_accept",
                "tcp_set_timeout",
                "udp_bind",
                "udp_send",
                "udp_recv",
                "udp_close",
                "http_get",
                "http_post",
                "http_request",
                "std.net.tcp_listen",
                "std.net.tcp_connect",
                "std.net.tcp_send",
                "std.net.tcp_recv",
                "std.net.tcp_accept",
                "std.net.tcp_close",
                "std.net.udp_bind",
                "std.net.udp_send",
                "std.net.udp_recv",
                "std.net.udp_close",
                "std.net.resolve_host",
                "std.http.get",
                "std.http.post",
                "std.http.request",
            ],
            EffectRow::from_strs(&["net", "alloc"]),
        );

        // === spawn (concurrency) ===
        self.reg(
            &[
                "spawn",
                "spawn_async",
                "chan_send",
                "chan_recv",
                "chan_close",
                "chan_try_recv",
                "chan_len",
                "chan_select",
                "task_group_spawn",
                "task_group_join",
                "task_group_cancel",
                "std.async.spawn",
                "std.async.chan_new",
                "std.async.chan_send",
                "std.async.chan_recv",
                "std.async.task_group_new",
                "std.async.task_group_spawn",
                "std.async.task_group_join",
                "std.async.task_group_cancel",
                "join",
                "cancel",
            ],
            EffectRow::from_strs(&["spawn", "alloc"]),
        );

        // === throw (exceptions/panics) ===
        self.reg(
            &[
                "panic",
                "unreachable",
                "todo",
                "assert",
                "std.assert.assert_eq",
                "std.assert.assert_ne",
                "std.assert.assert_true",
                "std.assert.assert_false",
                "std.assert.assert_some",
                "std.assert.assert_none",
                "std.assert.assert_ok",
                "std.assert.assert_err",
                "iris_panic",
                "unwrap",
                "unwrap_or",
            ],
            EffectRow::from_strs(&["throw"]),
        );

        // === ffi (foreign function interface) ===
        self.reg(
            &[
                "py_eval",
                "py_call",
                "py_import",
                "py_get",
                "py_set",
                "std.ffi.py_eval",
                "std.ffi.py_call",
                "dlopen",
                "dlsym",
                "dlclose",
                "std.ffi.dlopen",
                "std.ffi.dlsym",
                "std.ffi.dlclose",
                "iris_ffi_call",
            ],
            EffectRow::from_strs(&["ffi", "alloc"]),
        );

        // === time (clock) ===
        self.reg(
            &[
                "now",
                "now_ms",
                "now_us",
                "monotonic",
                "sleep",
                "sleep_ms",
                "sleep_us",
                "std.time.now",
                "std.time.now_ms",
                "std.time.now_us",
                "std.time.sleep",
                "std.time.sleep_ms",
                "std.time.monotonic",
            ],
            EffectRow::from_strs(&["time"]),
        );

        // === random (random numbers) ===
        self.reg(
            &[
                "random",
                "random_int",
                "random_float",
                "random_seed",
                "std.random.random",
                "std.random.random_int",
                "std.random.random_float",
                "std.random.seed",
            ],
            EffectRow::from_strs(&["random", "time"]),
        );

        // === env (environment) ===
        self.reg(
            &[
                "getenv",
                "setenv",
                "args",
                "argc",
                "argv",
                "std.env.getenv",
                "std.env.setenv",
                "std.env.args",
                "std.env.argc",
                "std.env.argv",
                "std.env.cwd",
                "std.env.set_cwd",
            ],
            EffectRow::from_strs(&["env", "alloc"]),
        );

        // === sys (system calls) ===
        self.reg(
            &[
                "system",
                "popen",
                "pclose",
                "exit",
                "abort",
                "std.sys.system",
                "std.sys.popen",
                "std.sys.pclose",
                "std.sys.exit",
                "std.sys.abort",
                "std.proc.exit",
                "std.proc.spawn",
                "kill",
                "raise",
                "signal",
            ],
            EffectRow::from_strs(&["sys", "alloc"]),
        );

        // === math (math library, pure but tracked for FP exceptions) ===
        // Pure math functions: sqrt, sin, cos, log, exp, etc.
        // Not annotated — default to pure.
    }
}

impl Default for EffectRegistry {
    fn default() -> Self {
        Self::new()
    }
}
