# IRIS Language Extension Changelog

## 1.0.3

### Bug Fixes

- **Language server no longer fails with an opaque "exited with code 1".** The
  extension picked its executable by checking only that a file existed, so a
  stale global install at `~/.iris/bin/iris.exe` was chosen even though it could
  not load — an older build still statically imported `onnxruntime.dll`,
  `tensorflow.dll`, `c10.dll` and `torch_cpu.dll`, so Windows failed it at image
  load with `STATUS_DLL_NOT_FOUND` (0xC0000135) before `main` ran. Candidates are
  now probed with `--version` and the loader status is decoded and reported.
- **A workspace build is now preferred** over a global install, so working on the
  compiler uses the binary matching the sources being edited.
- **`$iris` problem matcher was referenced by every task but never declared**, so
  compiler errors from tasks never reached the Problems panel. It is now defined
  against the real diagnostic format (`error[E0005]: …` / ` --> file:line:col`).
- **`TaskGroup.Run` does not exist** in the VS Code API; the run task set its
  group to `undefined`. esbuild does not typecheck, which is how this shipped.
- **Run/build no longer go through a shell**, so paths containing spaces are
  passed verbatim instead of being re-parsed.
- **Settings now take effect.** `iris.executablePath` and the `iris.inlayHints.*`
  flags are only read at server startup, and the change handler was an empty
  stub; both now trigger a restart.

### New Features

- **Syntax highlighting completed against `src/parser/lexer.rs`.** Added the 23
  reserved words the grammar was missing: `match`, `let`, `mod`, `by`, `effect`,
  `handle`, `resume`, `with`, `raise`, `try`, `catch`, `defer`, `select`, `yield`,
  `move`, `unsafe`, `dyn`, `defmacro`. Removed `and`, `or` and `not`, which are
  not IRIS keywords, and `layer`, `input`, `output` and `where`, which are
  contextual identifiers rather than reserved words and so mis-coloured ordinary
  variables. Loop labels are now highlighted in `break`/`continue`.
- **New commands**: Explain Error Code (seeded from the diagnostic under the
  cursor), Check Formatting, Run Tests in File, Run Benchmarks, Generate
  Documentation, Package Manager, Diagnose Toolchain, and Show Compiler Output
  for all eleven text emit kinds (`graph`, `cuda`, `cuda-ptx`, `simd`,
  `llvm-complete`, `onnx`, `tensorrt`, `pgo-*`) — previously only `ir` and `llvm`
  were reachable.
- **New settings**: `iris.sandbox`, `iris.noCache`, `iris.target`.
- **New snippets** for `trait`, `impl`, effect operations, `handle` blocks with
  and without `resume`, `defer`, `select`, labelled loops, and `assert`.

## 1.0.2

### New Features

- **Syntax highlighting**: Added support for Higher-Kinded Types (HKT) generic parameter placeholders `[_]`.
- **Language Specifications**: Highlighting for `where` clauses and angle bracket generic arguments.

## 1.0.1

### Bug Fixes

- **Syntax highlighting**: Fixed `-` arithmetic operator incorrectly matching inside `->` arrow type annotations
- **Syntax highlighting**: Fixed `!` logical-NOT operator incorrectly matching inside `!=` comparison operators
- **Grammar**: Added AIS/MAPE-K stdlib functions (`mapek_agent_init`, `mapek_step`, `mapek_run`, `homeostasis_*`, `active_inf_*`, `epistemic_*`, `safety_check`, etc.)
- **Grammar**: Added concurrency primitives (`atomic_cas`, `mutex_new`, `mutex_lock`, `mutex_unlock`, `par_for`, `par_map`, `par_reduce`)
- **Rebuild**: Clean production build of bundled extension JS

## 1.0.0

### New Features

- **Standard Library Hardening & Standardization** — Integrated actual working code of all stubs/fallbacks across `std.ais` (Agent loops, Q-learning, decision strategies) and `std.rl` (Experience replay buffers using O(1) RingReplayBuffer).
- **Generic SavedModel session runner in TensorFlow C API** — Rewrote TensorFlow dynamic shims to query SavedModel graph operations generically without hardcoded node names.
- **Flawless Ecosystem Integration** — Standardized C runtime, ML compute engines (PyTorch, TensorFlow, ONNX Runtime), SIMD/BLAS optimizations, and 34 standard library modules.
- **Passes All Tests** — Passed all 1,400+ unit and integration tests successfully with zero errors.

## 0.4.0

### New Features

- Bumped extension metadata for the IRIS 0.4.0 language release.
- Kept editor support aligned with the updated examples and ML pipeline docs.

## 0.3.0

### New Features

- **Native binary compilation** — `iris build file.iris -o output` compiles to a standalone executable via LLVM/clang; no external toolchain required when installed via the Windows installer
- **Closure support** — first-class closures with capture, higher-order functions, `list_map`, `list_filter`, `list_reduce` all invoke closures correctly
- **Merge sort and binary search** — standard library additions
- **Expanded fuzz corpus** — arithmetic, closures, concurrency, data structures, edge cases, lists, maps, methods, ML ops, pattern matching, strings, traits
- **Benchmark suite** — `benches/` directory with binary search, Collatz, hashmap, numerical, sieve, and tree benchmarks
- **Security hardening** — input validation and bounds checks in runtime
- **Profiler** — built-in performance profiling (`iris run --profile file.iris`)
- **Expanded stdlib** — ML/NN helpers (`src/stdlib/ml.iris`, `src/stdlib/nn.iris`)
- **Windows installer** — full self-contained installer bundles LLVM/clang + MinGW sysroot + VC++ runtime; installs to PATH automatically

### Improvements

- 1,384 tests passing (249 unit + 1,135 integration)
- Strength reduction and copy propagation passes
- Loop-invariant code motion (LICM) pass
- Improved LLVM IR emission: cleaner phi nodes, no double terminators
- Better diagnostics with span information
- LSP: inlay hints, code actions, and diagnostics improvements

### Bug Fixes

- Fixed phi predecessor handling after `Panic` blocks in LLVM IR (no double terminators at -O2)
- Fixed `for`/`while` body tail expression being silently dropped
- Fixed `spawn { }`, `while { }`, `for { }`, `loop { }` accepting optional trailing `;`
- Fixed `atomic(v)` alias for `atomic_new(v)`
- Fixed `Option`/`Result` unboxing in LLVM codegen

## 0.2.0

### New Features

- **Status bar** now shows real version from `iris --version` with rich tooltip (version, git commit, branch, build date, target, rustc)
- **Show Version Info** command and server-menu action — displays full GCC-style compiler info in the output panel
- **LSP best-practice diagnostics**: BP001 (long function), BP002 (missing doc comment), BP003 (too many params), BP004 (non-snake_case), BP005 (empty body), BP006 (double semicolons)
- **LSP code actions / auto-fix**: missing semicolons, type-mismatch casts, add doc comment, rename to snake_case, remove redundant semicolons, wrap in if-condition
- **C / Python / Rust FFI** builtins: `ffi_call_i64`, `ffi_call_f64`, `ffi_call_str`, `ffi_call_void`, `python_eval`, `python_exec`, `python_call`, `python_version`, `rust_lib_open`, `rust_call_i64`, `rust_call_f64`, `rust_call_void`
- **60+ new builtins** (Phase 105): async/concurrency, deque, sorted collections, bitset, OS/system, crypto/UUID, string extras, math constants, functional list operations
- Binary output now named after the source file (e.g., `hello.iris` → `hello.exe`)
- Verbose `iris --version` output: git commit, branch, build date, target, host, profile, rustc version

### Improvements

- Updated syntax grammar with all Phase 104/105/106 builtins and new types
- New snippets for FFI, error handling, concurrency, and more
- LSP completions and hover docs for all new builtins
- InlayHint and code-lens improvements
- Better error diagnostics from build/run output

### Bug Fixes

- `list_map`, `list_filter`, `list_reduce` now properly invoke closures (were stubs)
- Status bar correctly reads version from the installed iris binary

## 0.1.0

- Initial release
- Syntax highlighting for .iris files
- Language Server Protocol: hover, completions, diagnostics, goto-definition, document symbols, signature help, formatting
- Debug Adapter Protocol: breakpoints, step, variables, evaluate
- Commands: Run File (Ctrl+F5), Build Binary, Open REPL
- Snippets for common patterns
