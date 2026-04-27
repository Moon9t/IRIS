# Changelog

All notable changes to the IRIS programming language are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/) conventions.

---

## [0.6.0] — Performance & Security

### Added

- **Security audit infrastructure** (`src/security.rs`) — `SecurityPolicy` with
  per-capability allow/deny flags (fs_read, fs_write, network, ffi, process),
  allowlists and blocklists, resource limits (max_file_write_bytes, max_open_files,
  max_connections). Path validation detects traversal attacks, null byte injection,
  and Windows device name abuse. Audit logging records every security-relevant
  operation with timestamps.
- **Reference-counting GC** — C runtime implements side-table reference counting
  (`iris_retain`, `iris_release`) with deep-free semantics for strings, lists,
  maps, options, and results. `iris_gc_collect` sweeps zero-count entries.
  `iris_gc_stats_allocated` / `iris_gc_stats_freed` expose statistics.
- **Copy propagation pass** (`CopyPropPass`) — eliminates duplicate `ConstInt`
  and `ConstFloat` definitions across blocks, with transitive chain resolution.
  Reduces register pressure and enables further DCE.
- **Loop-invariant code motion** (`LicmPass`) — computes dominators, detects
  natural loops via back edges, and hoists pure loop-invariant instructions to
  the loop preheader. Full CFG analysis with iterative dataflow.
- **Benchmark suite expansion** — six new real-world benchmarks: binary search,
  tree traversal, hashmap insert/lookup, Collatz conjecture, Sieve of
  Eratosthenes, and Simpson's rule numerical integration (12 total).
- **Profiler** (`iris profile <file>`) — per-function timing, call counts,
  instruction counts, folded-stack format for flamegraph.pl / speedscope,
  built-in SVG flame graph generator, human-readable summary table.
  CLI options: `--svg`, `--folded`.
- **Sandboxed FFI** — C runtime mirrors Rust-side security policy with
  `iris_sandbox_set_policy`, `iris_sandbox_check_fs_read/write`,
  `iris_sandbox_check_network`, `iris_sandbox_check_ffi`. Default-deny when
  sandbox is active.

### Changed

- **Compiler pipeline** — now includes `CopyPropPass` (after `StrengthReducePass`)
  and `LicmPass` (after `OpExpandPass`) in all pipeline paths including
  `compile_to_module`.
- **CLI** — added `profile` subcommand.

### Tests

- 36 new integration tests in `phase132.rs` covering security policy, path
  validation, audit logging, profiler lifecycle / flame graphs / edge cases,
  CopyPropPass constant dedup, LicmPass safety, full pipeline integration,
  pass manager with all v0.6.0 passes, and benchmark file existence.

---

## [0.5.0] — ML & Compute

### Added

- **Real tensor runtime** — `IrisTensor` struct in C runtime with 30+ functions:
  create, reshape, transpose, element-wise ops, matrix multiply, reductions
  (sum, mean, max, min), unary ops (relu, sigmoid, tanh, exp, log, sqrt, abs),
  print, and memory management.
- **General einsum engine** — interpreter implements full Einstein summation
  notation with arbitrary subscript strings; handles dot products, matrix
  multiply, batched matmul, transpose, trace, and arbitrary contractions.
- **Tensor codegen** — LLVM IR, LLVM stub, and CUDA backends dispatch real
  tensor operations: einsum (with matmul fast-path), unary, reshape, transpose,
  reduce. SIMD-friendly loop nests for x86/ARM.
- **Reverse-mode automatic differentiation** — tape-based backpropagation via
  three new IR instructions (`TapeRecord`, `Backward`, `TapeGrad`). Supports
  17 operations: add, sub, mul, div, neg, sin, cos, exp, log, sqrt, relu,
  sigmoid, tanh, pow, abs, identity, chain rule. Full topological-sort gradient
  propagation in the interpreter.
- **Enhanced sparse tensor ops** — `Sparsify` converts both arrays and dense
  tensors to sparse (index, value) pairs; `Densify` reconstructs dense arrays
  from sparse representation. C runtime provides `iris_tensor_sparsify`,
  `iris_sparse_to_tensor`, `iris_sparse_dot`, `iris_sparse_nnz`.
- **48 new tests** — 24 tensor tests (phase129), 17 reverse-mode AD tests
  (phase130), 7 sparse tensor tests (phase131).

### Improved

- **ONNX binary export** — already functional from prior work; verified with
  8 passing tests.
- **GPU/CUDA backend** — updated codegen dispatch for all tensor op variants.
- **SIMD codegen** — auto-vectorization paths verified for tight loops.

---

## [Unreleased] — targeting v0.3.0

### Fixed

- **Closure codegen** — rewrote lambda calling convention in LLVM backend: all
  lambdas now use uniform `(ptr %env, params...)` signature with capture
  extraction preamble at entry, fixing crashes for basic closures, captured
  variables, and higher-order function usage. Replaced stub
  `iris_call_closure` runtime function with proper `iris_closure_fn()` and
  `iris_closure_get_capture()` helpers.
- **List sort** — replaced bubble sort with stable O(n log n) merge sort in the
  C runtime.
- **Native concurrency** — three bugs fixed:
  - `ChanRecv` now properly unboxes the pointer returned by `iris_chan_recv()`
    to extract the i64 value.
  - `spawn` now passes captured variables to the trampoline function so spawned
    closures can access parent-scope bindings.
  - `println` / `print` / `eprintln` are now lowered as built-in `Print`
    instructions instead of generic calls, fixing "unresolved Infer" type errors
    when their return values were used in expressions.
- **TCP networking** — replaced stub TCP instruction handlers in the interpreter
  with real TCP calls (`TcpStream::connect`, `TcpListener::bind`, `accept`,
  `read`, `write`, `close`) via the existing `tcp_store` module.

### Added

- `ROADMAP.md` with milestones v0.3.0 through v1.0.0 and beyond.
- `STABILITY.md` — feature-tier classification (Tier 1 Stable through Tier 4
  Experimental) and 12 stability milestones for v1.0 gate.
- `SPEC.md` — draft language specification covering syntax, semantics, type
  system, builtins, concurrency model, tensor ops, module system. Finalized for
  v0.3.0 (removed "Draft" label, fixed function type grammar, added
  implementation notes).
- Fuzz testing infrastructure — targets for lexer, parser, lowerer, and
  compiler; seed corpus of 18 `.iris` programs; CI job for continuous fuzzing.
- Benchmark CI — structured result collection, baseline comparison with ≥15%
  regression detection.
- **236 unit tests** for lexer, parser, IR types, IR instructions, pass manager,
  error formatting, diagnostics (byte-to-line-col, render_error, error codes,
  span underlines, colored output, help hints).
- **1050+ integration tests** across 128 test phases covering the full compiler
  pipeline from parsing through codegen and evaluation.
- VS Code extension v0.3.0 — syntax highlighting, snippets, theme,
  configuration, README.
- Built-in function return types registered in lowerer `fn_sigs` for all
  standard builtins (`println`, `sleep_ms`, `random_i64`, `random_f64`, `len`,
  `assert`, `assert_eq`, etc.).

### Improved

- **Error diagnostics** — `render_error` now includes:
  - Error codes (`[E0001]`, `[E0100]`, etc.) in every diagnostic.
  - Full span underlines (`^^^^^^^^`) instead of single-character carets.
  - Optional filename display via `render_error_with_file`.
  - ANSI-colored output via `render_error_colored` / `render_error_colored_with_file`
    with bold-red errors, bold-blue line numbers, bold-green help notes.
  - Contextual help hints for common mistakes (e.g. `@` → "decorators not
    supported", `#` → "use // for comments", `struct` → "use record", `enum`
    → "use choice", `match` → "use when", `import` → "use bring").
  - CLI automatically uses colored output when stderr is a terminal.

---

## [0.2.0] — 2026-03-03

### Added

#### Builtins (60+)

- **HTTP** — `http_get`, `http_post`
- **JSON** — `json_stringify`, `json_parse`
- **Regex** — `regex_match`, `regex_find_all`, `regex_replace`
- **DateTime** — `datetime_now`, `datetime_timestamp`, `datetime_format`
- **OS / System** — `cwd`, `list_dir`, `mkdir`, `remove_file`, `path_join`,
  `env_get`, `env_set`, `exec_cmd`, `pid`, `exit_code`, `type_of`
- **Crypto** — `sha256`, `hash`, `hex_encode`, `hex_decode`, `base64_encode`,
  `base64_decode`
- **Random** — `random`, `random_range`, `uuid`
- **Functional list ops** — `list_map`, `list_filter`, `list_reduce`,
  `list_any`, `list_all`, `list_zip`, `list_enumerate`, `list_flatten`,
  `list_unique`, `list_reverse`, `list_sorted`, `list_sum`, `list_min`,
  `list_max`
- **Collections** — deque, sorted set, bitset, heap, queue
- **String extras** — `str_pad_left`, `str_pad_right`, `str_chars`,
  `str_bytes`, `str_count`, `char_at`
- **Math constants** — `math_pi`, `math_e`, `is_nan`, `is_inf`
- **Concurrency extras** — `chan_try_recv`, `chan_len`, `select`, `timeout`,
  `thread_count`, `atomic_load`, `atomic_store`, `atomic_add`

#### SQLite

- Full-stack database operations: `db_open`, `db_exec`, `db_query`, `db_close`
  — parser, interpreter, LLVM codegen, C runtime (bundled via rusqlite).

#### FFI

- **C FFI** — `ffi_open`, `ffi_call_i64`, `ffi_call_f64`, `ffi_call_str`,
  `ffi_call_void`, `ffi_close`
- **Python FFI** — `python_eval`, `python_exec`, `python_call`,
  `python_version`
- **Rust FFI** — `rust_lib_open`, `rust_call_i64`, `rust_call_f64`,
  `rust_call_void`

#### Standard Library (25 modules)

- `math`, `string`, `fmt`, `fs`, `json`, `csv`, `http`, `time`, `crypto`,
  `ffi`, `os`, `testing`, `log`, `iter`, `set`, `queue`, `heap`, `deque`,
  `kv` (SQLite-backed), `table`, `dataset`, `dataframe`, `path`, `async`,
  `bitset`

#### Package Manager

- `iris pkg init/add/remove/install/build/run/list` — project scaffolding,
  dependency management, registry interaction.

#### LSP Enhancements

- AST-based hover (works even when compilation fails).
- Built-in and keyword hover documentation.
- Code actions: missing-semicolon quickfix, type-mismatch cast, add doc
  comment, rename to snake_case, remove redundant semicolons, wrap in
  if-condition.
- Best-practice diagnostics: BP001–BP006 (long function, missing doc, too many
  params, non-snake_case, empty body, double semicolons).
- Inlay hints, find references, rename support, diagnostic codes.

#### DAP Debugger Enhancements

- Step-back, step-over/into/out.
- Conditional breakpoints, hit counts, log-points.
- Richer stack traces with source info, loaded sources, exception info.
- Variable mutation, restart, pause.
- Debug-console completions, exception breakpoint filters.

#### REPL Enhancements

- Colored prompts, timing display, input history.
- Commands: `:ir`, `:time`, `:history`, `:clear`, `:reset`, `:bring`.

#### Tooling & Infrastructure

- Verbose `iris --version` — GCC-style output with git commit, branch, build
  date, target, host, profile, rustc version.
- Binary output naming — `hello.iris` → `hello.exe` / `./hello`.
- Incremental compilation cache infrastructure.
- Benchmark suite — factorial, fib, list, matrix, sort, string benchmarks.
- ARM64 CI — cross-platform CI on x86_64 + ARM64 (Linux, Windows, macOS).

#### Installers

- **Windows** — portable `.zip`, WiX `.msi`, Inno Setup `.exe` (bundles
  LLVM/clang, lld, MinGW ucrt64 sysroot).
- **Linux** — curl one-liner, `.deb`, `.rpm`, AppImage.
- **macOS** — curl one-liner, `.pkg`, `.dmg`.

#### VS Code Extension 0.2.0

- Status bar with version tooltip, Show Version Info command.
- Server menu (restart/stop LSP).
- LSP best-practice diagnostics and code actions.
- Inlay hint settings, timing on run.
- Updated TextMate grammar for all new builtins and types.
- New snippets for FFI, concurrency, error handling.

### Changed

- **C runtime rewrite** — now uses clang + lld exclusively (removed GCC/MSYS2
  dependency).
- **Build metadata** — `build.rs` captures git hash, branch, dirty flag, build
  date, rustc version, target/host/profile/opt-level.
- Error recovery improvements in parser.

### Changed (license)

- License changed from MIT to GPL-2.0-or-later.

---

## [0.1.0] — 2026-02-28

### Added

#### Core Language

- **Lexer** — tokenizer for `.iris` source files.
- **Parser** — recursive-descent parser producing AST.
- **SSA IR** — block-parameter SSA (MLIR-style), no phi nodes.
- **Lowerer** — AST → IR lowering with lambda-lifting for closures.
- **Pass pipeline** — Validate, TypeInfer, ConstFold, OpExpand, DCE, CSE,
  ShapeCheck, Inline, LoopUnroll, StrengthReduce, Exhaustive, GcAnnotate.
- **Tree-walking interpreter** — `iris run` / `--emit eval`.

#### Type System

- Primitives: `i32`, `i64`, `f32`, `f64`, `bool`, `str`.
- Composite: `tensor<T, shape>`, `list<T>`, `map<K,V>`, tuples, arrays.
- Records (`record`), enums (`choice`) with variant payloads.
- Generics — `def identity[T](x: T) -> T` with monomorphization.
- Traits / Impl — `trait Printable`, `impl Printable for Point`.
- Type aliases — `type Matrix = tensor<f64, [3, 3]>`.
- Function types — `fn(i64) -> i64`.
- `option<T>`, `result<T,E>`, `?` operator.

#### Control Flow & Pattern Matching

- `if/elif/else`, `for`, `while`, `break`, `continue`, `return`.
- `when` (pattern matching) with guards, range patterns, tuple destructuring.

#### Closures & Functions

- Lambda expressions — `|x: i64| x * 2`.
- Default parameters — `def greet(name: str = "world")`.
- Global constants — `const PI: f64 = 3.14`.

#### Concurrency

- `channel<T>`, `spawn`, `par for`.
- `async/await`, `atomic<T>`, `mutex<T>`.

#### ML Features

- Automatic differentiation — `grad<T>` dual numbers, `@differentiable`.
- Sparse tensors — `sparse<T>`, `sparsify`, `densify`.

#### Strings

- F-string interpolation — `f"Hello, {name}!"`.
- Builtins: `len`, `concat`, `split`, `join`, `contains`, `starts_with`,
  `ends_with`, `trim`, `to_upper`, `to_lower`, `repeat`, `find`, `slice`,
  `str_replace`, `str_reverse`.

#### Math Builtins

- `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `pow`, `min`, `max`,
  `clamp`, `floor`, `ceil`, `round`.

#### I/O

- `print`, `read_line`, `read_i64`, `read_f64`.
- TCP/network instruction lowering (interpreter).

#### Code Generation

- `--emit ir|llvm|eval|binary|onnx|cuda|simd` (stubs for ONNX/CUDA/SIMD).
- LLVM IR codegen — target triples, string globals, 70+ runtime declarations.
- Native binary compilation — `iris build` via clang.

#### Module System

- `bring std.math`, `pub def`, multi-file compilation.

#### FFI

- `extern def` for C function declarations.
- GC refcounting basics.

#### Tooling

- **REPL** — `:help`, `:env`, `:type`, `:quit`.
- **LSP** — hover, completions, diagnostics, go-to-definition, document
  symbols, signature help, formatting.
- **DAP** — breakpoints, step, variables, evaluate.
- **CLI** — `iris run`, `iris build`, `iris repl`, `iris lsp`, `iris dap`.

#### VS Code Extension 0.1.0

- Syntax highlighting (TextMate grammar).
- LSP integration, DAP debugger integration.
- Commands: Run File (Ctrl+F5), Build Binary, Open REPL.
- Snippets for common constructs.

---

[Unreleased]: https://github.com/Moon9t/IRIS/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Moon9t/IRIS/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Moon9t/IRIS/releases/tag/v0.1.0
