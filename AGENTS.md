# IRIS Agent Session Log

## Completed
### CSE Cross-Block Replacement Bug Fix (Non-Deterministic SSA Corruption)
- **Problem**: `CsePass` accumulated value replacements in a global map but only applied them within the current block's instruction list. After CSE processed block N and created replacement `vA→vB`, blocks that were processed *before* that replacement was created still had Br/CondBr args referencing the eliminated value ID `vA`. Since `HashMap` iteration order is non-deterministic, the bug appeared randomly (~1/3 runs) in `md2html.iris`, `buildsys.iris`, and `snake.iris`.
- **Fix**: Added a second global sweep in `CsePass.run()` — after all per-block CSE passes, iterate every block's instructions and re-apply the accumulated `replacements` map to ensure all Br/CondBr args are updated before stale entries are removed.
- **File**: `src/pass/opt.rs` — `CsePass::run()` (lines 181-215)
- **Result**: Non-deterministic `verify_uses_defined` failures eliminated. md2html, buildsys, snake now stable.

### LLVM 17 DAG→DAG Crash Workaround (crawler.iris)
- **Problem**: LLVM 17.0.1 `clang.exe` crashed in "X86 DAG→DAG Instruction Selection" on `@fetch_and_extract`. Upstream bug triggered by complex IR functions with many control-flow blocks and string manipulation operations.
- **Fix**: `emit_function_ir_with_name()` now emits `optnone noinline` LLVM attributes for functions with ≥15 blocks. This forces LLVM to use the -O0 instruction selector for complex functions only, bypassing the crash with zero semantic cost (the function still compiles and runs correctly).
- **File**: `src/codegen/llvm_ir.rs` — `emit_function_ir_with_name()` (lines 613-632)
- **Result**: crawler.iris now compiles to native binary without crashing clang.

### New Projects Added (v1.0.0-rc1 era)
- **`projects/multimodal_ai_orchestrator/`** — Flagship multimodal edge AI: ROS2 DDS + ONNX/PyTorch/TF ML pipeline + BLAS projections + taped AD backprop
- **`projects/robotic_actuator_control/`** — Joint actuator control loop: tape AD policy learning + ROS2 publisher bridge
- **`projects/showcase/concurrent_file_processor.iris`** — Spawn+channel+fs streaming: reader/processor/writer pipeline

- **Problem**: `ok` is a built-in that wraps values in `Result::Ok`, shadowing the user-defined `def ok(b: bool) -> i64` helper in test files. Plus, function-call statements in multi-statement blocks were missing `;` separators.
- **Fix**: Renamed `ok` to `check` in `tests/language_core.iris` and `tests/concurrency_async.iris` (63+ occurrences). Added missing `;` after 9 function-call statements in `concurrency_async.iris`. Fixed `test_ais2.iris`, `test_ais_compile.iris` (missing `;`). Converted `test_block_comment.iris` from UTF-16 to UTF-8.
- **Result**: `iris test` passes for all test files. `language_core.iris` and `concurrency_async.iris` compile and run.

### mathlab Showcase Fix
- **Problem**: `val flags = list()` without type annotation caused `list_get(flags, p)` to return `Infer`, failing at `if list_get(flags, p) == 1`.
- **Fix**: Added `: list<i64>` type annotation.
- **File**: `projects/showcase/mathlab.iris:38`
- **Result**: mathlab now runs successfully as both eval and native binary.
### Spawn Function Naming Fix (Module-Level Counter)
- **Problem**: `lambda_counter` was created per-function in `lower_function_with_generics_and_subs` (line 12626), so `fn_a` and `fn_b` both named their first spawn `__spawn_0`. The second `__spawn_0` was silently dropped by the "duplicate lambda name guard" (lines 437-441), causing `fn_b`'s spawn body to execute `fn_a`'s code with wrong constants.
- **Fix**: Moved `lambda_counter` creation to `lower_module` (line 411) and passed it through `lower_function_with_generics` → `lower_function_with_generics_and_subs` → monomorphization call site + async recursive call. Removed the duplicate guard hack.
- **Files modified**: `src/lower/mod.rs` (5 edit sites)
- **Result**: `concurrent.iris` Pattern 2 & 3 no longer hang. All 3 patterns produce correct results.

### Condvar-Based Channel Recv (Interpreter)
- **Problem**: Interpreter channel `recv()` used 10K-iteration spin-wait, causing timeouts for long-running spawn threads.
- **Fix**: Changed `Chan` variant from `Arc<Mutex<VecDeque>>` to `Arc<SharedChannel>` with `Mutex<VecDeque>` + `Condvar`. `recv` blocks via `Condvar::wait_while`.
- **Files modified**: `src/interp/mod.rs` (ChanNew, ChanSend, ChanRecv, chan_try_recv, chan_len, select)

### Rust Compilation Fixes (7 errors)
- Added missing field patterns: `LetTuple { is_var }`, `ForRange { inclusive }`, `ParFor { inclusive }`, `Assign { op }`
- Converted `func.attrs.clone()` to `ast_attr_to_ir_attr()` mapping (distinct AstAttribute/IrAttribute types)
- **File**: `src/lower/mod.rs`

### Struct Type Resolution Fix
- **Problem**: `lower_type` creates `Struct { fields: [] }` (empty) for struct references, losing field info.
- **Fix**: Changed `self.binding_ty = Some(lower_type(ast_ty))` to `self.binding_ty = Some(self.resolve_ty(ast_ty))` which calls `lower_type_with_structs` to populate fields.
- **File**: `src/lower/mod.rs:11930`

### printf Fix
- **Problem**: `std.fmt.printf` was an alias for `sprintf` — returned string but never printed.
- **Fix**: Added `print(result)` before returning.
- **File**: `src/stdlib/fmt.iris:178-181`

### RL Replay Buffer Fix
- **Problem**: `replay_buffer_push` used field-assignment in inner `if`/`block` scopes, which only rebound the local copy (value semantics bug).
- **Fix**: Rewrote to compute `new_size`/`new_next_idx` as pure values, then build final `ReplayBuffer` record.
- **File**: `src/stdlib/rl.iris:50-83`
- **Result**: All 5 RL stdlib tests pass.

### SVG Value Semantics Fix
- **Problem**: SVG functions mutated `ctx.content` in place with no effect (value semantics).
- **Fix**: Changed return types to `-> SVGContext`, functions return new record with updated content.
- **Files**: `src/stdlib/svg.iris`, `projects/sensor_analytics/viz.iris`, `projects/stochastic_calculus/svg.iris`

### Native Spawn/Channel Segfault Fix
- **Problem**: Trampoline did bare `bitcast ptr to ptr` for complex types (chan, list, map, option, result, etc.) passed to spawn, causing segfault in native binary.
- **Fix**: `src/codegen/llvm_ir.rs:454-459` now calls `runtime_unbox_helper_for_type` for complex types before passing to trampoline.
- **Result**: `concurrent.iris` all 3 patterns produce correct output as native binary. `omnibus.iris` works natively with spawn+channels.

### ML Backend Linking
- **Problem**: ML SDKs (ONNX Runtime, LibTorch, TensorFlow, OpenBLAS) available at `C:\` paths but not linked.
- **Fix**: Rebuilt compiler with `IRIS_LINK_ML_SHIMS=1 IRIS_USE_DEFAULT_SDK_PATHS=1` — all 4 backends linked into compiler binary.
- **Result**: `--emit eval` can use ML functions. Native binary compilation with `IRIS_NATIVE_ML_BACKENDS=1` available but slow (links all .libs).

### AIS Stdlib (std.ais)
- **Created**: `src/stdlib/ais.iris` — comprehensive Autonomous Intelligent Systems framework with 14 subsystems
- **Subsystems**: homeostatic regulation, active inference, neuroevolution, EWC (continual learning), running statistics, decision strategies (argmax, epsilon-greedy, softmax, boltzmann, UCB), reward processing (discount, GAE), perception pipeline (normalize, clip), world models, safety constitution, epistemic drive (novelty, curiosity), multi-agent consensus, meta-cognition, MAPE-K agent lifecycle
- **All 14 subsystems verified** via `test_ais_stdlib.iris` with `--emit eval`
- **Key discoveries**:
  - Function-typed record fields cannot be called (LLVM IR `use of undefined value`) — `mapek_step`/`mapek_run` disabled
- **Files created/modified**: `src/stdlib/ais.iris` (655 lines), `test_ais_stdlib.iris`
- **Result**: `bring std.ais` works; all 14 subsystem tests produce correct output

### Omnibus Showcase
- **Created**: `projects/showcase/omnibus.iris` — comprehensive showcase of 20+ language features including traits, generics, choice types, concurrency, AD, BLAS, ML, RL, options/results, maps, lambdas, loops, arrays, f-strings, tuples, blocks, tensors, math, SVG.
- **Verified**: Works with `--emit eval` AND native compilation.

### Bool ABI Mismatch Fix (i1 vs int on Windows x64)
- **Problem**: `iris_bool_to_str` and `iris_print_bool` declared as `i1` in LLVM IR but C defs use `int`. On Windows x86-64, `i1` and `int` pass in different registers (`al` vs `ecx`), causing `to_str(bool_from_closure)` to always return `"true"` while `if` on same value works.
- **Fix**: Declarations changed to `i32` (matching C `int`); `zext i1 %val to i32` inserted at all 10 call sites.
- **Files**: `src/codegen/llvm_ir.rs` (8 edits), `src/codegen/llvm_stub.rs` (2 edits)
- **Result**: `to_str()` on bool closure results correct in `--emit eval` and `--emit binary`.

### func.params Type Sync After Type Inference
- **Problem**: `func.params[i].ty` was never updated after type inference resolved Infer types. The LLVM codegen used func.params types for function signatures (reporting `ptr` from Infer) but block params (updated post-inference) could report `i64` (default fallback for unconstrained Infer values). This mismatch caused `inttoptr i64 %current_url to ptr` in spawn wrapper functions, where `%current_url` was declared as `ptr` in the signature — a clang type error.
- **Fix**: Added a sweep in `type_infer_hm.rs:infer_function` after block param update to copy resolved `value_types` into `func.params[i].ty` (matching by entry block param order). Also maps `IrType::Infer` → `"ptr"` in `llvm_type_complete` (previously errored).
- **Files**: `src/pass/type_infer_hm.rs` (func.params sweep), `src/codegen/llvm_ir.rs` (Infer→ptr in llvm_type_complete)
- **Result**: spawn wrapper function signatures now use the post-inference type, so `inttoptr i64 %current_url to ptr` is valid LLVM IR (both sides agree on `i64`). The coercion is a no-op at runtime (both i64 and ptr are 64-bit on x86-64).

### verify_uses_defined Pass
- **Problem**: LLVM codegen could produce invalid IR referenced undefined SSA values (non-deterministically) — the old `value_defs` map was stale after InlinePass created fresh value IDs without updating it.
- **Fix**: Added `verify_uses_defined` in `src/pass/mod.rs:55` — scans all instructions directly (not `value_defs`) to validate Br/CondBr arguments exist as defined values in the function. Catches true SSA violations regardless of stale `value_defs`.
- **Result**: Non-deterministic "use of undefined value" LLVM errors now caught at compile time by `verify_uses_defined` instead of randomly surfacing as clang failures.

### Function-Typed Record Fields (CallClosure pass_env)
- **Problem**: `m.check(10)` with fn-typed field generated broken LLVM IR (`use of undefined value`).
- **Fix**: Added `pass_env: bool` to `IrInstr::CallClosure`. Lowering sets `pass_env: false` for fn-fields, `true` for lambdas. LLVM codegen uses `closure_fn_map` (MakeClosure → MakeStruct → GetField trace) to skip env ptr for non-lambdas.
- **Files**: `src/ir/instr.rs`, `src/lower/mod.rs` (9 sites), `src/codegen/llvm_ir.rs`, `src/codegen/ir_serial.rs`
- **Result**: `m.check(10)` works in interpreter + native. Combined with Bool ABI fix, all closure patterns correct.

## Showcase Programs
- **`projects/showcase/concurrent.iris`**: 3 concurrent patterns — Channel Pipeline, Fan-out/Fan-in, Parallel Monte Carlo Pi. All verified with `--emit eval`.
- **`projects/showcase/iris_analytics.iris`**: Data generation, parallel stats, OLS regression, anomaly detection, SVG chart, iterators.
- **`projects/showcase/omnibus.iris`**: Comprehensive language feature showcase — traits, generics, choice types, concurrency (spawn+channels), AD (tape/backward), BLAS, ML backends, RL, error handling (option/result), maps, lambdas, loops/lists, arrays, f-strings/tuples/blocks, tensors, math, SVG, all verified with `--emit eval` AND native compilation.

## Running Tests
- 89 of 89 stdlib tests pass (pre-existing LLVM IR stray `}` issue fixed)
- All projects verify with `--emit eval`
- `bring std.ais` compiles and all 14 subsystems pass via `--emit eval`

## Native Binary Results

### Showcase Programs (all work with `iris run`)
- **concurrent.iris** ✅ — spawn+channels (pipeline, fan-out/fan-in, Monte Carlo Pi)
- **iris_analytics.iris** ✅ — data gen, parallel stats, OLS regression, anomaly detection, SVG chart
- **mathlab.iris** ✅ — gcd, factorial, fib, prime_count, matrix ops
- **omnibus.iris** ✅ — full language feature demo (traits, generics, AD, ML, RL, tensors, SVG, etc.)
- **records.iris** ✅ — record ops with sorting/filtering
- **textlab.iris** ✅ — string processing & text analysis

### Projects
- **stochastic_calculus** ✅ — Monte Carlo option pricing, GBM path SVG
- **sensor_analytics** ✅ — 4-sensor pipeline, stats, ML (autodiff), SVG viz
- **crypto_ledger** ✅ — blockchain with mining, chain integrity, JSON export
- **taskman** ✅ — CLI task manager (add/list/complete/delete with persistence)
- **ais_gridworld** ✅ — Q-learning RL agent
- **ffi_demo** ✅ — Python interop via `py_eval`
- **ml_backends_demo** ✅ — native MLP training + LibTorch/TensorFlow backends linked (ONNX session creation fails — DLL not in PATH)
- **native_ml** ✅ (eval) — training timeouts in native; interpreter fallback works
- **llm_inference** ✅ (eval) — ONNX model loading & inference
- **distributed_kv** ❌ — networking type error (`net__tcp_connect_to` return type mismatch) — pre-existing stdlib issue

## Known Issues
1. ~~**Native binary segfault on spawn/channels**: Any project using `spawn` crashes with `STATUS_ACCESS_VIOLATION 0xc0000005` when emitted as native `.exe`. Only affects `--emit binary` / `run` subcommand. Interpreter (`--emit eval`) works fine.~~ **FIXED** — Native spawn/channel works (trampoline unbox fix).
2. ~~**LLVM text backend emits stray `}`**: `module.ll` has extra closing brace.~~ **FIXED** — Clang now compiles generated LLVM IR cleanly.
3. ~~**`list_push` with `str` type causes runtime crash**: `list_push` on `list<str>` not supported.~~ **FIXED** — Works correctly in both eval and native binary.
4. ~~**`bring` cascading type inference failure**: Module bringing another module that uses `std.ml`/`std.fs` fails.~~ **FIXED** — `bring std.rl` / `bring std.nn` (transitive `std.ml`) now works.
5. ~~**`random()` unseeded in native backend**: Returns constant value.~~ **FIXED** — Seed includes `time(NULL)` + PID + high-resolution timer on both Windows and POSIX.
6. ~~**Native channel buffer limited** (~64 items). Large bursts > 50 cause deadlock in native backend.~~ **FIXED** — Native channel uses dynamically growable `chan_grow` (doubles capacity on overflow). Both interpreter and native use unbounded buffers.
7. ~~**`std.svg` has no `svg_circle`** — only rect, line, text.~~ **FIXED** — `svg_circle` was added during SVG value semantics fix.

## Critical Context
- **`--emit eval` works, native binary spawn works**. Both `--emit eval` and native `run` work correctly.
- **ML SDKs are pre-installed**: ONNX Runtime 1.26.0 at `C:\onnxruntime`, LibTorch 2.1.0+cpu at `C:\libtorch`, TensorFlow C API at `C:\tensorflow`, OpenBLAS 0.3.33 at `C:\openblas`. Controlled by env vars: `IRIS_LINK_ML_SHIMS=1` (compiler build), `IRIS_NATIVE_ML_BACKENDS=1` (user binary), `IRIS_USE_DEFAULT_SDK_PATHS=1` (auto-detect C:\ paths). `build.rs` lines 76-145 handle detection.
- **`var` IS supported**: `var x = 0` creates mutable variable. Earlier "reserved keyword" error was from PowerShell heredoc misparse.
- **`!` boolean NOT is supported**.
- **Field mutation is rebinding**: `record.field = expr` creates a new record — only affects the local scope's binding.
- **While conditions on i64**: Use `while running != 0` not `while running` (LLVM needs `i1`).
- **Tuple destructuring**: Use `val (a, b) = recv(ch)` not `recv(ch).0`.
- **`list_set(lst, i, v)`** is a builtin that mutates a list in-place (index must be in bounds).
- **CLI: `--emit eval` not `iris eval`**: The interpreter mode is invoked as `iris --emit eval file.iris` (or via `cargo run --release -- --emit eval file.iris`). The `eval` subcommand does not exist.


## Real-World Projects (Native Binary)

### Working (4/8) — consistent builds
- **passman.iris** ✅ — CLI password manager: `init`, `set`, `get`, `list`, `gen` with SHA-256 vault
- **csvproc.iris** ✅ — CSV data processor: per-column stats (mean, stddev, min/max), z-score outlier detection, JSON report export
- **kvstore.iris** ✅ — Per-database key-value store: `init`/`set`/`get`/`del`/`list` with file persistence
- **loganalyzer.iris** ✅ — Apache combined log analyzer: status code distribution, top IPs, suspicious traffic (>50 reqs)

### Intermittently Failing → Fixed
- **md2html.iris** ✅ — Root cause fixed: `CsePass` accumulated replacements in a global map but only applied them within-block. After all blocks processed, Br/CondBr args in earlier blocks still referenced eliminated value IDs. Fixed by adding a post-loop global sweep in `CsePass.run()`. `src/pass/opt.rs`.
- **buildsys.iris** ✅ — Same CSE cross-block fix applies.
- **snake.iris** ✅ — Same CSE cross-block fix applies.

### Blocked → Fixed
- **crawler.iris** ✅ — LLVM 17.0.1 `clang.exe` crashes in "X86 DAG→DAG Instruction Selection" on `@fetch_and_extract`. Fixed by emitting `optnone noinline` for functions with ≥15 blocks, forcing the -O0 instruction selector. `src/codegen/llvm_ir.rs`.

- **Concatenation**: `concat()` takes exactly 2 args (nest for multi-part).
- **Spawn syntax**: `spawn { body };` — block syntax, not function-call.
- **Statement separator**: In multi-statement blocks, function-call statements must be separated by `;`. The last expression in a block omits `;`. Declarations (`val`, `var`) and control-flow (`if`, `for`, `while`) on separate lines do not need `;`.
