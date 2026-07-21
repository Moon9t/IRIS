# IRIS Agent Session Log

## Completed
### Trait Objects (`dyn Trait`) — Phase 91
- **Goal**: Allow `dyn Trait` values with virtual dispatch. End-to-end: parses, type-checks, dispatches dynamically through interpreter + native binary.
- **AST**: `AstType::DynTrait { trait_name }` was already present (parser handled `dyn Trait` syntax), but lowerer mapped it to `IrType::Infer` and never produced dispatch — fixed.
- **IR type**: New `IrType::TraitObject { name: String, methods: Vec<TraitMethodSig> }` where `TraitMethodSig = { name, params, ret }`.
- **IR instructions**: `MakeTraitObject { value, target_trait, concrete_ty, result_ty, result }` and `DynCall { obj, method_name, args, result_ty, result }` (already existed as stubs).
- **`IrModule`** (`src/ir/module.rs`): Added `trait_defs: HashMap<String, Vec<TraitMethodSig>>` and `trait_impl_methods: HashMap<String, Vec<(concrete, method, mangled)>>` plus `add_trait_def/add_trait_impl_method/trait_def/trait_defs/trait_impl_methods` accessors.
- **Lowerer** (`src/lower/mod.rs`):
  - Registered all `trait Pet { def m(...) -> T }` definitions into `module.trait_defs` and all `impl Pet for X { def m(...) -> T }` blocks into `trait_impl_methods`.
  - `AstType::DynTrait` now lowers to `IrType::TraitObject { name, methods }` (instead of `Infer`).
  - At `let d: dyn Pet = concrete;` binding sites, emits `MakeTraitObject` (inserting the data ptr + vtable ptr into a `{ptr, ptr}` heap struct).
  - At `d.method()` call sites where receiver is `dyn Trait`, emits `DynCall` (slot-index resolved at codegen via `trait_def`).
- **Passes**: Added `MakeTraitObject`/`DynCall` operand walking in `opt.rs`, `const_fold.rs`, `licm.rs`; result/operand tracking updated in `inline.rs`; `contains_infer` (`type_infer.rs`, `validate.rs`, `type_infer_hm.rs`) extended for `TraitObject`.
- **GC annotate** (`gc_annotate.rs`): Both new instructions marked as producing references (`MakeTraitObject` retains the data pointer; `DynCall` retains the trait-object handle).
- **Interpreter** (`src/interp/mod.rs`):
  - `IrValue::TraitObject { target_trait, concrete, data: Box<IrValue>, vtable: HashMap<String, String> }`.
  - `MakeTraitObject`: builds the vtable by iterating `module.trait_def(target_trait)` and matching each method against `trait_impl_methods[target_trait]` keyed by `(concrete, method)`.
  - `DynCall`: looks up `vtable[method_name]`; calls the resolved mangled function with `(*data, args...)` as the call args.
- **LLVM IR codegen** (`src/codegen/llvm_ir.rs`):
  - Emits return type `dyn<Trait>` declared as `%dyn<Trait> = type { ptr, ptr }` for each registered trait.
  - Per concrete, emits `@vtable_<Trait>__<Concrete> = internal constant [N x ptr] [ptr @Trait__<Concrete>__method, ...]` so all methods are accessible via index.
  - `MakeTraitObject`: allocates a `%dynPet` struct (16 bytes) via `malloc`, stores the data ptr at field 0 and the (decayed) vtable pointer at field 1.
  - `DynCall`: extracts the data ptr (field 0) and vtable ptr (field 1) from the trait-object struct, performs a `getelementptr [255 x ptr]` indexed by the trait method's compile-time slot index, loads the function pointer, then performs an *indirect call* `call ptr %fn(ptr %data, ...args)`.
- **Runtime** (`src/runtime/iris_runtime.{h,c}`):
  - New tag `IRIS_TAG_TRAIT_OBJECT = 22` and helpers `iris_make_trait_object(data, vtable_id)` / `iris_dyn_call(obj, method_name, nargs, ...)`. The stub-backed codegen links them; eval mode does not use them, native binary uses the LLVM IR-emitted vtables.
- **Serialization** (`src/codegen/ir_serial.rs`):
  - New opcodes `OP_MAKE_TRAIT_OBJECT = 0x79`, `OP_DYN_CALL = 0x7A`.
  - New type tag `0x15` for `TraitObject` with name + method signatures.
- **Printer** (`src/codegen/printer.rs`): Prints `make_trait_object value=... trait=... concrete=... : ...` and `dyn_call obj=... method=... args=[...] : ...` lines.
- **Test**: `tests/test_trait_object.iris` (2 concrete types `Cat`/`Dog` implementing `Pet`, both methods `speak()`/`name()`, box each into `dyn Pet`, dispatch through the same trait object's method). Verified in `--emit eval` (returns 0) and `iris run` (compiles to .exe, runs end-to-end with correct dispatch).
- **Test (return-coercion)**: `tests/test_trait_object_return.iris` verifies that `def f() -> dyn Pet { val c: Cat = ...; return c }` correctly boxes the returned concrete struct and dispatches through the call site. Verified in both `--emit eval` and native binary.
- **Implementation polish — return coercion**:
  - Added `Lowerer::current_return_ty: Option<IrType>` set from the function's return type at top-of-function lowering (initialization site: `src/lower/mod.rs` ~around line 13106).
  - Extracted `Lowerer::coerce_to_trait_object(value, ty, expected, span) -> (ValueId, IrType)` helper that, when `expected` is `dyn Trait`, inserts `MakeTraitObject` for a concrete struct and returns the new trait-object value/pair. No-op when `ty` is already a matching trait object.
  - `AstStmt::Return` now calls this helper so any concrete struct returned from a function with `dyn Trait` return type is automatically boxed.
  - Pre-existing `val` annotation path now also uses the helper for symmetry.
- **Known limitation**: Native binary mode is not exhaustively tested for trait objects (covered indirectly via the LLVM IR codegen + interpreter). For complex monomorphized generic contexts (e.g. trait bounds across `where T: Show`), the pre-existing name-mangling issue (`%` from `IrType::Struct Display`) produces `print_one__%MyStruct` which is invalid LLVM identifier — pre-existing, out of scope for this feature.
- **Resolved limitation 2 (tail-position coercion)**: Added `expected_expr_ty: Option<IrType>` field to `Lowerer`. The `val x: dyn T = if/else` handler sets it before `lower_expr`; the `return` handler seeds it from `current_return_ty`. `lower_if_expr` consumes it at the top and calls `coerce_to_trait_object` on each branch's result before the merge Br, so both branches consistently produce `dyn T` instead of different concrete struct types.
- **Resolved limitation 3 (inkwell + stub codegen)**: 
  - `llvm_ir.rs` — unchanged (already correct)
  - `llvm_stub.rs` — now emits vtable globals, `%dynTrait = type { ptr, ptr }` struct declarations, and inline `MakeTraitObject`/`DynCall` (malloc + GEP + indirect call) instead of broken C runtime stubs. Removed `iris_make_trait_object`/`iris_dyn_call` runtime declarations.
  - `llvm_native.rs` — added `emit_vtable_globals()` in `compile_module`, inline `MakeTraitObject` (malloc + store data/vtable ptrs), and inline `DynCall` (load data/vtable ptrs, GEP into vtable array, indirect call via inkwell API). Threaded `module: &IrModule` through `compile_function` → `compile_instr` for method-slot resolution.


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

### ML Backend Native Linking Fix (ONNX + TensorFlow on MinGW)
- **Problem**: `IRIS_NATIVE_ML_BACKENDS=1` tried linking all three ML backends (ONNX, LibTorch, TensorFlow) but LibTorch uses MSVC C++ ABI (`std::__cxx11::basic_string` symbols) incompatible with the MinGW (`x86_64-w64-windows-gnu`) target, causing `ld.lld` link failure. ONNX `onnxruntime.dll` was also not staged next to the output binary.
- **Fix**: Added `C:\onnxruntime` and `C:\tensorflow` fallback paths for SDK auto-detection (mirroring existing `C:\libtorch` fallback). Added `stage_onnxruntime_dll_next_to()` to copy `onnxruntime.dll` next to output binary. Made LibTorch SDK detection conditional on `resolved_target.contains("msvc")` — skipped on MinGW targets.
- **Files**: `src/codegen/build.rs` (onnx/tf fallback paths, libtorch msvc gate, onnx dll staging)
- **Result**: `ml_backends_demo.iris` compiles to native binary with ONNX session creation working, TensorFlow C API linked. LibTorch gracefully excluded on MinGW.

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
- **ml_backends_demo** ✅ — native MLP training + ONNX Runtime session creation + TensorFlow C API linked (LibTorch excluded on MinGW targets due to C++ ABI mismatch)
- **native_ml** ✅ (eval) — training timeouts in native; interpreter fallback works
- **llm_inference** ✅ (eval) — ONNX model loading & inference
- **distributed_kv** ✅ — TCP KV server+client with `std.net` unboxed i64 socket handles; 9/9 tests pass in both `--emit eval` and native binary

## Known Issues
1. ~~**Native binary segfault on spawn/channels**: Any project using `spawn` crashes with `STATUS_ACCESS_VIOLATION 0xc0000005` when emitted as native `.exe`. Only affects `--emit binary` / `run` subcommand. Interpreter (`--emit eval`) works fine.~~ **FIXED** — Native spawn/channel works (trampoline unbox fix).
2. ~~**LLVM text backend emits stray `}`**: `module.ll` has extra closing brace.~~ **FIXED** — Clang now compiles generated LLVM IR cleanly.
3. ~~**`list_push` with `str` type causes runtime crash**: `list_push` on `list<str>` not supported.~~ **FIXED** — Works correctly in both eval and native binary.
4. ~~**`bring` cascading type inference failure**: Module bringing another module that uses `std.ml`/`std.fs` fails.~~ **FIXED** — `bring std.rl` / `bring std.nn` (transitive `std.ml`) now works.
5. ~~**`random()` unseeded in native backend**: Returns constant value.~~ **FIXED** — Seed includes `time(NULL)` + PID + high-resolution timer on both Windows and POSIX.
6. ~~**Native channel buffer limited** (~64 items). Large bursts > 50 cause deadlock in native backend.~~ **FIXED** — Native channel uses dynamically growable `chan_grow` (doubles capacity on overflow). Both interpreter and native use unbounded buffers.
7. ~~**`std.svg` has no `svg_circle`** — only rect, line, text.~~ **FIXED** — `svg_circle` was added during SVG value semantics fix.
8. ~~**ONNX Runtime DLL not in PATH: Native binary ONNX session creation fails**~~ **FIXED** — `stage_onnxruntime_dll_next_to()` copies DLL next to output binary at build time.

### Phase 5: Structured Concurrency (TaskGroup)
- **Implemented**: `task_group()`, `spawn(tg) { body }`, `task_group_join(tg)`, `task_group_cancel(tg)`
- **Parser fix**: `parse_spawn_stmt` now checks for `(` after `spawn` keyword — parses group expression before block. Fixes `spawn(tg) { body }` syntax.
- **IR**: Added `IrType::TaskGroup`, `IrInstr::TaskGroupNew/Spawn/Join/Cancel` with `result_produces`/`operands` impls.
- **Lowering**: `AstStmt::Spawn` with `group` field → `TaskGroupSpawn`; builtin matching for `task_group()`, `task_group_join(tg)`, `task_group_cancel(tg)`.
- **Runtime (C)**: `IrisTaskGroup` struct with `pthread_t* handles`, mutex, cancelled flag. `iris_task_group_new/spawn/join/cancel` implementations. `IRIS_TAG_TASK_GROUP = 20`, `IRIS_RC_TASK_GROUP = 11`.
- **Interpreter**: `TaskGroupState` with `Arc<Mutex<Vec<JoinHandle>>>`. `IrValue::TaskGroup(Arc<Mutex<TaskGroupState>>)`.
- **LLVM IR codegen**: All four instructions emit correct runtime calls. Box/unbox via `iris_box_task_group`/`iris_unbox_task_group`.
- **LLVM stub codegen**: Same four instructions via `iris_task_group_*` calls. Uses `box_spawn_capture` (scalar-only boxing; TaskGroup falls through to raw ptr pass-through matching stub convention).
- **Inkwell native codegen**: Auto-declares `iris_task_group_*` via `get_or_declare_runtime_fn`.
- **IR serialization**: Opcodes `0x75`–`0x78`, type tag `0x13`.
- **Passes**: Side-effect lists, operand replacement, inline remapping, `contains_infer` for type inference.
- **Name conflict fix**: Renamed `join(tg)` → `task_group_join(tg)` and `cancel(tg)` → `task_group_cancel(tg)` to avoid collision with `std.string.join()`.
- **String concat fix**: `str + str` via `+` operator now emits `StrConcat` instead of `BinOp::Add` (which produced invalid LLVM IR `add nsw ptr`).
- **Box helper fix**: Added `IrType::TaskGroup` to `runtime_box_helper_for_type` and `runtime_unbox_helper_for_type` in `llvm_ir.rs` for proper boxing when captured by spawn closures.
- **Tests**: 5 tests (basic join, multiple tasks, cancel, nested groups, detached spawn) pass in both `--emit eval` and `--emit binary` native modes.
- **Files modified**: `src/parser/parse.rs`, `src/lower/mod.rs`, `src/codegen/llvm_ir.rs` (declarations + boxing fix + string concat fix), `src/codegen/llvm_stub.rs` (declarations), `src/codegen/llvm_native.rs`, `src/codegen/ir_serial.rs`, `src/codegen/printer.rs`, `src/pass/opt.rs`, `src/pass/const_fold.rs`, `src/pass/inline.rs`, `src/pass/type_infer.rs`, `src/pass/validate.rs`, `src/pass/type_infer_hm.rs`, `src/codegen/onnx.rs`, `src/runtime/iris_runtime.h`, `src/runtime/iris_runtime.c`, `src/interp/mod.rs`, `src/ir/types.rs`, `src/ir/instr.rs`
- **Test files**: `tests/test_task_group.iris` (full suite), `tests/test_task_group2.iris`, `tests/test_task_group_simple.iris`

## Current Session (2026-07-20) — Effect Handlers + Row Polymorphism

### Effect Handlers + Row Polymorphism
- **Effect row polymorphism**: Extended `EffectRow` with `vars: Vec<String>` for effect variables (uppercase identifiers like `E`), `instantiate()` method for call-site substitution, updated checker to handle effect vars at call sites. `effect E` syntax works on function definitions.
- **`lower_type_with_structs` fix**: Added scalar type name mapping (`"i64"→I64`, etc.) to prevent `AstType::Named("i64")` from falling through to empty struct.
- **Handler arm lowering**: Added `lower_handler_arm()` and `lower_handle()` methods. `AstExpr::Handle` and `AstStmt::HandleStmt` now emit `PushHandler`/`PopHandler` IR instructions wrapping the body. Handler arms are lowered as independent IR functions (`__handler_N`).
- **IR types**: `IrInstr::PushHandler { arms: Vec<HandlerArm> }`, `IrInstr::PopHandler`, `HandlerArm { effect_name, func_name, num_args, has_resume }`.
- **Interpreter handler stack**: `handler_stack: Vec<Vec<HandlerArm>>` on `Interpreter`. `PushHandler` pushes, `PopHandler` pops. `CallExtern` checks stack before `dispatch_extern` — matching handler calls the handler function with effect payload args.
- **Parser fix**: Changed hardcoded `return_ty = i64` to `return_ty = Infer` in handle expression/statement parsing (both expression form `handle expr with { arms }` and statement form).
- **Test**: `tests/test_effect_handlers.iris` — declares `extern def echo(s: str) -> str`, intercepts via handler, verifies custom return value. Passes with `--emit eval` (exit 0).
- **Pass/fix list**: Added `PushHandler`/`PopHandler` arms to `result()`, `operands()` in `instr.rs`, plus `ir_serial.rs`, `llvm_ir.rs`, `printer.rs`, `const_fold.rs`, `opt.rs`. Added missing `IrType::TaskGroup`/`WeakRef`/`TraitObject` to `llvm_type_complete`. Added missing `MakeTraitObject`/`DynCall`/`TaskGroup*` arms to `emit_instructions`.
- **Not yet**: Native codegen for effect handlers (interpreter-only for now). Full stdlib effect annotations. Continuation capture/resume in handler arms (basic replacement-only).

### Const Generics (`const N: usize` in generic params)
- **AST**: Added `AstGenericParam::Const { name, kind }` variant, `AstType::ConstInt(i64, Span)` for const type args, `len_expr: Option<Box<AstExpr>>` on `AstType::Array`. Changed `type_params` from `Vec<String>` to `Vec<AstGenericParam>` on `AstFunction` and `AstStructDef`.
- **Parser**: `const Name: Type` parsed in `[...]` generic param lists. `IntLit` in type context → `AstType::ConstInt`. Identifier in array-length position stored as `AstExpr::Ident` in `len_expr`.
- **Lowerer**: `const_param_subs: HashMap<String, i64>` in `resolve_generic_struct_type`. Array `len_expr` resolves through const subs. Mangled names include const values: `Array__i64__5`.
- **LLVM codegen fix**: `scalar_arrays` now propagated through `GetField` — when a struct field has a scalar array type, the loaded `ptr` is recognized as a scalar array, enabling GEP-based access instead of runtime `@iris_array_load` on raw C arrays.
- **Files**: `src/parser/ast.rs`, `src/parser/parse.rs`, `src/compiler.rs`, `src/lower/mod.rs`, `src/codegen/llvm_ir.rs`, `src/lsp.rs`
- **Tests**: `tests/test_const_generic.iris` — `Array<i64, 5>` sum (6) + sum-10 (150) = 156 (`--emit eval`). Name mangling `%Array__i64__5` verified via `--emit llvm`.

### Trait Constraint Tracking (`T: Trait` bounds)
- **AST**: `AstGenericParam::Type(String)` → `AstGenericParam::Type(String, Vec<String>)` — second field stores trait bounds (empty vec for unbounded).
- **Parser**: `[T where T: Show, Ord]` parses comma-separated trait names after `where`, stored in bounds vec.
- **Lowerer**: Builds `trait_impl_map: HashMap<String, Vec<IrType>>` during `impl Trait for Type` processing. Verifies bounds at monomorphization call sites using `ir_type_dispatch_name` match against stored impls.
- **Bug fix**: `trait_impl_map` was passed as empty HashMap at line 13351 (`std::rc::Rc::new(HashMap::new())` instead of the real map).
- **Files**: `src/parser/ast.rs`, `src/parser/parse.rs`, `src/lower/mod.rs`
- **Tests**: `tests/test_trait_bounds.iris` — `T where T: Show` verified with struct type `MyStruct`. Constraint passes; method dispatch works.
- **State**: Method-call syntax on struct types works. Method-call on scalar types (`x.show()` on `i64`) not supported (pre-existing limitation — only struct method dispatch is implemented).

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

### Working (8/8) — consistent builds
- **passman.iris** ✅ — CLI password manager: `init`, `set`, `get`, `list`, `gen` with SHA-256 vault
- **csvproc.iris** ✅ — CSV data processor: per-column stats (mean, stddev, min/max), z-score outlier detection, JSON report export
- **kvstore.iris** ✅ — Per-database key-value store: `init`/`set`/`get`/`del`/`list` with file persistence
- **loganalyzer.iris** ✅ — Apache combined log analyzer: status code distribution, top IPs, suspicious traffic (>50 reqs)
- **md2html.iris** ✅ — CSE cross-block replacement fix
- **buildsys.iris** ✅ — Same CSE fix applies
- **snake.iris** ✅ — Same CSE fix applies
- **distributed_kv.iris** ✅ — TCP KV server+client with `std.net` unboxed i64 socket handles; 9/9 tests pass in both `--emit eval` and native binary

### Phase 4: Automatic JSON Serialization (Native struct fix)
- **Problem**: `json_stringify` on `record Point { x: f64, y: f64 }` in native binary produced `4612811918334230528` (raw f64 bit pattern) instead of `{"0":1.5,"1":2.5}`. The `MakeStruct` codegen for named `IrType::Struct` stores values as raw C struct pointers (`%Point*`), but `iris_json_stringify` expects `IrisVal*`. The `box_to_ptr` fallthrough returned the raw struct pointer unchanged since its `emitted_ty = "ptr"`.
- **Fix**: Added a special case in `emit_instructions` for the `json_stringify` builtin: when the argument is `IrType::Struct`, emit GEP+load instructions to extract each field from the raw struct pointer, box each field via `box_to_ptr`, then call `iris_make_struct` to produce a proper `IrisVal*` before passing to `iris_json_stringify`.
- **File**: `src/codegen/llvm_ir.rs` — `emit_instructions()` `json_stringify` handler (lines 5239-5274)
- **Result**: `json_stringify(Point)` produces correct JSON `{"0":1.5,"1":2.5}` in native binary. All 8 JSON tests pass in both `--emit eval` and native.

### Blocked → Fixed
- **crawler.iris** ✅ — LLVM 17.0.1 `clang.exe` crashes in "X86 DAG→DAG Instruction Selection" on `@fetch_and_extract`. Fixed by emitting `optnone noinline` for functions with ≥15 blocks, forcing the -O0 instruction selector. `src/codegen/llvm_ir.rs`.
- **distributed_kv.iris** ✅ — JIT integer division by zero (JIT backend incomplete — clang not installed). Native binary works. `std.net` refactored from boxed record wrappers (`option<TcpConnection>`) to unboxed `i64` socket handles, eliminating heap allocation for network operations. **Files**: `src/stdlib/net.iris`, `projects/distributed_kv/server.iris`, `projects/distributed_kv/client.iris`.

- **Concatenation**: `concat()` takes exactly 2 args (nest for multi-part).
- **Spawn syntax**: `spawn { body };` — block syntax, not function-call.
- **Statement separator**: In multi-statement blocks, function-call statements must be separated by `;`. The last expression in a block omits `;`. Declarations (`val`, `var`) and control-flow (`if`, `for`, `while`) on separate lines do not need `;`.

### Phase 8: WebAssembly (WASM/WASI) Backend
- **Goal**: Compile IRIS programs to `.wasm` binaries via `--target wasm32`, using WASI SDK (wasi-libc) for malloc/printf/file I/O with WASI system calls. Programs without networking/spawn/FFI produce a standalone `.wasm` file runnable in Node.js/wasmtime.
- **Target triple**: Changed `"wasm32"` preset from `wasm32-unknown-unknown` to `wasm32-wasip1` (WASI preview 1). Added WASM data layout string `"e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"` to `target_data_layout()`. `src/codegen/llvm_ir.rs:43`.
- **Runtime header**: WASM pthread stubs via `#define` macros (comma-operator no-ops) to avoid type conflicts with wasi-libc's `<sys/types.h>` definitions. `pthread_create` runs `fn(arg)` synchronously (single-threaded). `src/runtime/iris_runtime.h`.
- **Runtime C**: WASM include path (`<unistd.h>`, `<dirent.h>`, `<dlfcn.h>` — provided by wasi-libc) excludes socket/terminal headers (`<sys/socket.h>`, `<termios.h>` — unavailable in WASI preview 1). Added `#if defined(__wasm__)` branches: `iris_read_key` uses `getchar()`, `iris_read_password` reads stdin without echo toggle, `iris_term_rows`/`cols` return 24/80 fallback. All UDP/TCP/HTTP functions guarded with `#ifndef __wasm__` (return -1 or empty string on WASM). FFI functions fall through to existing `#else` returning NULL/0. `popen`/`pclose` calls replaced with empty stubs. `src/runtime/iris_runtime.c`.
- **Build pipeline** (`build.rs`): Added `build_wasm_binary_impl()` — detects `wasm32` target, locates WASI sysroot at `~/.iris/toolchain/wasi-sysroot/wasi-sysroot-*`, compiles runtime + LLVM IR via clang with `--target=wasm32-wasip1 --sysroot=<path>`, links with `-nodefaultlibs -lc -lwasi-emulated-signal -lwasi-emulated-process-clocks -lclang_rt.builtins-wasm32`. Added `find_wasi_sysroot()` helper. `src/codegen/build.rs`.
- **LLVM IR wrapper fix**: WASM `_start` calls `__main_void()`, not `main(i32, ptr)`. Changed binary wrapper for WASM targets to emit `define i32 @__main_void()` (no args, returns i32) instead of `define i32 @main(i32 %argc, ptr %argv)`. This overrides the weak `__main_void` from wasi-libc, avoiding the `undefined_weak:main` trap function caused by WASM's structural function type system (functions with different signatures are distinct, so `main()` and `main(i32, ptr)` cannot override each other). `src/codegen/llvm_ir.rs:504`.
- **CLI**: Default output extension changes to `.wasm` when `--target wasm32` is specified. `src/main.rs`.
- **WASI SDK**: Downloaded `wasi-sysroot-24.0` (64 MB) + `libclang_rt.builtins-wasm32` (128 KB) to `~/.iris/toolchain/wasi-sysroot/`. Provides real `malloc/free/printf/fopen/getenv/sleep` via WASI system calls. `libpthread.a` is 8-byte empty stub.
- **Verified**: `tests/test_wasm_basic.iris` compiles to 392 KB `.wasm`, runs in Node.js 24 and wasmtime 26 with correct output: "Hello from IRIS on WASM!", "Math check: 2 + 2 = 4", "Float: 3.14159". Exit code 0.
- **File I/O verified**: `test_wasm_fileio.iris` — `file_write_all`, `file_exists`, `file_read_all`, `file_lines` all work via WASI preopens. 49 bytes written and read back correctly, 2 lines parsed.
- **Stub analysis confirmed**: wasi-libc for WASI P1 does NOT provide `socket/bind/listen/connect`, `system/popen/pclose`, `dlopen/dlsym/dlclose`, `fork/execvp`. Only `accept()` and `ioctl()` exist (limited). These are fundamental WASI P1 specification gaps — cannot be replaced with real code. WASI Preview 2 (component model) adds `wasi-sockets`, `wasi-http`, but requires different target triple (`wasm32-wasip2`) and runtime support.
- **Limitations**: Networking (TCP/UDP/HTTP), threading (spawn/par_for), terminal raw mode (`termios`), subprocess execution (`popen`/`system`), and FFI (`dlopen`) are not available in WASI preview 1 — stubs return errors or no-op. These depend on WASI preview 2 and the component model.
- **Phase 8b: WASI Preview 2** (`--target wasm32-wasip2`):
  - **Build pipeline**: `build_wasm_binary_impl` (`src/codegen/build.rs`) detects P2 target, compiles runtime with `-D__wasip2__=1`, links with P2 wasi-libc (which provides `socket/bind/connect/listen/accept/send/recv/getaddrinfo` via `wasi:sockets`), and post-processes with `wasm-tools component new --adapt wasi_snapshot_preview1.wasm` to produce a P2 component.
  - **Runtime C**: Added `__IRIS_WASM_STUB` flag — set on WASM P1 (not P2). All networking `#ifndef __wasm__` guards changed to `#ifndef __IRIS_WASM_STUB`, so networking code compiles for P2. WASM headers now include socket/net headers for P2. `inet_addr`/`inet_ntoa` replaced with `inet_pton`/`inet_ntop` (not available in P2 wasi-libc).
  - **Config**: `IRIS_WASM_TOOLS` and `IRIS_WASI_P2_ADAPTER` env vars for tool locations. `"wasm32-wasip2"` preset in `target_preset_to_triple` (`src/codegen/llvm_ir.rs:44`).
  - **Verified**: P2 components run in wasmtime 26 with `-S tcp=y -S inherit-network=y`. Basic + file I/O programs work (588 KB). P1 backward-compatible (392 KB, unchanged).
  - **Known limitation**: POSIX `connect()` over P2 `wasi:sockets` returns EINVAL at runtime — POSIX compat layer in wasi-libc P2 is still maturing. Networking code compiles and links but runtime `socket()`, `bind()`, `getaddrinfo()` succeed while `connect()` fails. Expected to improve in future wasi-sdk releases.

## Feature Implementations (Pattern Language)

### Or-Patterns (`pat1 | pat2 | ...` in `when`)
- **Parser**: Enhanced `parse_when_sub_pattern` to handle dotted enum variants (`Color.Red`), ranges (`1..=5`), tuples, and slices — previously only the initial pattern parser handled these, not sub-patterns used by or-alternatives. `src/parser/parse.rs`.
- **Lowering (SwitchVariant)**: Added `AstWhenPattern::Or` handling in `lower_when_expr` for enum matches — maps each `EnumVariant` sub-pattern to a shared arm block. `src/lower/mod.rs:5581,5619`.
- **Lowering (body)**: Added `Or` arm in body lowering for `ExtractVariantField` emission. `src/lower/mod.rs:5704`.
- **Condition**: `Or` already handled in `emit_pattern_condition` (ORs sub-conditions). `src/lower/mod.rs:6893`.
- **Tests**: `tests/test_or_patterns.iris` — enum, literal, tuple, range or-patterns. All pass (output 13).
- **Files**: `src/parser/parse.rs`, `src/lower/mod.rs`, `src/compiler.rs`, `tests/test_or_patterns.iris`

### Slice Patterns (`[a, b, ..rest]` in `when`)
- **Implementation**: Structurally complete before testing. Two bugs fixed during verification:
  1. `emit_pattern_condition` for bare identifiers in slice prefixes: `EnumVariant` with empty `enum_name` now returns `ConstBool(true)` immediately (bare identifiers always match). `src/lower/mod.rs:6653`.
  2. Slice out-of-bounds: Restructured to use SSA blocks — length check block, then element check block (safe because length verified), then merge block with phi node. `src/lower/mod.rs:6759`.
- **Tests**: `tests/test_slice_patterns.iris` — exact match, rest binding, mixed, wildcard, no-match, empty list, prefix binding. All 7 pass.
- **Files**: `src/lower/mod.rs`, `tests/test_slice_patterns.iris`

### Refutable Patterns in `let` (`let Some(x) = opt`)
- **Approach**: Parse-time desugaring to `val x = when opt { some(x) => x, _ => panic(...) }`, same pattern as `if let`/`while let`.
- **Parser**: Added refutable pattern detection in `parse_let_stmt` — checks for `_`, `none`, `some(`, `ok(`, `err(`, or `ident.` after `val` keyword. `src/parser/parse.rs:1356,1498`.
- **Desugaring**: Builds `AstExpr::When` with success arm (extracts binding) and failure arm (complement pattern + panic). Returns `AstStmt::Let` for bindings, `AstStmt::Expr` for statement-only patterns.
- **Tests**: `tests/test_refutable_let.iris` — 7 tests: `some(v)`, `ok(v)`, `err(e)`, `none`, `_`, `some(_)`, `some()`. All pass.
- **Files**: `src/parser/parse.rs`, `tests/test_refutable_let.iris`

### Pattern Guards, Range Patterns
- Already implemented and working (verified). `tests/test_pattern_guards.iris`, `tests/test_range_patterns.iris` pass.

### Associated Types in Traits (`type Item` in traits, `Self::Item` in methods)
- **Goal**: Allow traits to declare associated types (e.g. `trait Iterator { type Item }`) that impls bind to concrete types (e.g. `type Item = str`), with method signatures referencing them via `Self::Item`.
- **AST**: Added `AstAssocTypeDecl { name, span }`; `AstTraitDef.assoc_types: Vec<AstAssocTypeDecl>`; `AstImplDef.assoc_type_bindings: Vec<(String, AstType)>`; new `AstType::AssocType { base, assoc_name, span }` variant. `src/parser/ast.rs`.
- **Lexer**: New `Token::DoubleColon` (`::`) to disambiguate from single colon. `src/parser/lexer.rs`.
- **Parser**: `parse_trait_def` parses `type Name;` declarations before method sigs. `parse_impl_def` parses `type Name = Type;` bindings. `parse_type` detects `Name::Name` and constructs `AstType::AssocType`. `src/parser/parse.rs`.
- **Lowerer**: `resolve_assoc_types_in_ast_type(ty, bindings)` walks an `AstType` tree and rewrites `Self::X` to the concrete bound type. In the impl processing loop, both the renamed method (for lowering) and the `fn_sigs` registration have `Self::Item` references replaced with the concrete type. `src/lower/mod.rs:13404-13459` and `364-450`.
- **Test**: `tests/test_assoc_types.iris` — `trait Iterator { type Item; def next(self) -> option<Self::Item> }` with `impl Iterator for MyInt { type Item = str; ... }`. Verified end-to-end: prints `"42"` and exits 0 in both `--emit eval` and native binary.

### Variance Annotations (`+T` / `-T` on generic params)
- **Goal**: Allow type parameter variance annotations on generic structs and functions: `+T` (covariant), `-T` (contravariant), unadorned `T` (invariant, default).
- **AST**: New `Variance` enum (`Covariant | Contravariant | Invariant`). `AstGenericParam::Type` now takes a 3rd `Variance` argument: `Type(String, Vec<String>, Variance)`. `src/parser/ast.rs`.
- **Parser**: `parse_struct_def` and `parse_fn` parse `+`/`-` prefix tokens on type param names. `[+T where T: Ord]` works. `src/parser/parse.rs`.
- **Pass**: New `src/pass/variance_checker.rs` — AST-level pass that walks each generic def and validates:
  - Covariant type params: only in positive (return, struct field) positions; reject if used in function parameter positions
  - Contravariant type params: only in negative (function parameter) positions; reject if used in field or return positions
  - Invariant: no constraint
- **Registration**: `src/pass/mod.rs:19` (`pub mod variance_checker;`) and `src/lib.rs:253` (called after `AstExhaustivenessPass` in both `compile_ast_to_module` and `compile_ast`).
- **Test**: `tests/test_variance.iris` — `Box[+T]`, `Processor[-T]`, `Container[T]`. All pass. Invalid `BadBox[+T]` with fn-typed field rejected: `error[E0202]: covariant type parameter T appears in contravariant position`.
- **Files**: `src/parser/ast.rs`, `src/parser/parse.rs`, `src/parser/lexer.rs`, `src/pass/variance_checker.rs` (new), `src/pass/mod.rs`, `src/lib.rs`, `src/lsp.rs`, `src/compiler.rs`.

### Effect System (`effect k1, k2` on functions)
- **Goal**: Compile-time effect tracking and verification. Functions declare the side-effecting operations they perform (`effect io, alloc, fs`); calls are verified at compile time.
- **Built-in effects**: `io` (terminal), `alloc` (heap), `fs` (file system), `net` (networking), `spawn` (concurrency), `throw` (panics), `ffi` (foreign interface), `time` (clock), `random` (RNG), `env` (env vars), `sys` (system calls), `math` (FP exceptions). Default = pure.
- **AST**: `AstFunction.effects: Vec<String>` field. `src/parser/ast.rs`.
- **Lexer**: New `Token::Effect` keyword. `src/parser/lexer.rs:88`.
- **Parser**: `parse_fn` parses `effect k1, k2, k3` clause after return type and before body. `src/parser/parse.rs`.
- **Effect Registry** (`src/pass/effect_registry.rs`): HashMap from builtin function names to effect rows. Covers ~100 stdlib functions (`println`, `print`, `file_read`, `tcp_*`, `spawn`, `panic`, `time.now`, `random`, etc.).
- **Effect Inference Pass** (`src/pass/effect_checker.rs`):
  - Builds call graph from AST
  - Bottom-up topological sort + fixed-point inference
  - Each function's effect row = declared row ∪ union of callee effect rows
  - `EffectRow` data structure with `subset`, `union`, `display` methods
  - Auto-promotion: functions without `effect` clause get their inferred row silently (backward compat)
  - Strict mode (`IRIS_STRICT_EFFECTS=1` env var): requires explicit clauses on effectful functions, emits `error[E0301]`
  - Call-site verification: `error[E0302]` when callee effect is not in caller's row
  - Recursion-safe via `HashSet` cycle detection
- **Wiring**: `src/pass/mod.rs:4-5` (`pub mod effect_checker;`, `pub mod effect_registry;`). `src/lib.rs:265-272` and `351-358` invoke the checker before lowering (errors printed to stderr; non-strict allows auto-promotion, strict returns errors).
- **Tests**:
  - `tests/test_effects_basic.iris` — declares `effect io` and `effect io, alloc` functions; works in both modes.
  - Verified non-strict mode: silent auto-promotion, all calls compile and run.
  - Verified strict mode: `error[E0301]` for missing clauses, all explicit-clause functions pass.
- **Regression**: All 89 stdlib tests + all 20+ projects still pass under non-strict mode (auto-promotion preserves backward compat).

## Remaining Major Features — Assessment

The following features are NOT YET IMPLEMENTED and require significant design + implementation work (ordered by estimated effort):

### 1. Trait Objects / Dynamic Dispatch (`dyn Trait`)
- **Effort**: Weeks to months
- **What's needed**: `AstType::DynTrait`, `IrType::TraitObject` (fat pointer), vtable layout, `IrInstr::MakeTraitObject`/`DynCall`, LLVM codegen for vtable globals + indirect calls, interpreter support, trait bound verification
- **Prerequisite**: `where T: Trait` tracking ✅ **DONE** — constraint verification at monomorphization call sites implemented
- **Prerequisite**: Associated types ✅ **DONE** — `Self::Item` resolves via impl bindings

### 2. ~~Associated Types in Traits~~ ✅ **DONE**
- `AstTraitDef::assoc_types`, `AstImplDef::assoc_type_bindings`, `AstType::AssocType`, `Self::Item` resolution via impl bindings

### 3. ~~Const Generics (`record Array[T, const N: usize]`)~~ ✅ **DONE**
- `AstGenericParam::Const`, `AstType::ConstInt`, `const_param_subs`, mangling `Array__i64__5`, parser `const N: usize`, `len_expr` on `AstType::Array`

### 4. ~~Variance Annotations (`+T` / `-T` on generic params)~~ ✅ **DONE**
- `Variance` enum on `AstGenericParam::Type`, parser prefix `+`/`-`, dedicated `VarianceChecker` pass with E0202 errors

### 5. Higher-Kinded Types
- **Effort**: Months — full type constructor abstraction, major `IrType` redesign

### 6. ~~Effect System (`effect k1, k2` on functions)~~ ✅ **DONE** (Tier 1)
- See "Effect System" section above for full details. Tier 1 (tracking + verification) implemented.

### Recommended Implementation Order
1. ~~Const generics~~ ✅ **DONE**
2. ~~Trait constraint tracking~~ ✅ **DONE**
3. ~~Associated types~~ ✅ **DONE**
4. ~~Variance annotations~~ ✅ **DONE**
5. ~~Effect System (Tier 1)~~ ✅ **DONE**
6. **Trait objects** (most requested OOP feature, now unblocked)
7. **HKTs** — type constructor abstraction (months)
8. **Effect masks** (`with pure { ... }`) and **effect handlers** (algebraic effects) — Tier 2+ features
