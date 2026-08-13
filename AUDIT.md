# IRIS Language Expressiveness Audit

**Date**: 2026-08-05
**Version**: IRIS v1.0.0-rc1
**Supersedes**: the 2026-07-22 audit, which is substantially out of date — roughly
a dozen features it lists as *Missing / Critical* have since shipped.

Everything below is extracted from the codebase (lexer keyword table, AST enums,
`IrType`/`IrInstr` enums, pass directory, stdlib sources, `EmitKind`, CLI
commands), not from prior documentation.

**Confidence grades**: **Verified** = built and run in this session ·
**Tested** = dedicated test in-tree · **Present** = in compiler source, behaviour
not independently confirmed · **Absent** = no implementation found.

---

## 1. Scale at a glance

| Dimension | Count |
|---|---|
| Reserved keywords | 46 |
| AST expression forms | 35 |
| AST statement forms | 18 |
| AST type forms | 23 |
| IR types (`IrType`) | 22 |
| IR instructions (`IrInstr`) | **133** |
| Compiler passes | 23 modules |
| Builtin / intrinsic names | ~203 |
| Standard library modules | 42 (≈650 public functions) |
| Codegen backends (`EmitKind`) | 15 |
| CLI subcommands | 16 |
| Tests | 1,767 Rust `#[test]` · 153 Rust files · 125 `.iris` programs |

---

## 2. Syntax surface

**Keywords (46)** — `async await bool break bring by catch choice const continue
def defer defmacro dyn effect else extern false for if impl in let loop match mod
model move par pub raise record resume return select spawn str tensor to trait
true try type unsafe usize val var when while with yield`

**Expressions (35)** — `ArrayLit Await BinOp Block BoolLit Call Cast Deref
FieldAccess FloatLit Handle Ident If Index IntLit Lambda MacroCall MapLiteral Mask
MethodCall Move NullCoal Raise Ref RefMut Splat StringLit StructLit Try TryCatch
Tuple TupleIndex UnaryOp Unsafe When`

**Statements (18)** — `Assign Break Continue Defer Expr ForEach ForRange
HandleStmt Let LetTuple Loop MaskStmt ParFor Return Select Spawn While Yield`

Notable: `if`/`when`/`block`/`handle`/`with` are all *expressions*, so the language
is expression-oriented rather than statement-oriented.

---

## 3. Type system

**Type forms (23)** — `Array AssocType Atomic Chan ConstInt DynTrait Fn Generic
Grad List Map MaskEffectType Mutex Named Option Ref RefMut Result Scalar Sparse
Tensor Tuple WeakRef`

| Capability | Grade | Detail |
|---|---|---|
| Hindley-Milner inference | Present | Union-find, `HmTypeInferPass`; unresolved slots default to `i64` |
| Parametric generics | Tested | `def f[T](x: T) -> T`, monomorphised |
| Const generics | Tested | `const N: usize`, `AstType::ConstInt`, mangles to `Array__i64__5` |
| Trait bounds | Tested | `[T where T: Show, Ord]`, checked at monomorphisation |
| Associated types | Tested | `type Item;` in trait, `Self::Item` resolution |
| Variance annotations | Tested | `+T` / `-T` / invariant, dedicated `variance_checker` pass |
| Trait objects | Tested | `dyn Trait`, `IrType::TraitObject`, vtable dispatch |
| Blanket impls | Tested | `impl[T where T: Show] Trait for T` |
| Refinement types | Present | `type Positive = i64 where x > 0` — integer comparisons only |
| Reference types | Tested | `&T` / `&mut T`, erased at lowering (zero-cost) |
| Higher-kinded types | **Unverified** | `F[_]` parsed; README claims full HKT, old audit says partial — see §8 |

**Scalars**: `i8 u8 i32 u32 i64 u64 f32 f64 bool str char usize`

---

## 4. Memory, ownership, effects

| Capability | Grade | Detail |
|---|---|---|
| Reference counting | Present | `Retain`/`Release` IR, side-table RC in C runtime |
| Cycle collection | Present | Bacon-Rajan, `iris_gc_cycle_collect_locked` |
| Borrow checking | Tested | `src/pass/borrow_checker.rs`; rejects conflicting borrows |
| Move semantics | Tested | `move` keyword; use-after-move and borrow-after-move rejected |
| Weak references | Present | `weak_ref<T>`, `weak_new` / `weak_upgrade` |
| `defer` | Tested | Scope-exit cleanup |
| `unsafe` blocks | Present | `AstExpr::Unsafe` |
| Effect system | Present | `effect io, alloc, fs…`; inference + call-site verification; `IRIS_STRICT_EFFECTS=1` |
| Algebraic effects | Tested | `handle … with { … }`, `resume`, `PushHandler`/`PopHandler`, native dispatch |
| Effect masks | Tested | `with pure { … }` |
| Effect row polymorphism | Present | Effect variables, instantiation, subset checks |

---

## 5. Concurrency

`spawn` (OS threads) · `spawn(group)` with structured `task_group` join/cancel ·
`chan<T>` blocking channels with condvar · `select` · `par for` (thread pool) ·
`atomic<T>` · `mutex<T>` · `Barrier` · `async`/`await`.

Honest limits: `await` desugars to a channel receive — there is **no state-machine
async runtime**, no reactor, no waker. Threading is 1:1 OS threads; no green
threads or work stealing.

---

## 6. Pattern matching

Enum variants · option `some`/`none` · result `ok`/`err` · wildcard · integer,
bool and string literals · tuples · inclusive ranges · **or-patterns** ·
**slice patterns** with rest binding · **guards** · refutable `let` · compile-time
**exhaustiveness checking** (`ast_exhaustive` + `exhaustive` passes).

Absent: binding patterns (`x @ pat`), struct patterns, deep nesting.

---

## 7. Standard library — 42 modules

Sized by public function count:

- **ML / AIS (the domain focus)** — `ml`(87) `ais`(61) `tensor`(37) `adaptive`(36)
  `uncertainty`(22) `rl`(20) `nn`(18) `meta_learning`(10) `stochastic`(8)
- **Robotics / systems** — `ros2`(29) `ffi`(18) `net`(11) `os`(10) `fs`(10)
  `async`(13) `time`(7)
- **Data** — `set`(15) `iter`(12) `table`(9) `heap`(9) `deque`(9) `queue`(7)
  `dataframe`(7) `dataset`(7) `csv`(7) `bitset`(5) `collections`(4)
- **Services** — `sql`(12) `http`(12) `http_server`(9) `kv`(7) `json`(9)
- **Text / output** — `crypto`(12) `unicode`(11) `testing`(11) `fmt`(11)
  `string`(10) `log`(10) `svg`(6) `path`(5) `termplot`(1)

`std.ais` (61 functions across 14 subsystems — homeostatic regulation, active
inference, neuroevolution, EWC, decision strategies, MAPE-K lifecycle) is the
clearest expression of what differentiates IRIS.

Thin: `collections`(4) reads as an early sketch; `termplot`(1) is minimal.

---

## 8. Compilation and tooling

**Passes (23)**: `type_infer_hm validate type_infer const_fold strength_reduce
copy_prop licm inline loop_unroll exhaustive ast_exhaustive dce cse shape_check
shape_infer_graph gc_annotate borrow_checker variance_checker effect_checker
effect_registry lint dead_node graph_pass`

**Backends (15 `EmitKind`)**: `Ir Llvm LlvmComplete Binary Eval Jit Cuda CudaPtx
Simd Graph Onnx OnnxBinary TensorRt PgoInstrument PgoOptimize`. Targets include
native, WASM (WASI P1 and P2 component model), and NVPTX.

**Tooling (16 CLI commands)**: `build run repl lsp dap pkg bench profile test
explain upgrade install setup docs fmt`. LSP with context-aware completion,
go-to-definition and hover; DAP debugger; flame-graph profiler; formatter; doc
generator; package manager; test runner.

---

## 9. Genuinely missing

| Feature | Severity | Note |
|---|---|---|
| Native tail calls | Important | `tests/test_tco.iris` is **interpreter-only**; no `musttail` in codegen — native deep recursion can still overflow |
| `Clone`/`Copy` traits | Important | No copy-vs-move distinction; everything refcounted |
| `From<T>` conversions | Important | No implicit error conversion, weakening `?` ergonomics |
| Async runtime | Important | `await` is a channel receive, not a state machine |
| Green threads / M:N | Nice-to-have | OS threads only |
| Binding patterns (`x @ pat`) | Nice-to-have | Struct patterns *are* supported — see correction below |
| Date/time parsing | Important | No `DateTime` type |

> **Correction (2026-08-05).** An earlier revision of this document graded five
> features *Absent* on the strength of a grep over a narrow file set. Reviewing
> `tests/` disproved all five. They are implemented and tested:
>
> | Feature | Evidence | Syntax |
> |---|---|---|
> | Labelled `break`/`continue` | `tests/test_labeled_break.iris` | `loop myLoop { break myLoop; }`, `for outer i in 1..10 { break outer; }`, `while counter c { … }`, `continue outer2` |
> | Struct patterns | `tests/test_struct_pattern.iris` | `when p { Point { x: 1, y } => … }` |
> | Higher-kinded types | `tests/test_hkt.iris`, `test_hkt_syntax.iris` | `record Wrapper[F[_], A] { value: F<A> }`, used as `Wrapper<Box, i64>`; monomorphised |
> | Conditional compilation | `tests/test_conditional_compile.iris` | `#define` / `#ifdef` / `#ifndef` / `#else` / `#endif` preprocessor |
> | Extension methods | `tests/test_extension_methods.iris` | `21.double()` — any module-level fn whose first param is `T` |
>
> The HKT correction also resolves the contradiction flagged in §9: the README's
> claim is **supported**, and the 2026-07-22 audit's *Partial* grade was wrong.
>
> The methodological lesson is worth recording: grepping compiler source is weak
> evidence for absence, because a feature may be implemented under names the grep
> did not anticipate. `tests/` is the reliable index of what exists.

Platform limits not attributable to IRIS: WASI Preview 1 provides no sockets,
subprocesses or `dlopen`.

**One documented claim still needs correction before external use:**

- **TCO** — interpreter-only; there is no `musttail` in codegen, so it should not
  be described as tail-call optimisation without qualification.

*(The HKT discrepancy is resolved — see the correction above. `tests/test_hkt.iris`
confirms `F[_]` type-constructor parameters with recursive monomorphisation, so
the README's claim stands.)*

---

## 10. Assessment

By raw surface — 133 IR instructions, 23 passes, 42 stdlib modules, 15 backends,
traits with associated types and blanket impls, const generics, variance, effects,
borrow checking — IRIS is a substantially complete systems language, not a
prototype. The AIS and ML libraries are genuinely differentiated.

**The gap is interaction robustness, not breadth.** Coverage in `tests/` is broad
but structured one-feature-per-file; 15 of 23 surveyed modern features have zero
usage in `projects/`. The consequence is measurable: writing one new six-module
program ([`projects/autonomous_regulator/`](projects/autonomous_regulator/))
surfaced two compiler defects within ten minutes:

- **Defect 1** — assigning to an enclosing `var` from `when` arms fails SSA
  construction (`variable '%30' is used before it has been assigned`). Workaround
  exists; the diagnostic does not point at the cause.
- **Defect 2** *(open)* — a `str` field in a record inside `result<Record, str>`
  mis-types as `i64` and fails LLVM verification (`'%v14' defined with type 'i64'
  but expected 'ptr'`). Both ingredients are individually covered by 8+ test files;
  the combination is not.

Neither involves an exotic feature. Both are what a new user hits in their first
hour.

**Recommended order**: fix Defect 2 → improve Defect 1's diagnostic → add ~12
multi-feature interaction programs to `tests/` → re-verify the HKT and TCO claims.

The defensible framing is *broad working coverage with a known, addressable
hardening gap and a concrete plan to close it*. What would not survive scrutiny is
a demo that breaks when a reviewer types their own program — which, on today's
build, it can.
