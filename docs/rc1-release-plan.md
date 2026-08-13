# v1.0.0-rc1 Release Plan

**Date**: 2026-08-05
**Basis**: findings from [AUDIT.md](../AUDIT.md) and [known-issues.md](known-issues.md)

The version is already stamped `1.0.0-rc1`. The gap is not features — the language
surface is large and mostly works. The gap is **verified correctness**, and one
systemic cause underlies almost every defect found: the `.iris` test suite does
not check its own results.

Phases are ordered by dependency. Phase 0 must complete first because nothing
else can be trusted without it.

---

## Phase 0 — Establish ground truth

**Nothing in this plan is meaningful until the test suite runs.**

| # | Task | Why |
|---|---|---|
| 0.1 | `cargo test` must compile | A test-only `IrFunction` literal in `src/pass/validate.rs` was missing the `is_const` field, breaking the whole test build. A one-line fix was applied but **never verified** — `cargo check` does not build test targets. Run `cargo check --all-targets` first (faster), then `cargo test`. |
| 0.2 | Record a real pass/fail baseline | 1,767 Rust tests have never been observed passing in this session. The count could be 0 failures or 50. Every later claim depends on this number. |
| 0.3 | `cargo clippy --all-targets` + `cargo fmt --check` | Both are CI gates that have been dark for as long as the test build was broken. |

**Exit criteria:** a known, written-down pass/fail count for `cargo test`, and a
clean clippy/fmt run.

---

## Phase 1 — Correctness blockers

These produce **wrong answers**, not crashes, which is the worst failure class.

| # | Task | Severity |
|---|---|---|
| 1.1 | **Fix named arguments.** `f(a=1, b=2)` silently evaluates to 0; mixed `f(1, b=2)` drops the named argument. Positional calls are correct, so the defect is in named-argument binding during lowering. | Critical |
| 1.2 | **Convert the 104 print-only `.iris` tests to assert.** They pass whenever the program compiles and exits 0. This is the root cause of 1.1 reaching an RC. Mechanical work; expect it to surface further defects of the same class. | Critical |
| 1.3 | **Fix `str` field in a record inside `result<T, E>.`** Mis-types as `i64`, fails LLVM verification. Blocks `projects/autonomous_regulator/` from running. | High |
| 1.4 | **Improve the `when`-arm assignment diagnostic.** Assigning an enclosing `var` from `when` arms reports an internal SSA value (`variable '%30' is used before it has been assigned`) instead of pointing at the assignment. A workaround exists; the diagnostic is the defect. | Medium |

**Do 1.2 before declaring 1.1 fixed** — otherwise the same blind spot that hid it
will hide the next one.

**Exit criteria:** every `.iris` test either calls `assert(...)` or returns
non-zero on mismatch; named arguments produce correct values; `autonomous_regulator`
runs end to end.

---

## Phase 2 — Toolchain and clang independence

Substantial groundwork already landed: ML backends now load via `dlopen`, so
`iris_runtime.o` is SDK-independent, and the prebuilt-object infrastructure is
written. What remains is finishing and populating it.

| # | Task | Notes |
|---|---|---|
| 2.1 | **Fix the linker fallback.** Every single build currently prints `ld.lld link failed … falling back to clang` because `ld.lld` is skipped for MinGW targets. This is a per-build cost and a hard clang dependency. Either retarget Windows to `windows-msvc` + `lld-link`, or bundle `lld` in `toolchain/llvm/bin/` (`find_lld()` already searches there). | High — visible on every run |
| 2.2 | **Generate the prebuilt runtime objects.** `IRIS_GENERATE_PREBUILT=1 cargo build` populates `src/runtime/prebuilt/<triple>/`. Add a CI matrix job so releases ship them for every supported target. | High |
| 2.3 | **Prefer `llvm_c_api` over clang for `.ll` → `.o`.** The clang-free path is already written and used only as a fallback. Mind the documented LLVM/TensorFlow in-process conflict — gate on ML backends being active, or isolate in a child process. | Medium |
| 2.4 | **Wire a CLI command for `build_torch_plugin`.** The function exists; nothing calls it, so the LibTorch path is currently unreachable for users. | Low |

**Exit criteria:** `iris build` completes on a machine with no clang installed, and
prints no linker fallback warning.

---

## Phase 3 — Make the claims true

Cheap, and each one is a diligence risk if left.

| # | Task |
|---|---|
| 3.1 | **Correct the TCO claim.** `tests/test_tco.iris` is interpreter-only; there is no `musttail` in codegen. Native deep recursion can still overflow. |
| 3.2 | **Make `cuda.rs` fail loudly.** Unhandled tensor ops emit `@iris_tensor_op()`, which returns NULL at runtime. Convert to a hard codegen error — a silent NULL is worse than an unsupported-feature message. |
| 3.3 | **Delete dead runtime symbols.** `iris_dyn_call` and `iris_make_trait_object` have zero emission sites (LLVM emits vtables inline). `iris_call_closure`/`_void` are now unreferenced after the `native-llvm` removal. |
| 3.4 | **Verify the async claim.** `tests/test_async_runtime.iris` exists but is one of the 104 non-asserting tests. `await` is believed to desugar to a channel receive with no state machine — confirm before README or proposal text describes it as an async runtime. |

---

## Phase 4 — Repository hygiene

**This matters more than it sounds.** `HEAD` does not contain the `pub mod`
declarations for six modules that exist only as untracked files — `borrow_checker`,
`docs`, `formatter`, `package_manager`, `preprocessor`, `llvm_c_api`. A clone of
master today is a materially older compiler than the working tree.

| # | Task |
|---|---|
| 4.1 | Commit the six untracked modules plus the ~45 modified tracked files. The working tree *is* the release candidate; git does not know it yet. |
| 4.2 | Delete the ~20 stray debug dumps at the repo root (`dump_err.txt`, `stdout2.txt`, `nf_llvm.txt`, …) and extend `.gitignore`. |
| 4.3 | Tag only after Phases 0–3 pass. |

---

## Phase 5 — Examples and documentation

| # | Task |
|---|---|
| 5.1 | Extend `examples/` to the uncovered surface: macros, algebraic effects, `dyn Trait`, HKT, const generics, task groups, `defer`, `move`/borrow, `mod` blocks. **Every example must run and assert** — the same discipline Phase 1.2 imposes on tests. |
| 5.2 | Add ~12 multi-feature *interaction* programs to `tests/`. The current suite is one-feature-per-file, which is why new combinations break. Highest-leverage structural change in the repo. |
| 5.3 | Finish `projects/autonomous_regulator/` once 1.3 lands — it is currently a bug reproduction, not a demo. |
| 5.4 | Fill the thin stdlib spots: `std.collections` (4 functions) and `std.time` (no `DateTime` or date parsing). |

---

## Suggested sequencing

```
Phase 0  ──►  Phase 1  ──►  Phase 4.1  ──►  tag rc1
              (1.2 first)
Phase 2  ──►  runs in parallel; gates the "no external toolchain" claim
Phase 3  ──►  cheap, do alongside Phase 1
Phase 5  ──►  after Phase 1; 5.2 is the durable fix
```

Phases 0, 1 and 4.1 are the minimum for an honest rc1 tag. Phase 2 is what
supports the clang-independence story. Phase 5.2 is what stops this recurring.

---

## Release gate

Do not tag until all of the following hold:

- [ ] `cargo test` compiles and its pass/fail count is known and acceptable
- [ ] `cargo clippy --all-targets` and `cargo fmt --check` pass
- [ ] Every `.iris` test asserts its results
- [ ] Named arguments produce correct values
- [ ] `projects/autonomous_regulator/` runs end to end
- [ ] `iris build` works with no clang installed, with no fallback warning
- [ ] Every module in the working tree is committed
- [ ] No documented capability overstates what the code does (TCO, async, HKT)

---

## For the funding conversation

The honest framing is defensible and does not require any of this to be hidden:

> A systems language with a large, working feature surface — traits with
> associated types and blanket impls, const generics, variance, algebraic
> effects, borrow checking, 133 IR instructions, 23 optimisation passes, 42
> stdlib modules, and native/CUDA/WASM backends — with a differentiated
> AIS/ML library. Current engineering focus is hardening: converting a
> print-based test suite to an assertion-based one and closing the
> feature-interaction gaps that surfaced.

That is a credible position with a concrete plan. What would not survive scrutiny
is a demo that breaks when a reviewer types their own program — which, today, it
can.
