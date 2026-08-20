# IRIS — project instructions

IRIS is a compiled, statically-typed systems language for Autonomous Intelligent
Systems. Single Rust crate (~112k lines) at this repo root.

## Skills — invoke these, they carry the project's hard-won rules

| Skill | Use when |
|---|---|
| `iris-verify` | Before claiming any feature works/is missing; before grading anything |
| `iris-write-code` | Before creating or editing any `.iris` file |
| `iris-compiler-change` | Before editing `src/ir`, `src/lower`, `src/pass`, `src/codegen`, `src/parser` |
| `iris-claims` | Before writing docs, audits, README text, or proposal language |

## The three rules that matter most

1. **Run it before claiming it.** Grepping `src/` is weak evidence — it produced
   five false "missing feature" gradings. `tests/` is the index of truth.
   Stronger still: **write a program that asserts.** Two example suites added on
   2026-08-14 (1,405 lines) found **eleven defects, about one per 130 lines** —
   none of which was visible from reading the compiler, and three of which were
   hiding behind another bug.
2. **`cargo check` does not build test targets.** Use `cargo check --all-targets`
   when the test build matters. This gap hid a compile error for a whole session.
3. **Every `.iris` test and example must `assert`.** 122 of 125 still do not,
   which is how a feature that silently returns `0` reached a release candidate.
   Nothing globs `tests/*.iris`, so the corpus is never executed by `cargo test`.
4. **Never edit `src/` while a `cargo test` run is in its build phase** — the run
   silently mixes the changes and its result attributes them to the wrong commit.
   Wait for the build to finish (unit tests appearing in the log) or for the run
   to complete.
5. **A stdlib usage survey must read `tests/*.rs`, not just `.iris` files.**
   Eight modules were deleted on 2026-08-15 as having zero dependents; seven had
   asserting Rust tests and had to be restored. The Rust suite exercises the
   stdlib directly, so an `.iris`-only survey sees a fraction of the real
   dependency graph.
6. **Run a suspicious program several times before believing either result.**
   Compilation was non-deterministic until 2026-08-15: the same source produced
   three different IR outputs in six runs, three of them invalid, so a program
   passed or failed roughly half the time. It was filed as a *backend
   divergence* for exactly this reason. `--emit ir | md5sum` across five runs is
   now a cheap and meaningful check.
7. **Do not pipe program output through text tools when bytes matter.** `sed`,
   `grep` and `tail` normalise line endings on msys; redirect to a file and use
   `od -c`. A `\r` corruption was "confirmed" through `sed` before this was
   noticed, and the compiled binary had been correct all along.

## Current state (2026-08-14, branch `rc1-hardening`)

Version is stamped `1.0.0-rc1`. `cargo build` and `cargo check` are clean.

**Full `cargo test`: 1767 passed / 2 failed / 156 binaries** (was 1726 / 35 / 155
at `720f358`). A run takes ~60 min. Both remaining failures are known:
- `test_bring_file_private_not_visible` — `pub` is not enforced across module
  boundaries; needs per-item provenance, see known-issues #13
- `test_multimodal_ai_orchestrator_project` — environmental, needs a local
  LibTorch install (this is the one remaining `0xc0000005`)

**Both former ship-stoppers are fixed.** All six modules are declared and a fresh
clone builds; the hardcoded developer `PATH` is gone from `build.rs`. Do not
re-report these.

**Known broken — see `docs/known-issues.md`:**
- ✅ Named arguments `f(a=1, b=2)` — **fixed** 2026-08-15 (#1)
- 🟠 A record field typed by a *brought* module mangles as generic (#7)
- 🟠 `pub bring` re-exports functions but not `record`/`choice` types (#9)
- 🟠 `effect` clauses are rejected on trait method declarations (#10)
- 🟠 Assigning an enclosing `var` from `when` arms fails SSA construction
- ✅ `dyn Trait` — native vtables and coercion at bindings, call arguments and `list<dyn Trait>` (#18, #18b fixed 2026-08-16)
- 🟠 Tuples cannot be compared with `==` on either backend (#26)

**A type parameter used only in the return type needs an annotation** (#14):
`val s: Set<str> = set_new()`. Inference runs after lowering, so it cannot be
recovered from later use. Unconstrained now reports a clear error.

## Key documents

- `docs/portability-readiness.md` — master fix/rebuild/add/remove list
- `docs/rc1-release-plan.md` — phased plan with a release gate
- `AUDIT.md` — feature inventory with evidence grades
- `docs/architecture-vs-rustc.md` — why type-after-lowering causes the type bugs
- `docs/autonomy-stack-assessment.md` — AIS 7/10, ML 6/10, ROS2 2/10
- `docs/known-issues.md` — live defect list with reproductions
- `AGENTS.md` — historical session log (long; grep it, don't read it whole)

## Build commands

```bash
CARGO=/c/Users/Moon/.cargo/bin/cargo   # not on PATH by default
$CARGO check --all-targets             # ~3 min on this 2-core box
$CARGO build                           # ~6 min
$CARGO test --no-fail-fast             # ~60 min, 156 binaries
target/debug/iris.exe --emit eval file.iris
target/debug/iris.exe --strict-effects --emit eval file.iris   # effect gate
target/debug/iris.exe build file.iris  # real native binary
```

**`--emit eval` builds natively and falls back to the interpreter**; it is not
the interpreter path. `IRIS_FORCE_INTERP=1` forces the interpreter and bypasses
codegen, so never use it to validate a codegen change. The library's
`EmitKind::Eval` behaves the same way, which means the Rust suite *does*
exercise codegen.

Under `--strict-effects`, a function with no `effect` clause that compiles has
been proven to allocate nothing, do no I/O and call nothing external anywhere in
its reachable call graph. Violations fail the build.

**One exception remains.** Direct calls, method calls, extension methods,
`dyn Trait` dispatch and closures called where they are defined are all tracked
(#64, fixed 2026-08-20). Effects reached through a **function-valued parameter**
are not: `def hidden(f: |i64| -> i64) -> i64 { f(1) }` performs whatever `f`
performs and is certified pure, because `f` is a parameter and there is no name
to record. Closing it needs effect polymorphism on function-typed parameters.
See known-issues #65, and state this limit whenever the claim is made.

**The VS Code language server must not point at `target/debug/iris.exe`.** Every
build relinks it; on Windows a running server also holds it locked, so the two
fight. Point `iris.executablePath` at a stable copy (`~/.iris/bin/iris.exe`) and
refresh it after a build.

`ld.lld link failed … falling back to clang` on every build is a known expected
warning on this machine, not a failure.

## Governance

`.antigravity/orchestrator.md` requires an Implementation Plan and **human
approval before writing code** for substantial features. Honour it for anything
beyond a contained fix.

## Architecture in one line

`source → preprocessor → lexer → parser → AST → [AST passes] → Lowerer (also
monomorphises) → IrModule (block-parameter SSA, 133 instrs) → ~23 passes →
codegen (llvm_ir | cuda | simd | wasm | onnx)`

Type inference runs *after* lowering and defaults unresolved slots to `i64` —
this is the root cause of the silent type defects. See
`docs/architecture-vs-rustc.md` before attempting a type fix.
