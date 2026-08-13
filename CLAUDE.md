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
2. **`cargo check` does not build test targets.** Use `cargo check --all-targets`
   when the test build matters. This gap hid a compile error for a whole session.
3. **Every `.iris` test and example must `assert`.** 104 of 122 currently do not,
   which is how a feature that silently returns `0` reached a release candidate.

## Current state (2026-08-05)

Version is stamped `1.0.0-rc1`. `cargo build` and `cargo check` are clean.
`cargo test` has **never been observed running** — verifying it is task zero.

**Known broken — see `docs/known-issues.md`:**
- 🔴 Named arguments `f(a=1, b=2)` silently evaluate to `0`
- 🔴 `str` field in a record inside `result<T,E>` mis-types as `i64`
- 🟠 Assigning an enclosing `var` from `when` arms fails SSA construction

**Ship-stopper:** `HEAD` lacks `pub mod` declarations for six modules that exist
only as untracked files (`borrow_checker`, `docs`, `formatter`,
`package_manager`, `preprocessor`, `llvm_c_api`). **A fresh clone does not build
the current compiler.** Committing the working tree is the top priority.

Also: a developer's local `PATH` is hardcoded into `build.rs` (Cheat Engine,
Passixer, Apple Mobile Device Support). Must be removed before external release.

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
target/debug/iris.exe --emit eval file.iris
```

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
