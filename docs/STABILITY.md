# IRIS Stability Policy

> **Current version: 0.6.1** — IRIS is pre-1.0 and evolving rapidly.
> The current locked language/tooling surface is tracked in
> [`docs/current-language-lock.md`](docs/current-language-lock.md).

This document defines the stability guarantees, deprecation process, and
versioning discipline the project follows on the path to 1.0 and beyond.

---

## Versioning Scheme

IRIS follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

| Segment | When to bump |
| --------- | -------------- |
| **MAJOR** | Breaking changes to syntax, semantics, IR, or stdlib public API |
| **MINOR** | New features, new builtins, new passes — fully backward-compatible |
| **PATCH** | Bug fixes, performance improvements, documentation updates |

### Pre-1.0 Rules (current)

While the version is `0.x.y`:

- **MINOR** bumps (`0.3 → 0.4`) may include breaking changes but must be
  documented in `CHANGELOG.md` under a **Breaking** section.
- **PATCH** bumps (`0.6.1 → 0.5.1`) are always backward-compatible.
- Migration guides are provided for any breaking change.

### Post-1.0 Rules

Once IRIS reaches `1.0.0`:

- **All breaking changes require a MAJOR bump.**
- Deprecated features survive for at least **two MINOR releases** before removal.
- The compiler emits deprecation warnings with actionable migration advice.

---

## Stability Tiers

Every language feature and API surface is categorized into one of three tiers:

### Tier 1 — Stable

These features have been extensively tested, are covered by the specification,
and will not change in backward-incompatible ways without a major version bump.

- Core syntax: `def`, inferred `val`/`var`, `if/else`, `while`, `for`, `when`
- Scalar types: `i32`, `i64`, `f32`, `f64`, `bool`, `str`
- Records (`record`) and enums (`choice`)
- `option<T>` and `result<T,E>` with `?` operator
- Pattern matching (`when`)
- Standard builtins: `print`, `len`, `concat`, `to_str`, etc.
- CLI: `iris run`, `iris build`, `iris repl`
- ML built-ins: `tensor<T,[dims]>`, `einsum`, `grad<T>`, `sparse<T>`
- Model DSL (`model { ... }`)
- ONNX/CUDA/SIMD codegen targets
- DAP debugger protocol
- `atomic<T>`, `mutex<T>`

### Tier 2 — Provisional

These features are fully implemented and tested but may see refinement in
syntax or semantics before stabilization.

- Generics and traits (`trait`, `impl`)
- Closures and higher-order functions
- Concurrency primitives: `channel<T>`, `spawn`, `par for`, `async/await`
- Collections: `list<T>`, `map<K,V>`, deque, heap, queue, bitset
- FFI: C, Python, Rust
- Standard library modules (25 modules)
- Package manager (`iris pkg`)

### Tier 3 — Experimental

These features are available but subject to redesign or removal.

- Obscure/undocumented CLI flags
- Custom compiler passes API

---

## Deprecation Process

1. **Announce** — The feature is marked deprecated in `CHANGELOG.md` and the
   compiler emits a warning: `warning[D001]: 'old_name' is deprecated, use
   'new_name' instead (will be removed in 0.X)`.
2. **Grace period** — The deprecated feature continues to work for at least
   two minor releases (post-1.0) or one minor release (pre-1.0).
3. **Remove** — The feature is removed. The compiler emits a hard error with a
   migration hint pointing to the replacement.

---

## Roadmap to 1.0

The following milestones must be met before the `1.0.0` release:

| # | Milestone | Status |
| --- | --------- | -------- |
| 1 | All Tier 1 features pass fuzz testing (lexer, parser, lowerer, compiler) | Done |
| 2 | Unit test coverage for all `src/` modules (≥200 unit tests) | Done (249) |
| 3 | Language specification published (`SPEC.md`) | Done |
| 4 | Benchmark regression CI with automated alerting | Done |
| 5 | Stable C runtime (O(n log n) sort, no undefined behavior) | Done |
| 6 | At least 1000 integration tests across all phases | Done (1135) |
| 7 | Cross-platform CI: x86_64 + ARM64 on Linux, Windows, macOS | Done |
| 8 | VS Code extension published on marketplace | Done |
| 9 | Package registry for third-party packages | Done |
| 10 | Security audit of FFI and filesystem operations | Done |
| 11 | Complete CHANGELOG.md from 0.1.0 to current | Done |
| 12 | At least 3 external contributors or reviewers | Done |

---

## Compatibility Promise (Post-1.0)

Once IRIS reaches 1.0, the following guarantee applies:

> **Any valid IRIS program that compiles and runs correctly under version X.Y.Z
> will continue to compile and produce identical output under any version X.*.* 
> (same major version), unless it relies on features in Tier 3 (Experimental).**

This means:
- **Syntax** — No keywords removed or repurposed without a major bump.
- **Semantics** — No silent changes to runtime behavior.
- **Builtins** — No builtins removed; signatures are frozen.
- **CLI** — `iris run` and `iris build` flags are stable.
- **IR format** — The textual IR format is informational, not stable. The binary
  IR cache format may change between minor versions (caches are invalidated
  automatically).

---

## Reporting Breakage

If you believe a release introduced an unintentional breaking change:

1. Open a GitHub issue with the tag `breaking-change`.
2. Include the IRIS version, platform, minimal reproducing `.iris` file, and
   expected vs. actual behavior.
3. The maintainers will triage within 48 hours and issue a patch release if the
   breakage is confirmed unintentional.

---

*This policy is itself versioned alongside the project. Last updated: 2026-03-13.*
