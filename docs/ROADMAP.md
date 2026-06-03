# IRIS Roadmap

> **Current version: 0.6.1** — active release. Pre-1.0, evolving rapidly.

This document outlines the planned milestones for IRIS from the current release
line through the stable 1.0 release and beyond.

---

## Current State (v0.6.1)

- Core language, interpreter, LLVM native codegen — functional
- Closures (basic, captures, HOF) — working in native backend
- LSP, DAP debugger, REPL — functional
- 34 stdlib modules — registered and implemented (including std.ais, std.rl, std.ml, std.nn)
- **1,400+ tests**, fuzz corpus expanded
- VS Code extension v0.6.1 built (VSIX); published to marketplace
- Installer scripts for Windows, Linux, macOS
- Package manager: `iris pkg init/add/install/update/list/check/build/run` (local + git deps, lockfile)
- CHANGELOG.md, ROADMAP.md, SPEC.md, docs/ — all present and current
- Tensor/ONNX/CUDA/SIMD backends functional; security audit and profiler added

Note: Some items below are marked as completed on `main` but are not part of
the latest tagged release yet.

---

## v0.3.0 — Hardening & Polish

**Goal:** Ship a reliable, well-documented 0.x release that outsiders can use.

| # | Task | Priority | Status |
| --- | ------ | ---------- | -------- |
| 1 | Create `CHANGELOG.md` from 0.1.0 → 0.2.0 → 0.3.0 | High | ✅ Done |
| 2 | Add ~130 unit tests to hit the ≥200 target | High | ✅ Done (249) |
| 3 | Add ~150 integration tests to hit the ≥1000 target | High | ✅ Done (1135) |
| 4 | Publish VS Code extension to marketplace | High | ✅ Done |
| 5 | Clean up stale files (`phase21_TEMP.rs`, register external stdlib) | Medium | ✅ Done |
| 6 | Finalize `SPEC.md` — remove "Draft", fill grammar gaps | Medium | ✅ Done |
| 7 | Fix concurrency in native backend (`spawn`/`channel` crash) | High | ✅ Done |
| 8 | Fuzz all Tier 1 features — expand corpus, run extended campaigns | Medium | ✅ Done |
| 9 | Implement TCP/network I/O — replace interpreter stubs | Medium | ✅ Done |
| 10 | Error message improvements — source spans, colored diagnostics | Low | ✅ Done |

---

## v0.4.0 — Ecosystem & Packaging

**Goal:** Make it easy to share and consume IRIS libraries.

| # | Task | Priority | Status |
| --- | ------ | ---------- | -------- |
| 1 | Package registry — central registry for `iris pkg add` | High | ✅ Done |
| 2 | Dependency resolution — semver solver, lockfile (`iris.lock`) | High | ✅ Done |
| 3 | `iris doc` — auto-generate HTML docs from comments | Medium | ✅ Done |
| 4 | `iris fmt` — standalone formatter (currently LSP-only) | Medium | ✅ Done |
| 5 | `iris lint` — standalone linter with BP001–BP006 rules | Medium | ✅ Done |
| 6 | Cross-compilation — `iris build --target aarch64-linux` | Medium | ✅ Done |
| 7 | Incremental compilation — use cache infrastructure for faster rebuilds | Low | ✅ Done |

---

## v0.6.1 — ML & Compute

**Goal:** Make the ML headline features real, not stubs.

Release status: Released (May 2026)

| # | Task | Priority | Status |
| --- | ------ | ---------- | -------- |
| 1 | Tensor runtime — replace shape-tracking stubs with real compute | Critical | ✅ Done |
| 2 | `einsum` codegen — generate loop nests or dispatch to BLAS | High | ✅ Done |
| 3 | ONNX export — binary protobuf, not just text format | High | ✅ Done |
| 4 | GPU backend — CUDA or Vulkan compute shaders for tensor ops | Medium | ✅ Done |
| 5 | SIMD codegen — auto-vectorize tight loops on x86/ARM | Medium | ✅ Done |
| 6 | Automatic differentiation v2 — reverse-mode AD | Medium | ✅ Done |
| 7 | Sparse tensor ops — CSR/COO kernels, sparse matmul | Low | ✅ Done |
| 8 | ML compute kernels — conv2d, maxpool, batchnorm, softmax, GELU | Critical | ✅ Done |
| 9 | Loss functions — MSE, cross-entropy, binary CE in C runtime | Critical | ✅ Done |
| 10 | Optimizers — SGD, Adam with state management | Critical | ✅ Done |
| 11 | BLAS dispatch — OpenBLAS/MKL via `-DIRIS_USE_BLAS` | High | ✅ Done |
| 12 | Blocked matmul — 32×32 tiled for cache friendliness | High | ✅ Done |
| 13 | AIS framework (`std.ais`) — agent loop, perception, decision | High | ✅ Done |
| 14 | RL module (`std.rl`) — Q-learning, SARSA, policy gradient, replay | High | ✅ Done |
| 15 | System requirements documentation (`REQUIREMENTS.md`) | Medium | ✅ Done |
| 16 | Nightly CI workflow — extended fuzz, smoke tests, benchmarks | Medium | ✅ Done |
| 17 | User IR optimization bump — `-O1` → `-O2` (LLVM 18) | Medium | ✅ Done |

---

## v0.6.1 — Performance & Security

**Goal:** Production-grade performance and trustworthy FFI.

Release status: Released (May 2026)

| # | Task | Priority | Status |
| --- | ------ | ---------- | -------- |
| 1 | Security audit — FFI surface, filesystem ops, network I/O | Critical | ✅ Done |
| 2 | GC / memory management — refcounting or tracing GC for native backend | High | ✅ Done |
| 3 | Optimization passes — DCE, constant folding, inlining, loop unrolling | High | ✅ Done |
| 4 | Benchmark suite expansion — real-world workloads | Medium | ✅ Done |
| 5 | Profiler — `iris profile` with flame graphs | Medium | ✅ Done |
| 6 | Sandboxed FFI — restrict filesystem/network for untrusted packages | Low | ✅ Done |

---

## v1.0.0 — Stable Release

**Goal:** Freeze Tier 1 features, commit to backward compatibility.

| # | Gate | Criteria |
| --- | ------ | ---------- |
| 1 | All 12 STABILITY.md milestones met | Including ≥3 external contributors |
| 2 | Zero known crashers in fuzz campaign | 72-hour clean run |
| 3 | SPEC.md finalized and versioned | No "Draft" label |
| 4 | CHANGELOG complete | Every breaking change since 0.1.0 |
| 5 | Tier 1 features frozen | Syntax, builtins, CLI flags locked |
| 6 | GC implemented | No memory leaks in long-running programs |
| 7 | Tensor ops functional | At least matmul, elementwise, reduce on CPU |
| 8 | ML readiness | Loss functions, optimizers, conv/pooling layers functional |
| 9 | AIS framework | Agent lifecycle, decision strategies, RL primitives tested |
| 10 | Published artifacts | VS Code extension, package registry, 3-platform installers |

---

## Post-1.0 — Future Directions

- **Language server v2** — semantic tokens, call hierarchy, type hierarchy
- **Debugger enhancements** — conditional breakpoints, hot-reload
- **Self-update CLI subsystem** — Built-in `iris upgrade` / `iris self-update` to fetch lightweight, fast compiler & stdlib binary delta updates on-demand
- **IDE plugins** — IntelliJ, Neovim, Emacs
- **Self-hosting** — IRIS compiler written in IRIS
- **WebAssembly backend** — `iris build --target wasm32`
- **Distributed compute** — multi-node tensor parallelism

---

*Last updated: 2026-05-29*
