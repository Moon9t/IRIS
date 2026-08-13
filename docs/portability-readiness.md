# IRIS — Master Readiness Document

**Date**: 2026-08-05
**Goal**: everything to **fix**, **rebuild**, **add**, and **remove** so IRIS can
be tested on machines other than the development box.

This consolidates findings from the expressiveness audit, the rustc architecture
comparison, the autonomy assessment, and defects found by writing and running new
IRIS programs. Nothing here is aspirational scope-limiting — the ambition stays;
this is the list that makes the ambition testable.

**Legend** — 🔴 blocker (a fresh machine fails) · 🟠 correctness · 🟡 quality ·
🟢 enhancement

---

# PART 0 — Ship-stoppers for external testing

These fail on any machine that is not this one. Nothing else matters until they
are done.

### 0.1 🔴 A fresh clone does not build the current compiler

`HEAD` lacks the `pub mod` declarations for six modules that exist only as
untracked files: `borrow_checker`, `docs`, `formatter`, `package_manager`,
`preprocessor`, `llvm_c_api`. Ten untracked files under `src/` in total, plus ~45
modified tracked files.

**A clone of master today is a materially older compiler than the working tree.**
Anyone you hand this to gets something that does not match any of these documents.

**Action:** commit the working tree. This is the single highest-priority item in
the entire document.

### 0.2 🔴 Developer's local `PATH` is hardcoded into the build

`build.rs` and `src/codegen/build.rs` contain literal paths including:

```
C:\Program Files\Cheat Engine\win64
C:\Program Files (x86)\Passixer\Passixer iPhone Unlocker
C:\Program Files\Common Files\Apple\Mobile Device Support
C:\Program Files (x86)\Common Files\Apple\Mobile Device Support
```

These appear to be a scraped local `PATH` (see the "inject known dependency paths
for LoadLibrary search" logic). They are meaningless on any other machine, and
they are the kind of detail that ends a technical review badly.

**Action:** delete. Replace with `PATH`-relative discovery plus an explicit
override env var.

### 0.3 🔴 Prebuilt runtime objects are not generated

The infrastructure landed (hash-validated embedding, `IRIS_GENERATE_PREBUILT=1`),
but no objects exist yet, so every target machine still needs clang to compile
6,620 lines of C runtime.

**Action:** run the generator per target; add a CI matrix job so releases carry
them.

### 0.4 🔴 `ld.lld` is skipped on MinGW → clang fallback on *every* build

Every run this session printed:

```
iris_codegen: ld.lld link failed (… skipped for MinGW target x86_64-pc-windows-gnu),
falling back to clang
```

A hard clang dependency plus a visible warning on every user build.

**Action:** retarget Windows to `windows-msvc` + `lld-link`, or bundle `lld` in
`toolchain/llvm/bin/` (`find_lld()` already searches there).

### 0.5 🔴 `cargo test` has never been observed running

A test-only `IrFunction` literal in `src/pass/validate.rs` was missing `is_const`,
breaking the whole test build. A one-line fix was applied and **never verified** —
`cargo check` does not build test targets.

**Action:** `cargo check --all-targets`, then `cargo test`. Record the number.

### 0.6 🟠 MSYS2/ucrt64 assumptions (13 references)

Include and library paths assume an MSYS2 ucrt64 install. Fine here, absent
elsewhere.

**Action:** make conditional with a clear diagnostic when missing, not a silent
path that yields confusing link errors.

---

# PART 1 — FIX (defects)

### 1.1 🟠 Named arguments silently evaluate to `0` — **critical**

| Call | Expected | Actual |
|---|---|---|
| `add(a=3, b=4)` | 7 | **0** |
| `add(b=10, a=5)` | 15 | **0** |
| `add(2, b=8)` | 10 | **2** |
| `scale(x=3, y=5, z=2)` | 30 | **0** |
| `add(7, 3)` | 10 | 10 ✓ |

Parses and is accepted; arguments never reach the callee. Silent wrong answers
are the worst failure class. Without a type annotation it instead fails codegen
(`use of undefined value '%v0'`).

### 1.2 🟠 104 of 122 `.iris` tests assert nothing — **systemic root cause**

They pass whenever the program compiles and exits 0. This is *why* 1.1 reached a
release candidate: `test_named_args.iris` prints "All named arg tests passed!"
while printing `t1=0 t2=0 t6=0 t7=0`.

**Action:** every `.iris` test must `assert(...)` or return non-zero on mismatch.
Expect this to surface more defects of the same class — that is the point.

### 1.3 🟠 `str` field in a record inside `result<T, E>`

Mis-types as `i64`, fails LLVM verification (`'%v14' defined with type 'i64' but
expected 'ptr'`). Both ingredients individually well covered; the combination is
not. Blocks `projects/autonomous_regulator/`.

### 1.4 🟡 `when`-arm assignment diagnostic

Assigning an enclosing `var` from `when` arms reports `variable '%30' is used
before it has been assigned` — an internal SSA value, meaningless to a user. A
workaround exists (use `when` as an expression); the diagnostic is the defect.

### 1.5 🟠 `cuda.rs` silently emits NULL for unhandled tensor ops

Unhandled `TensorOp` cases emit `@iris_tensor_op()`, which returns NULL at
runtime. **Action:** hard codegen error instead.

### 1.6 🟠 Unresolved types default to `i64`

Architectural, but list it here because it is the mechanism behind 1.3: an
unresolved inference slot becomes an integer rather than an error. **Action:**
make it a spanned compile error. Cheap change, large fallout, all of it currently
silent corruption.

### 1.7 🔴 `std.ros2` subscriptions cannot read payloads

`wait_for_message(sub, timeout_ms) -> bool` detects arrival but cannot return the
message. No perception, no closed loop. Blocks every robotics claim.

---

# PART 2 — REBUILD (architecture)

From the rustc comparison. rustc types the program on HIR — a source-shaped tree —
then lowers to its CFG. Types flow downhill. **IRIS lowers first and infers types
on the CFG afterwards**, so type information must be re-derived after the source
structure is gone. That ordering is the root cause of the 1.3/1.6 class.

| # | Change | Effort |
|---|---|---|
| 2.1 | **Typed HIR between AST and lowering** — name-resolved, desugared, fully typed. Move HM inference onto it. `IrType::Infer` then becomes unrepresentable in lowering's output rather than something later passes clean up. | Weeks — highest payoff |
| 2.2 | **Borrow check on the IR CFG, not the AST.** A tree cannot be path-sensitive; rustc uses MIR dataflow, which is what makes NLL possible. The CFG already exists and is simply unused for this. | Weeks |
| 2.3 | **Split monomorphisation out of the lowerer.** `src/lower/mod.rs` is 17,262 lines doing desugaring, generic resolution, substitution and SSA construction at once. | Weeks |
| 2.4 | **Spans on all diagnostics**, and continue past the first error in the type and lowering stages as the parser already does. | Days |

**Do not** adopt rustc's query system. It serves incremental compilation across a
large multi-crate codebase; IRIS is a single crate with a file-level `BuildCache`.
Large refactor, problem you do not have.

---

# PART 3 — ADD

### 3.1 Language

| Feature | Priority | Note |
|---|---|---|
| Native tail calls | 🟠 | `test_tco.iris` is interpreter-only; no `musttail` in codegen — native deep recursion can still overflow |
| `From<T>` conversions | 🟡 | Weak `?` ergonomics without implicit error conversion |
| `Clone` / `Copy` traits | 🟡 | No copy-vs-move distinction; everything refcounted |
| Real async runtime | 🟢 | `await` is a channel receive — no state machine, reactor or waker. **Verify before any proposal describes it as an async runtime** |
| Binding patterns `x @ pat` | 🟢 | |

### 3.2 ROS 2 — Phase 1, unblocks all robotics

`subscription_take() -> option<Msg>` (payload!) · executor + callback dispatch ·
standard messages (`LaserScan`, `PointCloud2`, `Image`, `Odometry`, `IMU`,
`JointState`) · QoS profiles (reliability, durability, history depth) · **tf2**
transform buffer/lookup/broadcast · clock and simulated time.

*Deliverable that proves the stack:* one node that subscribes to `/scan`,
transforms into `base_link`, and publishes `/cmd_vel`.

Then Phase 2: services, actions, parameters, lifecycle nodes.

### 3.3 The missing estimation/control layer

`std.ais` is the cognitive layer; `std.ml` the learning layer. The layer between —
where a robot spends most of its cycles — is absent:

- `std.spatial` — quaternions, SE(3), rotations, pose composition *(prerequisite for tf2)*
- `std.filter` — Kalman, EKF, UKF, complementary
- `std.control` — PID, LQR, rate limiting, anti-windup
- `std.planning` — A*, RRT*, occupancy grid, costmap

### 3.4 Stdlib gaps

`std.collections` has only 4 functions (a sketch). `std.time` has no `DateTime`
and no date parsing. `std.ais` is not wired to `std.uncertainty` despite the
latter having 22 functions.

### 3.5 Tests and examples

~12 **multi-feature interaction** programs — the current suite is
one-feature-per-file, which is precisely why new combinations break. Examples for
the uncovered surface: macros, effects, `dyn Trait`, HKT, const generics, task
groups, `defer`, `move`/borrow, `mod` blocks. **Every one must run and assert.**

---

# PART 4 — REMOVE

| # | Item | Reason |
|---|---|---|
| 4.1 | ~~`native-llvm` / inkwell / `llvm_native.rs`~~ | ✅ **Done.** Pinned to LLVM 14, never built by CI, unbuildable against any installed LLVM |
| 4.2 | Hardcoded Cheat Engine / Passixer / Apple paths | 🔴 See 0.2 — must not ship |
| 4.3 | `iris_dyn_call`, `iris_make_trait_object` | Zero emission sites; LLVM emits vtables inline |
| 4.4 | `iris_call_closure`, `iris_call_closure_void` | Unreferenced after 4.1; the only caller was `llvm_native.rs` |
| 4.5 | `iris_tensor_op` / `_load` / `_store` legacy stubs | Return NULL; replace with hard errors (see 1.5) |
| 4.6 | ~20 root debug dumps (`dump_err.txt`, `stdout2.txt`, `nf_llvm.txt`, …) | Not gitignored; would ship in a source tarball |
| 4.7 | `IRIS_NATIVE_ML_BACKENDS` | Obsolete after the `dlopen` migration; currently warns |
| 4.8 | `llvm_stub.rs` | Legacy; `llvm_ir.rs` is the live text backend |
| 4.9 | Dead `link_with_lld` SDK params | `_onnx_sdk`, `_tf_sdk`, `_libtorch_sdk` are ignored post-migration |

---

# PART 5 — Cross-platform verification matrix

Before declaring "testable on other devices", each cell must be exercised:

| Platform | Build | `--emit eval` | Native binary | No clang installed |
|---|---|---|---|---|
| Windows MSVC | ☐ | ☐ | ☐ | ☐ |
| Windows MinGW | ☐ | ☐ | ☐ | ☐ |
| Linux x86_64 | ☐ | ☐ | ☐ | ☐ |
| Linux ARM64 | ☐ | ☐ | ☐ | ☐ |
| macOS ARM64 | ☐ | ☐ | ☐ | ☐ |
| WASM (WASI P1) | ☐ | n/a | ☐ | ☐ |

The CI matrix already covers six targets — it has simply been red since the test
build broke (0.5).

Note: `.cargo/config.toml` scopes `lld-link` under
`[target.x86_64-pc-windows-msvc]`, so it does not affect other platforms.

---

# PART 6 — Sequencing

```
PART 0  (ship-stoppers)          ──►  external testing becomes possible at all
   │
   ├─ 1.2 (assert the tests)     ──►  makes every later fix verifiable
   ├─ 1.1, 1.3, 1.5, 1.6         ──►  correctness
   ├─ PART 4 (removals)          ──►  cheap, do alongside
   │
   ├─ 3.2 ROS 2 Phase 1          ──►  unblocks every robotics claim
   ├─ 3.3 estimation/control     ──►  makes a real robot demo possible
   │
   └─ PART 2 (architecture)      ──►  removes the conditions that cause 1.3/1.6
```

**Minimum to hand IRIS to another person:** Part 0 complete, 1.1 and 1.2 done.

**Minimum for a robot demonstration:** the above, plus 3.2 Phase 1 and enough of
3.3 for a control loop.

**Minimum for the architecture to stop generating this class of bug:** Part 2.1.

---

## What this does not limit

None of the above narrows the ambition. The differentiated assets are real and
untouched by this list: 61 functions of research-grade autonomy in `std.ais`
(active inference, EWC, homeostasis, intrinsic motivation, MAPE-K, safety
constitution), pause-free deterministic memory, native + CUDA + SIMD + WASM
backends, native tensors with tape autodiff, a static effect system, and borrow
checking.

The effect system in particular is the long-term differentiator: being able to
**prove at compile time that a real-time control path performs no allocation and
no I/O** is something no mainstream robotics language offers. This document is
what makes that claim demonstrable on somebody else's hardware.
