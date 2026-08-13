# IRIS Expressiveness Audit

**Date**: 2026-08-13
**Method**: language surface counted from `src/parser/ast.rs`; feature coverage
cross-referenced against `tests/`; behaviour taken from `cargo test
--no-fail-fast` runs performed during this session.

Grades follow the project standard: **Verified** (built and ran it here, read the
output) · **Tested** (a dedicated test exists in-tree, not run here) ·
**Present** (found in compiler source, behaviour unconfirmed) · **Absent** (no
implementation *and* no test). Limits are stated next to each capability.

---

## 1. Headline

The *surface* is broad and, in places, genuinely ahead of mainstream systems
languages — algebraic effects with resumable continuations, higher-kinded types,
and a rich pattern language sit in a 35-expression-form grammar.

The *reliability of that surface in combination* is the gap, and it is
concentrated in one place: **type parameters that cannot be resolved without an
annotation.** That single cause accounts for 18 of the ~40 currently failing
tests.

| Dimension | Count |
|---|---|
| Expression forms (`AstExpr`) | 35 |
| Statement forms (`AstStmt`) | 18 |
| Type forms (`AstType`) | 23 |
| Reserved words (`src/parser/lexer.rs`) | 51 |
| `.iris` corpus files | 125 |

---

## 2. The evidence problem, stated first

Two facts limit what any expressiveness claim can rest on. Both were measured
here, not inherited.

**2.1 The `.iris` corpus is not executed by `cargo test`.** No Rust test globs
`tests/*.iris`. Of the 125 files, **120 define `def main()`** — which `iris test`
cannot discover, since it looks for `test_` functions — and only 32 define a
`test_` function. So the richest demonstrations of expressiveness in the
repository run only when someone invokes them by hand.

**2.2 105 of the 125 `.iris` files contain no assertion and no failure return.**
They print results and exit 0 regardless of correctness. (The figure in
`docs/known-issues.md` was 104 of 122; the corpus has grown, the ratio has not
improved.)

Consequently: the ~1600 passing Rust integration tests are the real evidence
base, and they are decent evidence — they drive `compile()` on inline source and
assert on output. But they largely exercise **one feature per file**, which is
precisely why feature *interactions* are where defects concentrate.

**Never cite the test count as evidence of correctness without 2.1 and 2.2.**

---

## 3. Data modelling

| Capability | Grade | Limit stated adjacent |
|---|---|---|
| Records, field access, struct update | Tested | `test_struct_update` |
| Default record fields, partial override | Tested | prior session reports Verified; not re-run here |
| `choice` enums with payloads | Tested | |
| Exhaustiveness checking | Tested | 4 dedicated test files |
| Tuples + `.0`/`.1` access | Tested | |
| Type aliases | Tested | resolved at lowering |
| Generic records / functions | Tested | **but see 3.1 — unusable without annotation in the common case** |
| Higher-kinded types `F[_]` | Tested | `test_hkt`; recursive monomorphisation. Safe to claim |
| Associated types | Tested | `test_assoc_types`, not run here |
| Refinement types | Present | only `test_refine_fail` — the negative case |

### 3.1 The central expressiveness defect — **Verified broken**

A generic constructor called with no argument has nothing to infer its type
parameter from, and lowering fails:

```iris
val xs = list()          // List(Infer)
var sum = 0
for x in xs { sum = sum + x }
//            ^^^^^^^^^ Lower(TypeMismatch { expected: "i64", found: "_" })
```

Same shape for `set_new()` and `heap_new()`, which are declared
`set_new[T](v: T)` — the tests call them with zero arguments:

```
Pass(TypeError { func: "set__set_new__",
  detail: "type mismatch: %set__Set vs %set__Set__set__Set__T" })
```

Measured impact: **18 failing tests** — 12 in the two `foreach` binaries, 4 in
`set`, 2 in `heap`. There are **224 zero-argument constructor call sites** across
`tests/`.

Root cause is structural, not local: type inference runs *after* lowering
(`docs/architecture-vs-rustc.md` §3.1), so a type that was obvious in the source
must be re-derived from the CFG. The escape hatch already exists —
[`src/lower/mod.rs:4828`](../src/lower/mod.rs#L4828) resolves `list()`'s element
type from a `val x: T = …` annotation — so **annotated code works and unannotated
code does not**. That is the honest statement of the limit.

---

## 4. Pattern matching — the strongest area

| Capability | Grade |
|---|---|
| Struct patterns, incl. test-and-bind | Tested |
| Or-patterns `A \| B` | Tested |
| Range patterns `1..=9` | Tested |
| Guards `if x == y` | Tested |
| Slice patterns `[a, ..rest]` | Tested |
| `if let` / `while let` | Tested |
| Refutable `let` | Tested |
| Exhaustiveness + wildcard handling | Tested |

`docs/known-issues.md` reports struct/or/range/guard patterns as Verified in an
earlier session. I did not re-run them, so they are graded Tested here rather
than inheriting that claim.

**Known limit:** assigning an enclosing `var` from `when` arms fails SSA
construction (`known-issues.md` §3). Use `when` as an expression instead — which
is better style anyway, but the diagnostic points at an internal SSA value rather
than at the user's assignment.

---

## 5. Functions and abstraction

| Capability | Grade | Limit |
|---|---|---|
| Default parameters | Tested | binary passes in this session's runs |
| **Named arguments** | **Broken** | **`f(a=1, b=2)` silently evaluates to `0`.** Worst defect in the tree: wrong answer, no error. Treat as unusable |
| Extension methods (first param `T`) | Tested | prior session reports Verified |
| Closures + closure types `\|T\| -> R` | Verified (partial) | `\|str\| -> str` compiles and runs inside `std.http_server`, exercised by 30 passing tests |
| Traits, trait bounds | Tested | |
| Blanket impls | Tested | `test_blanket_impls`, `test_blanket_multi_type` |
| Trait objects, incl. return position | Tested | |
| Variance | Tested | |
| Operator overloading | Tested | `test_arith_overload` |
| `const def` (compile-time eval) | Tested | |
| Macros `defmacro` | Tested | |

Named arguments deserve emphasis: it is the one feature here that produces a
**silently wrong value** rather than an error, and it is a feature users reach for
early. Either fix it or make it a hard parse error before release.

---

## 6. Control flow

| Capability | Grade | Limit |
|---|---|---|
| Labelled `break`/`continue` on `loop`/`for`/`while` | Tested | Label follows the keyword — `for outer i in 0..n { break outer; }`. No sigil |
| `for … by` step | Tested | |
| `for` destructuring | Tested | |
| `try`/`catch` | Tested | |
| `defer` | Tested | |
| Generators / `yield` | Tested | `test_yield_fail` exists alongside the passing cases — limits not characterised here |
| `for x in list` | **Broken for untyped lists** | see 3.1 |
| Tail-call optimisation | Present, **interpreter only** | no `musttail` in codegen; native deep recursion can still overflow the stack. Always qualify this claim |

---

## 7. Effects — the genuine differentiator

| Capability | Grade |
|---|---|
| `effect` rows on signatures | Verified |
| `handle { … } with { op(x) => … }` | Tested |
| Resumable continuations `op(x) -> resume(k) => k(v)` | Tested |
| Effect masks | Tested |
| Effect polymorphism | Tested |
| Static effect checking | Tested |

This is the area with the strongest claim to novelty: an effect system that can
constrain what a control path is permitted to do is not available in mainstream
robotics languages.

**Limits, stated adjacent — two found and fixed here:**

1. **Effect names are ordinary identifiers with no registry.** A reserved word
   cannot be used as one: `effect net, spawn` in `std.http_server` failed to
   parse, and because the span was misreported past the end of the *importing*
   file, it broke every program that brought the module while pointing nowhere
   useful. Renamed to `thread`.
2. **An unregistered effect name silently dispatched to the wrong string.** The
   codegen string table never collected `CallExtern` names, so the lookup fell
   back to index 0 and passed whatever string happened to be first — producing
   runtime failures naming absurd effects (`no handler for effect ','`,
   `no handler for effect '=='`) and, where the table was empty, invalid LLVM
   (`use of undefined value '@.str.0'`). Now collected, and a miss is a hard
   codegen error rather than a wrong answer.

Both were silent-wrong-answer defects in the flagship feature. Worth noting when
judging how much of the surface is load-bearing.

---

## 8. Concurrency

| Capability | Grade | Limit |
|---|---|---|
| `spawn`, channels, `send`/`recv` | Tested | |
| `select` over channels | **Failing** | 3 failures in `test_select` this session |
| Task groups / nursery | Tested | |
| `par for`, `par_map` | Tested | |
| Atomics, mutexes | Tested | |
| `async def` / `await` | Present | **`await` desugars to a channel receive. There is no state machine, reactor or waker.** Do not describe this as an async runtime |

---

## 9. Memory and safety

| Capability | Grade | Limit |
|---|---|---|
| Reference counting + cycle collection | Tested | pause-free, bounded latency |
| Move semantics | Tested | `test_move`, `test_move_error` |
| Borrow checking | Tested | **Runs on the AST, not the CFG**, so it cannot be path-sensitive: it will accept some unsound programs and reject some sound ones. Not fixable on the AST — the information is absent from the representation (`architecture-vs-rustc.md` §3.2) |
| Weak references | Tested | |

---

## 10. Modules and strings

| Capability | Grade | Limit |
|---|---|---|
| `bring` for stdlib and local files | Verified | exercised throughout this session |
| `mod` blocks, selective imports | Tested | 1 failure in `module_system_*` this session |
| Circular import rejection | Tested | |
| `??` null coalescing, on `option` **and** `result` | Tested | `test_expressiveness` — one of the 20 files that does signal failure |
| `str * int` repetition, `+` concat, comparisons | Tested | same file |
| f-strings / interpolation | Tested | |
| Conditional compilation `#ifdef` | Tested | |

---

## 11. Verdict

**What is defensible today:** a broad, unusually expressive surface — effects
with resumable continuations, HKT, a complete pattern language, traits with
blanket impls and objects — backed by ~1600 passing Rust integration tests.

**What must be said alongside it:** the corpus that best demonstrates that
surface (125 `.iris` files) is not executed by CI, and 105 of those files cannot
fail. Generic constructors are unusable without a type annotation, named
arguments silently return `0`, and the flagship effect system contained two
silent-wrong-answer defects until today.

**The honest framing** (unchanged from `iris-claims`, and it holds up):

> Broad, working feature coverage with a known hardening gap in feature
> interaction, and a concrete plan to close it.

That framing is *more* credible with the specifics above attached, not less. The
root cause is single and named: inference after lowering, with an `i64` default
that turns "unknown" into "integer".

### Ranked by expressiveness value recovered per unit of work

1. Run the `.iris` corpus in CI and require every file to assert — converts 105
   files from decoration into evidence.
2. Make unresolved type parameters a diagnostic with a span, then annotate the
   224 call sites — recovers 18 tests and makes the limit legible.
3. Fix or hard-error named arguments — removes the only silent-wrong-answer in
   the user-facing surface.
4. Qualify TCO and `async` in all prose, or implement `musttail` and a real
   scheduler.
