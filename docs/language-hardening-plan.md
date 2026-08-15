# Language hardening plan

Goal: a language whose expressiveness is proven rather than assumed, with a
syntax stable enough to build on, tooling that keeps up with it, and a standard
library small enough to be correct.

This plan is written against measurements taken on 2026-08-14 at branch
`rc1-hardening`. Every number below is reproducible; none is inherited.

---

## What is actually true today

| | Measured | Verified? |
|---|---|---|
| AST expression forms | 97 | no |
| IR instructions | 132 | no |
| Reserved keywords | 51 | textual presence only |
| stdlib public functions | 621 | **314 referenced (50.6%)** |
| stdlib modules | 42 | **6 imported by nothing** |
| `.iris` corpus | 125 files | **3 contain an assertion** |
| `cargo test` | 1767 pass / 2 fail / 156 binaries | yes |

The suite figure is the one that misleads. **1767 passing Rust tests say the
compiler's Rust code behaves; they say very little about whether the language is
correct.** Nothing globs `tests/*.iris`, so the 125-file corpus is executed by
nothing, and 122 of those files assert nothing — they prove "compiles and exits
0". A feature that silently returns `0` reached a release candidate this way.

### The evidence that shapes the ordering

Two example suites written on 2026-08-14 — 1,405 lines of asserting IRIS —
surfaced **eleven defects, roughly one per 130 lines**:

- none was visible from reading the compiler source;
- three were hiding behind another bug, and only appeared once the first was
  fixed;
- two made native codegen emit invalid LLVM IR for ordinary domain modelling
  (a `choice` stored in a `record`);
- one made the standard parallel-counter idiom segfault.

That rate is the single most important input to this plan. Writing programs is
currently the highest-yield defect-finding activity available, by a wide margin.

---

## Sequencing

The corpus appears **first and last**, because it is the instrument as well as
the deliverable. You cannot know whether the language is solid — or whether a
syntax is worth freezing — until enough real programs exist to find out. Freeze
first and you preserve defects like `==` on two enums failing at runtime.

### Phase 0 — Adversarial corpus (≈20 files)

Small, deliberately hostile programs that combine features. The existing corpus
is one-feature-per-file, which is precisely why feature *interaction* breaks:
generics inside traits inside modules, effects across closures, enums in records
in results, `when` over generic ADTs, `par for` over captured state.

Each file must `assert` and must run under **both** `--emit eval` and
`IRIS_FORCE_INTERP=1`, because they take different paths and have disagreed.

**Gate:** the defect list stops growing — two consecutive new programs find
nothing. Until then, keep writing; the rate says there is more to find.

### Phase 0 — COMPLETE (2026-08-15)

**18 files, 17 entry points, all passing on both `--emit eval` and
`IRIS_FORCE_INTERP=1`.** `c08` is interpreter-only and marked so: `dyn Trait`
has no native backend (#18b).

**Fifteen defects found, nine fixed.** Rate: roughly one per 120 lines of
asserting IRIS.

| Fixed | |
|---|---|
| #2 | `str` in a record inside `result` — unblocked `projects/autonomous_regulator` |
| #6 | enum field in a record broke native codegen |
| #12 | `par for` never passed its captures to the loop body |
| #19 | **signed division miscompiled** by strength reduction |
| #20 | every diagnostic in every CRLF file was misplaced |
| #23 | FFI out-parameters — six ROS 2 `take_*` functions made reachable |
| #24 | `par for` was a data race by construction |
| #25 | deep recursion crashed the process instead of erroring |
| — | foreach (12 tests), `select` (3), generics (12), singles (7) |

| Still open | |
|---|---|
| #15 | a user function is silently discarded when a builtin shares its name |
| #18/#18b | `dyn Trait` has no native backend |
| #21/#22 | nested generics, and generics over container types |
| #7/#9/#10/#13 | module-system and trait/effect gaps |
| #20b | `span_table` invalidated by every optimisation pass |
| #25 | interpreter uses ~190 KB of host stack per IRIS call |

**Two observations worth carrying forward.**

*The two most serious defects were guards that existed and could never fire.*
Effect subsumption was unfalsifiable because `inferred` was seeded from
`declared`; the recursion guard was set to 5,000 against a real limit of ~350.
Both read as correct code in review. **For every guard, there should be a test
that watches it fire** — the sandbox policy, the borrow checker's rejection
paths and the effect masks have not been probed this way.

*Running both backends found four defects that either alone would have missed* —
labelled `continue` (interpreter only), `dyn Trait` (native only), the FFI cell
double-free (native only), and deep recursion (interpreter only). The
both-backends rule is not ceremony.

*And two were invisible to both.* The division miscompilation and the `par for`
race produced **correct answers on every run**. Neither was catchable by testing
outputs; they needed an arithmetic identity checked and a runtime function read.
That is the argument for asserting *properties*, not just values.

### Phase 1 — Expressiveness blockers

Every item below was hit while writing ordinary code, not while probing edges.
All are recorded with reproductions in `known-issues.md`.

| # | Defect | Why it blocks real programs |
|---|---|---|
| 8 | `==` on two `choice` values fails **at runtime** | comparing enum values is elementary; an untested branch ships broken |
| 7 | record field typed by a *brought* module mangles as generic | blocks composing your types over stdlib types |
| 9 | `pub bring` re-exports functions but not types | every file must import every type module directly |
| 10a | `effect` clause rejected on trait method declarations | a trait cannot state the effect bound its impls must respect |
| 10b | cross-module trait impl treated as `pure` | spurious `E0302`; becomes a false build failure under `--strict-effects` |
| 13 | `pub` unenforced across modules | needs per-item provenance; design work, not a patch |
| — | integer literals are always `i64` and do not coerce | `val r: i32 = f(); r == 5` fails |

**Gate:** each has an asserting `.iris` file that passes, and its `known-issues`
entry is marked FIXED with the mechanism.

### Phase 2 — Lock the syntax

A grammar that is *checked against the parser*, not written beside it — drift
must be a build failure, or the specification becomes prose that quietly stops
being true.

- EBNF for the 51 keywords and 97 expression forms
- A conformance test asserting the grammar accepts exactly what the parser does
- A stability policy: what may change, what may not, how deprecation works

**Gate:** the grammar file and the parser cannot disagree without CI failing.

### Phase 3 — Developer experience

Stronger than expected already: LSP with 14 capability providers, DAP, REPL,
formatter, docs generator, package manager. The remaining gaps are narrow.

- **Error quality.** Errors like `type mismatch: %set__Set vs %set__Set__set__Set__T`
  named a struct defined nowhere. Every diagnostic should name the source
  construct and the fix, as the new type-parameter error now does.
- **Discoverability.** `--strict-effects` existed only as an env var, named in a
  comment that corresponded to nothing. Audit for other unreachable features.
- **`iris policy build`** — emit `.o` + generated `.h` + an effect manifest, so a
  policy can be compiled into a host written in something else. The ABI already
  works: symbols emit unmangled and a scalar policy needs exactly one runtime
  symbol.

**Gate:** a new user can install, write a policy, and link it into a C host from
documentation alone.

### Phase 4 — Stdlib maturity is a *cut* decision first

307 of 621 public functions are referenced by nothing; six modules are imported
by nothing. Maturing all 42 modules is a year of work. Cutting to the autonomy
core and deepening that is a quarter.

- **Cut or mark experimental:** `svg`, `termplot`, `http_server`, `sql`, `csv`,
  `table`, `kv` — general-language surface, unexercised, and diluting the pitch.
- **Deepen:** `std.ais` (11/61 referenced) is the module with the strongest claim
  to novelty and the weakest evidence. `std.adaptive` (29/36) is the model to
  follow.
- **Finish:** `std.ros2` at 3/29 — the C bridge already implements the payload
  `take_*` functions; the stdlib binds none of them, and the FFI layer cannot
  pass an out-parameter. One mechanism unlocks all six.

**Gate:** every retained public function is called by an asserting program.

### Phase 5 — Conformance corpus (125 files)

Rewrite the corpus against the locked syntax so that it *is* the executable
specification, and wire it into `cargo test` — which today globs nothing.

- every file asserts, or returns non-zero on mismatch
- a Rust harness enumerates `tests/*.iris` with an explicit expected-failure list
  (files that test error detection, e.g. `test_borrow_error`)
- a ratchet: the count of non-asserting files may only decrease

**Gate:** `cargo test` executes all 125 files and every one asserts.

---

## What "guarantee" can honestly mean

Not "fully end-to-end guaranteed" — no language ships that claim, and per
`iris-claims` we should not write it down.

The defensible version, and it is a strong one:

> Every language feature has an executing program that asserts its result, and
> the grammar is machine-checked against the parser.

That is reachable, checkable by a reviewer in minutes, and more than most
languages can demonstrate.

---

## Standing rules

- **Diff failure *names*, never totals.** A flaky wall-clock test (#11) makes the
  suite total non-reproducible and has already disguised itself as a regression.
- **Never edit `src/` while a `cargo test` run is in its build phase.**
- **A fixed bug often reveals another.** Three did this session. Re-measure after
  every fix rather than assuming the count moved by one.
