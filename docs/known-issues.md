# Known Issues

Defects found by writing new IRIS programs against v1.0.0-rc1 and verifying
their output. Each entry has a minimal reproduction.

---

## 1. Named arguments silently evaluate to 0 — **FIXED**

> **Fixed** in `src/lower/mod.rs`. The parser had always collected named
> arguments into `AstExpr::Call::named_args`, but `lower_call` destructured the
> node with `..` and never read that field — so a call written entirely with
> named arguments was lowered with an argument list of length **zero**, and the
> callee read whatever was left in the argument registers.
>
> `lower_call` now takes `named_args` and folds them into positional order
> before any other code inspects `args`, against a new `fn_param_names` table
> built in the same loop as `fn_defaults`. Gaps are filled from parameter
> defaults; a gap with no default is a compile error rather than a short
> argument list. Unknown parameter names, duplicates and
> positional/named collisions all now fail at compile time, with a did-you-mean
> and the declared parameter list. Named arguments on a builtin or extern are
> rejected explicitly, because parameter names are not known for those.
>
> Resolution runs only when `named_args` is non-empty, so purely positional
> calls take exactly the path they always did.
>
> **This defect is the origin of project rule 3.** `tests/test_named_args.iris`
> printed seven values and then `All named arg tests passed!` while every one of
> those values was wrong; it asserted nothing, so it reported success for the
> entire life of the defect. That file now asserts all seven, plus four
> non-commutative cases (`sub(b=1, a=10) == 9`) that would catch a resolver
> which appended named arguments in written order rather than parameter order.
>
> Verified identical on `--emit eval` and `IRIS_FORCE_INTERP=1`.

### Original report

`f(a=1, b=2)` parses and is accepted, but the arguments do not reach the callee.

```iris
def add(a: i64, b: i64) -> i64 { a + b }

def main() -> i64 {
    val t: i64 = add(a=3, b=4);   // expected 7
    println(to_str(t));           // prints 0
    0
}
```

Observed across every form:

| Call | Expected | Actual |
|---|---|---|
| `add(a=3, b=4)` | 7 | **0** |
| `add(b=10, a=5)` | 15 | **0** |
| `add(2, b=8)` | 10 | **2** (only the positional argument arrives) |
| `scale(x=3, y=5, z=2)` | 30 | **0** |
| `add(7, 3)` | 10 | 10 — positional is correct |

Without a type annotation the same call fails at codegen instead:
`use of undefined value '%v0'`.

**Why it was not caught:** `tests/test_named_args.iris` prints results but never
asserts them, so it reports *"All named arg tests passed!"* while every value is
wrong. See issue 4.

**Status:** open. Named arguments should be treated as unusable until fixed.

---

## 2. `str` field in a record inside `result<T, E>` — **FIXED**

> **No longer reproduces**, confirmed 2026-08-16. A record with a `str` field
> constructed inside `ok(...)` and returned as `result<Record, str>` round-trips
> correctly on both backends, and `projects/autonomous_regulator/` — which this
> issue said it blocked — now runs end to end, which also satisfies that item
> of the release gate.
>
> Fixed incidentally by the `MakeStruct` field-store work recorded under #6,
> which made the store take the value's *emitted* type as authoritative. Pinned
> by `a_str_record_field_survives_a_result_round_trip` and
> `the_err_arm_of_a_str_record_result_still_carries_its_message` in
> `tests/when_arm_assignment.rs` so it cannot regress unnoticed.

### Original report

A record containing a `str` field, constructed inside `ok(...)` and returned as
`result<Record, str>`, mis-types the string as `i64` and fails LLVM verification:

```
error: '%v14' defined with type 'i64' but expected 'ptr'
```

Both ingredients are individually well covered — `result<T, str>` appears in 8+
test files and `str` record fields in several more — so this is an interaction
defect, not a missing feature.

Currently blocks `projects/autonomous_regulator/` from running.

---

## 3. Assigning to an enclosing `var` from `when` arms — **FIXED**

> **Fixed** on 2026-08-16. Two defects were stacked here, and the second was
> worse than the one reported.
>
> **The compile failure.** A braced arm body (`{ a = 1; }`) has no tail
> expression, and the fallback for a block with no value returned a bare
> `fresh_value()` — a `ValueId` that no instruction defines. Anything consuming
> it failed validation with "variable '%N' is used before it has been assigned",
> naming an internal SSA value rather than the source. It now emits a real `0`,
> and the same phantom was fixed in three other places (`Mask` bodies,
> `task_group_join`, `task_group_cancel`) where it was latent.
>
> **The silent wrong answer underneath.** With that fixed, `choice` arms
> compiled but produced the *old* value: `lower_when_expr` never threaded
> assigned variables to the merge block, so the assignment was simply lost. The
> option and result paths already threaded them, which is why only `choice` was
> wrong once it compiled. `when c { C.X => { a = 1; } }` left `a == 0`.
>
> The workaround (`when` as an expression) remains better style and is still
> tested, but it is no longer required.
>
> **Tested by** 9 tests in `tests/when_arm_assignment.rs`, including an arm that
> does *not* assign — every predecessor must still supply an argument for the
> merge parameter.

### Original report

```iris
var applied = 0.0;
when decision {
    ok(cmd)     => { applied = cmd.delta_c; }
    err(reason) => { applied = 0.0; }
};
```

fails SSA construction with:

```
error[E0200]: variable '%30' is used before it has been assigned a value
```

**Workaround:** use `when` as an expression and let each arm produce the value:

```iris
val applied = when decision {
    ok(cmd)  => cmd.delta_c,
    err(_)   => 0.0,
};
```

The workaround is also better style. The defect is that the diagnostic points at
an internal SSA value rather than at the assignment.

---

## 4. 104 of 122 `.iris` tests assert nothing — **GATED; backlog shrinking**

> **The mechanism is fixed; the backlog is not yet cleared.** Two separate
> problems were tangled here, and the one that was never stated is the one that
> let the other persist.
>
> **Nothing globbed `tests/*.iris`.** The corpus was never executed by
> `cargo test` at all, so a file could rot indefinitely with no run noticing.
> `tests/iris_corpus.rs` now drives every file through the real CLI. On its
> first run it found three native crashes (`test_methods`, `test_move`,
> `test_pattern_guards`) that hand sweeps had missed, because those sweeps used
> `IRIS_FORCE_INTERP=1`, which bypasses codegen — see #51.
>
> **Most files still assert nothing:** 96 of 139. That number is now *enforced
> downward* rather than tracked by hand:
>
> | Gate | What it prevents |
> |---|---|
> | `every_iris_test_asserts_or_is_listed` | a **new** non-asserting file joining the backlog unnoticed |
> | `the_needs_assertions_list_is_accurate` | the backlog list drifting out of date once a file is converted |
> | `every_iris_test_runs` | a corpus file rotting without CI noticing |
> | `negative_tests_still_fail` | a negative test silently starting to pass, i.e. losing the check it guards |
> | `the_known_broken_list_is_accurate` | the debt register describing a state that is no longer true |
>
> The last of those failed on its first run — `test_adaptive.iris` was listed as
> broken but runs natively — which is the register doing its job on day one.
>
> **What remains:** converting the 96. That is mechanical but cannot be
> automated honestly, because each file has to be *run* and its real output
> values read before an assertion can be written; guessing the expected value
> would produce a test that passes while asserting the wrong thing, which is
> worse than no assertion at all. `NEEDS_ASSERTIONS` in `tests/iris_corpus.rs`
> is the live list.
>
> **Do not cite the corpus as evidence of correctness until that list is
> empty.** A green corpus still demonstrates "compiles, runs, exits 0" for those
> 96 files.

### Original report

Most `.iris` tests print results without asserting them. They pass whenever the
program compiles and exits 0, regardless of whether the output is correct.

```
total .iris tests:                        122
with no assert and no failure return:     104
```

This is the root cause of issue 1 reaching a release candidate, and it means a
green `.iris` suite currently demonstrates *"compiles and does not crash"* rather
than *"computes the right answer."*

**Recommended fix:** require every `.iris` test to either call `assert(...)` or
return a non-zero code on mismatch. Converting the existing 104 is mechanical and
would likely surface further defects of the same class as issue 1.

---

## 5. Tail-call optimisation is interpreter-only — **FIXED (superseded by #35)**

> **Fixed** on 2026-08-16. The framing here was wrong in both directions, which
> is why #35 was raised separately and carries the real account.
>
> TCO was not "interpreter-only" — it did not exist *anywhere*. The interpreter
> had no such optimisation either; `test_tco.iris` only passed because the depth
> limit used to be 5,000 and it genuinely built 50,000 frames. Meanwhile deep
> recursion often *did* survive natively, because clang applies its own
> tail-call optimisation to the emitted IR.
>
> `TailCallPass` now does it as an IR transformation, so the property belongs to
> the language rather than to whichever backend happens to be running. See #35
> for the transformation, the measurements and the remaining limit.

**Original report:** `tests/test_tco.iris` states it exercises TCO "in the
interpreter". There is no `musttail` in codegen, so deeply recursive functions
can still overflow the stack in native binaries. The capability should not be
described without that qualification.

---

## 6. Record with an enum field, returned from a function — **FIXED**

> **Fixed** in `src/codegen/llvm_ir.rs`, `MakeStruct` field stores. The store now
> takes the value's *emitted* type as authoritative and coerces to the slot type
> (`inttoptr` / `ptrtoint`) when they disagree. Verified: store, read-back and
> `when` dispatch on the field all work natively, including alongside `str`
> fields and inside a `list`. `examples/08_fleet_service` builds natively as a
> result. The original report follows.

A record containing a `choice`-typed field produces invalid LLVM IR when it is
returned from a function. The enum tag is an `i64` but the struct slot is
emitted as `ptr`.

```iris
choice K { A, B }
record R { k: K, n: i64 }
def mk() -> R { R { k: K.A, n: 1 } }      // <- the trigger
def main() -> i64 { val r = mk(); assert(r.n == 1); 0 }
```

```
module.ll:357:13: error: '%v0' defined with type 'i64' but expected 'ptr'
  store ptr %v0, ptr %sgep2_0, align 8
```

Constructing the same record **inline** is fine — only the function-return path
fails, which is why this hides easily:

```iris
def main() -> i64 { val r = R { k: K.B }; 0 }   // compiles and runs
```

**Severity is higher than it looks.** Modelling a domain with `choice` and
storing it in a record is the ordinary way to make illegal states
unrepresentable, so this blocks idiomatic domain modelling in any natively-built
program. `examples/08_fleet_service` hits it and currently runs under the
interpreter only.

Same family as issue 2 — a record-field type mismatch in codegen.

**Status:** open.

---

## 7. Record field typed by a *brought* module's record — **open, lowering**

A record whose field type comes from another module is mangled as though the
record were generic over that field.

```iris
bring std.ais
record Holder { s: RunningStats, n: i64 }
def make() -> Holder { Holder { s: running_stats_new(), n: 0 } }
```

```
type error in function 'make' —
  type mismatch: %Holder__ais__RunningStats vs %Holder -- Return value %2
```

This blocks composing your own types over stdlib types, which is what building
anything real requires. Workaround: inline the fields you need as scalars.

**Status:** open.

---

## 8. `==` on two enum values is unsupported — **FIXED**

> **Fixed** in `src/interp/mod.rs`: the binop match gained `CmpEq`/`CmpNe` arms
> for `Enum`. `IrValue` already implemented `PartialEq` structurally for every
> aggregate; the arm was simply missing, so the comparison fell through to
> "unsupported binop" at *runtime*. Verified identical on both backends across
> equal, unequal and bound-variable cases.
>
> Note the failure mode this removes: an untested branch containing an enum
> comparison would compile, ship, and abort in production.
>
> The original report follows.

### Original report

```iris
choice Health { Nominal, Degraded }
assert(Health.Nominal == Health.Nominal);
```

```
error[E0404]: [runtime error] type error —
  unsupported binop CmpEq on Enum(3, []) and Enum(3, [])
```

Comparison must go through `when`. The idiomatic workaround is to define
equality on a rank function:

```iris
def health_rank(h: Health) -> i64 { when h { Health.Nominal => 0, Health.Degraded => 1 } }
def health_eq(a: Health, b: Health) -> bool { health_rank(a) == health_rank(b) }
```

**Status:** open. Note the error arrives at *runtime*, not compile time, so an
untested branch ships broken.

---

## 9. `pub bring` does not re-export types — **open, module system**

`pub bring "types.iris"` re-exports public *functions* but not `record` or
`choice` declarations, so a downstream file cannot name them and fails with
`cannot find 'Health'`. Every file that names a type must bring its defining
module directly. Brings are de-duplicated, so this is safe, just verbose.

**Status:** open.

---

## 10. Effect clauses and traits — **FIXED (declaration half)**

> `def read(self) -> i64 effect io` inside a `trait` was a **parse error**, so a
> trait had no way to state the effect bound its implementations must respect.
> `parse_trait_def` went straight from the return type to looking for `{`.
>
> It now parses the clause exactly as `parse_function` does and stores it on a
> new `AstTraitMethod::effects`. Verified on both backends.
>
> **Still open:** the stored bound is not yet *checked* against implementations,
> and 10b (a cross-module trait impl treated as `pure`) is untouched.

### Original report

**10a. The parser rejects an `effect` clause on a trait method declaration.**

```iris
trait Summarise {
    def summarise(self: Self) -> str effect io, alloc   // syntax error
}
```

So a trait cannot state the effect bound its implementations must respect —
which is precisely where such a bound is most useful.

**10b. A cross-module trait impl is treated as `pure`.** An impl reached through
a `bring` is mangled to `types__Summarise__types__Device__summarise`, and the
effect checker's caller-row lookup misses that name, so every effectful call
inside reports a spurious `E0302`:

```
error[E0302]: function `to_str` requires effects `alloc, io`
  but caller `types__Summarise__types__Verdict__summarise` has effects `pure`
```

Writing `effect io, alloc` on the impl is accepted by the parser but does **not**
silence it — evidence that the declared row never reaches the inferred map.
Harmless today (stderr only, exit 0) but it becomes a false build failure under
`--strict-effects`.

**Status:** open.

---

## 11. `test_autodiff_determinism_profiling` is flaky — **FIXED**

> The test asserted `std_dev < 100.0` microseconds across 1000 timed
> iterations, which measures the host's scheduler rather than the code. On this
> two-core machine it failed roughly one run in three, which is why the standing
> rule "diff failure names, never totals" existed: the suite's totals were not
> comparable between runs.
>
> The deeper problem is that a test named *determinism* asserted nothing about
> determinism, and nothing about correctness either — a version that always
> returned `0.0` would have passed every time it was fast enough.
>
> **Rewritten** to assert what the name promises: all 1000 runs must produce a
> **bit-identical** gradient (comparing `to_bits`, so a stray `-0.0` or `NaN`
> is caught), and that gradient must equal `y + cos(x)`, the true derivative of
> `x*y + sin(x)`. The correctness assertion passed first time, so the autodiff
> was right all along — it simply had nothing checking it.
>
> Timing is now reported rather than asserted, apart from a sanity bound three
> orders of magnitude looser than scheduling noise, which still catches a real
> performance collapse. 5 of 5 runs pass where the old version failed 1 in 3.
>
> A test that fails for reasons unrelated to the code is worse than no test: it
> trains readers to ignore red.

### Original report

The test asserts that the standard deviation of 1,000 timed autodiff runs is
under 100 µs (`tests/autodiff_determinism.rs:192`). On a 2-core laptop under
normal scheduling it swings wildly:

```
run 1: ok
run 2: FAILED — Standard deviation is too high: 354.7619 microseconds
run 3: FAILED
```

Three consecutive runs of the *same binary*, no code change between them.

**Why it matters beyond the noise:** it makes the suite total non-reproducible,
so a run of 34 failures versus 35 cannot be read as an improvement without
diffing the failure *names*. It briefly looked as though an unrelated change had
fixed an autodiff test.

Wall-clock variance on a scheduled OS is not a property of the compiler. The
assertion should either be removed, moved behind an explicit benchmark feature,
or rewritten to measure instruction counts rather than elapsed time.

**Status:** open.

---

## 12. `atomic_add` inside a `par for` segfaults natively — **FIXED**

> **Fixed** in `src/codegen/llvm_ir.rs`, three sites. Root cause: `IrInstr::ParFor`
> carries the lambda-lifted body's captures in `args`, and codegen destructured it
> with `..` — the captures were never passed. The lowerer emits
> `__par_body_N(loop_var, cap0, cap1, …)` while the runtime can only call
> `fn(int64_t i, void* env)`, so every capture parameter read an uninitialised
> register.
>
> The fix packs the captures into a closure environment with the same
> `iris_make_closure` helper closures already use, passes it as the runtime's
> long-present 4th argument (`void* arg` — the C side always had it; codegen
> simply never supplied it), gives `__par_body_*` the matching
> `(i64 %var, ptr %env)` signature, and unpacks them in the entry block.
>
> A second defect surfaced during the fix: the capture-extraction fallback
> returned the *boxed* `IrisVal` unchanged, so `iris_atomic_add` received the
> wrapper rather than the atomic. It now routes through
> `runtime_unbox_helper_for_type`, which already covered every heap kind.
>
> Verified natively: single capture, multiple captures of mixed kinds, and
> correct values (16 iterations × step 3 = 48), matching the interpreter.
> `examples/08_fleet_service` uses the parallel-counter idiom directly.
>
> The original report follows.

Each construct works on its own; only the combination crashes.

```iris
def main() -> i64 {
    var t = atomic(0)
    par for i in 0..4 { atomic_add(t, 1); }
    assert(atomic_load(t) == 4);
    0
}
```

```
JIT/Native execution failed (falling back to interpreter):
  runtime error (exit exit code: 0xc0000005)
```

`0xc0000005` is an access violation. Isolated by bisection:

| Program | Native |
|---|---|
| `atomic_add` outside any loop | works |
| `par for` with no atomic in the body | works |
| `atomic_add` **inside** `par for` | **segfault** |

The interpreter computes the right answer in all three, so `--emit eval` masks
it via the fallback — the failure is only visible in the warning line, or in a
`iris build` binary.

Likely the atomic handle is not being captured correctly into the parallel
body's closure environment. This is the same shape as the counter idiom in most
parallel code, so it is worth fixing before `par for` is presented as usable.

**Status:** open. `examples/08_fleet_service/main.iris` works around it by
exercising `par for` and atomics separately.

---

## 13. `pub` is not enforced across module boundaries — **FIXED**

> **Fixed** on 2026-08-16. The design the original report asked for was mostly
> present already: `mangle_module_symbols` renames every brought symbol to
> `module__name` and rewrites the defining module's own references to the
> mangled name directly. Internal calls therefore never go through name
> resolution and only *cross-module* references do, so visibility can be
> enforced in one place without reconstructing the module boundary.
>
> `resolve_unqualified_name` now refuses a `prefix__name` candidate when the
> item is not `pub`. Note there are **two** resolution paths, and fixing only
> the obvious one changed nothing: a fallback below it scans every signature for
> any `*__name` suffix to support transitive brings, and that is what was
> actually resolving the private call.
>
> Two false starts worth recording, because both failed *silently*:
>
> * A `thread_local` for the private set never fired — compilation runs on a
>   spawned thread (`lib.rs` gives it a larger stack), so the mangling and the
>   resolution are on different threads and the set read back was always empty.
> * Clearing that set per compilation broke it a second way: the merged AST is
>   cached, so a later compilation reuses it *without* re-mangling, leaving the
>   set empty again.
>
> The visibility set is therefore carried **on the merged AST** and published to
> the lowering thread when lowering begins. That is correct under a cache hit
> and safe when tests compile concurrently.
>
> The diagnostic names the cause rather than only the symptom:
>
> ```text
> error[E0100]: cannot find 'secret' - this variable or function is not defined...
>   help: 'secret' is defined in module 'utils' but is not marked `pub`, so it is
>         not visible here. Add `pub` to its definition to export it.
> ```
>
> `format_undef` no longer wraps a sentence-length suggestion in
> "did you mean '...'?", which read as nonsense.
>
> **Limit, stated:** traits are not covered, because `AstTraitDef` carries no
> `is_pub` — a trait cannot currently be declared private, so there is nothing
> to enforce. Functions, records, choices, consts and type aliases are.
>
> Verified: `test_bring_file_private_not_visible` passes — it was the **only**
> failing test in the suite — and the other 7 module tests, exercising `pub`
> functions, records, consts and transitive brings, still pass. No stdlib module
> declares a non-`pub` item, so nothing there relied on the old behaviour.

### Original report

A non-`pub` function in a brought module is callable from the importing module.

```iris
// utils.iris
def secret(x: i64) -> i64 { x + 100 }   // no `pub`

// main.iris
bring "utils.iris"
def f() -> i64 { secret(0) }            // compiles; should not
```

`tests/module_system_..._visibility_...rs::test_bring_file_private_not_visible`
expects an error and does not get one.

**Why this is not a small fix.** Brought modules are merged into a *single flat
AST* — `merge_dep` in `src/compiler.rs` and the equivalent in
`compile_multi_to_ast`, whose comment says "Merge all functions (including
internal ones)". Simply dropping non-`pub` items would also break the defining
module's own internal calls, since after the merge there is no module boundary
left to distinguish them.

Doing it properly needs per-item module provenance plus a visibility check at
each call site (or module-qualified renaming of private items with rewriting of
only the defining module's references). `is_pub` is already parsed and stored on
every AST item, so the information exists; the enforcement does not.

**Status:** open. Treat `pub` as documentation of intent, not an access control.

---

## 14. A type parameter used only in the return type needs an annotation — **by design**

```iris
val s = set_new()              // error: cannot infer type parameter `T`
val s: Set<str> = set_new()    // fine
```

IRIS runs type inference *after* lowering, so a generic call is monomorphised
before anything downstream could constrain it. A parameter that no argument
mentions therefore has to come from the binding annotation.

As of this session the monomorphiser does consult that annotation (it unifies
the declared return type against the expected type), and an unconstrained
parameter is reported as:

```
cannot infer type parameter `T` for `set_new` — no argument mentions it,
so annotate the binding (e.g. `val x: Set<...> = set_new(...)`)
```

Previously it produced the struct name `%set__Set__set__Set__T`, which is
defined nowhere, and surfaced much later as an opaque pass failure.

This mirrors Rust's "type annotations needed" for `let s = HashSet::new();`.
The difference is that Rust can recover the parameter from *later* use and IRIS
cannot, so the annotation is required rather than merely usual.

---

## 15. A user function is silently discarded when a builtin shares its name — **FIXED**

> **Fixed** in `src/lower/mod.rs`: a name the source defines now routes to the
> general user-function path instead of the builtin. Verified — `def band(a,b)
> { 999 }` called as `band(12,10)` returns 999, where it previously returned 8.
>
> **The obvious implementation is wrong.** `fn_sigs` looks like "user functions"
> and is not: it is pre-populated with builtin return types (`println`, `print`,
> `sleep_ms`, …) so call sites get concrete types. Using it as the shadow test
> makes every program that prints fail to compile. The check needs the set of
> names the *source* declares, published through a thread-local alongside the
> existing bring-prefix context.
>
> ### What the fix uncovered
>
> **Thirteen stdlib functions were being shadowed too**, so their IRIS
> implementations were dead code: `contains`, `cross_entropy`, `http_get`,
> `http_post`, `http_post_json`, `http_request`, `is_empty`, `len`, `list_mean`,
> `list_std`, `max`, `min`.
>
> At least one pair genuinely disagrees:
>
> | `list_std([2,4,6,8])` | |
> |---|---|
> | builtin | 2.2360 — population, ÷n |
> | `std.ais` | 2.5820 — sample, ÷n−1 |
>
> Two implementations under one name, different answers, and the winner decided
> by an unwritten rule. `tests/conformance/c14` was asserting the builtin's value
> without knowing it.
>
> **Follow-up, not yet done:** audit the other twelve pairs for the same
> divergence and delete or rename the loser in each. Two implementations of
> `min` is a defect whichever one wins.
>
> The original report follows.

### Original report

```iris
def band(a: i64, b: i64) -> i64 { 999 }

def main() -> i64 {
    println(to_str(band(12, 10)));   // prints 8, not 999
    0
}
```

The builtin wins. No error, no warning, no diagnostic of any kind — the
user's function is simply never called.

**203 builtin names** are resolved in the lowerer and are effectively reserved,
but none of them is a keyword and none is rejected at the definition site.
Among them are names an ordinary program is likely to choose:

```
abs band bor contains filter get hash len map max min
pop push repeat round set sign split trim
```

If the arity differs the failure is at least loud, though the message is
misleading — defining `def band(n: i64) -> str` and calling `band(0)` reports
*"band() requires exactly 2 arguments"*, describing the builtin the programmer
never intended to call.

This was found by writing a `.iris` program that happened to name a function
`band`. Any program that names a function after one of the other 202 has the
same silent substitution waiting in it.

**Fix direction:** reject a user definition that shadows a builtin at the
definition site (clear, breaks nothing silently), or let the user definition
win with a warning. Either is acceptable; silence is not.

**Status:** open, and the highest-severity item on this list.

---

## 16. A `handle` expression as a function's return value — **FIXED**

> The block parser routed `handle` to `parse_handle_stmt` unconditionally and
> pushed it into `stmts`, so it could never be a block's tail. A function whose
> body was a `handle` computed the value, released it, and returned nothing:
>
> ```
> def run() -> str {
>     %1 = call_extern @echo(%0)
>     pop_handler
>     release %1, Str
>     return              <- declared `-> str`
> }
> ```
>
> Callers then read an undefined value. The comment three lines above the guilty
> branch already explained that `with <effects> { }` deliberately falls through
> to the expression parser "so it works as a tail expression (returns a value)
> like `if`/`when`/`block`" -- `handle` now does the same. A `handle` followed by
> `;` is still a statement by the ordinary expression-statement rule, which needs
> no special case.
>
> Verified on both backends; all seven effect-handler tests pass.

### Original report

```iris
extern def echo(s: str) -> str

def intercepted() -> str {
    handle { echo("original") } with { echo(s) -> resume(v) => v("intercepted") }
}

def main() -> i64 {
    println(intercepted());
    0
}
```

```
error[E0401]: [runtime error] internal error: undefined value %0
              — this is a compiler bug, please report it
```

Natively the same program fails LLVM verification with
`use of undefined value '%v0'`.

The handler is fine *inline* — this works:

```iris
val result = handle { echo("original") } with { echo(s) -> resume(v) => v("x") }
```

which is the shape `tests/test_resume_handler.iris` uses, and why the defect
survived: the only handler test in the tree never returns one from a function.
The value produced by the handle block is not materialised when the block is a
function's tail expression.

Algebraic effects with `resume` are one of the more distinctive things IRIS
offers, so this is worth fixing before the capability is described anywhere.

**Status:** open.

---

## 17. Labelled `continue` fails in the interpreter — **FIXED (and it was not a backend divergence)**

> The original diagnosis was wrong in a way worth recording. This was filed as
> "works natively, fails interpreted". It is neither: **the compiler was
> non-deterministic**, and the same source compiled to different IR on different
> runs. Six runs of one file produced **three distinct IR outputs, three of
> which were invalid**, so the program succeeded or failed about half the time
> on *either* backend. Whichever backend was run first looked broken.
>
> **Three separate defects, all fixed.**
>
> **(a) LICM merged nothing across back edges.** `continue outer` gives the
> outer header two back edges — one from the inner loop's exit, one from the
> continue block. Each was treated as its own natural loop, so each body
> excluded the other's latch, and that latch then looked like a block *outside*
> the loop and became a candidate preheader. Back edges sharing a header are now
> unioned into one loop, per the textbook definition.
>
> **(b) LICM did not require the preheader to dominate the header.** It took any
> predecessor outside the body, chosen with `.iter().find()` over a `HashSet` —
> non-deterministic, and unsound. It now requires the candidate to dominate the
> header (which makes it dominate every block in the loop, hence every use it
> hoists past) and takes the lowest index of those, which is also deterministic.
>
> **(c) `LoopUnrollPass` left the original body block referring to deleted
> definitions.** It cleared the loop header's block params and instructions but
> stubbed only the header, not the body — and the body reads the induction
> variable, which arrives as a header param. Both blocks are unreachable after
> rewiring, so both backends kept producing correct answers over invalid IR.
> Both are stubbed now.
>
> **Two more sources of non-determinism, fixed alongside.** Closure captures,
> `spawn` captures and `par for` captures were all collected by iterating
> `Lowerer.scope`, a `HashMap`. Order was self-consistent within a run — the
> lifted parameter list and the call's argument list come from the same vector —
> so programs were correct, but no two builds agreed. All three are sorted by
> name now.
>
> **The guard that should have caught this could not.** `verify_uses_defined`
> runs after every pass, but checked only `Br`/`CondBr` arguments — and the
> broken use was `%18 = add %3, %17`. It now checks every operand via
> `instr.operands()`. Turning it on immediately found (c), which had been
> compiling invalid IR silently. Across all 125 `.iris` files, zero invalid IR
> remains.
>
> This is the third instance of the pattern in `language-hardening-plan.md`: **a
> guard that existed and could not fire.** It is also the third defect that
> produced correct answers on every run that happened to work.
>
> Regression test: `tests/conformance/c19_labelled_flow.iris`, whose expected
> values are weighted so that a wrong *set* of iterations fails even when the
> iteration count is right. Determinism itself is checked by compiling each
> conformance file five times and comparing IR hashes.

### Original report

```iris
def main() -> i64 {
    var evens = 0
    for scan k in 0..10 {
        if k % 2 == 1 { continue scan; };
        evens = evens + 1
    };
    assert(evens == 5);
    0
}
```

| | Interpreter | Native |
|---|---|---|
| labelled `break` | works | works |
| labelled `continue` | **`undefined value %8`** | works |

The tree's own `tests/test_labeled_break.iris` reproduces it — it is one of the
only three `.iris` files that assert anything, and it fails on one backend. No
Rust test globs `tests/*.iris`, so nothing has ever run it.

**Two consequences worth stating.**

*The capability is currently over-claimed.* `iris-claims` lists "Labelled
break/continue" under "Safe to claim (verified this session)". That verification
was native-only. Labelled `break` is safe to claim; labelled `continue` is not,
until this is fixed.

*A backend divergence is a distinct failure class.* Everything else on this list
is wrong or absent on both paths. Here the two backends disagree, so testing
either one alone reports success. Conformance programs must run under both
`--emit eval` and `IRIS_FORCE_INTERP=1`; this was found on the first file that
did.

**Status:** open.

---

## 18. `dyn Trait` coercion only at an annotated binding — **FIXED**

> A value became a trait object only where a binding said so:
>
> ```iris
> val sc: dyn Speaker = c            // worked
> describe(c)                        // runtime: "DynCall on non-trait-object"
> list_push(zoo, c)                  // compile: "type mismatch: dyn Speaker vs Struct"
> ```
>
> The first failure is the worse one: a mistake knowable at compile time was
> deferred to execution, so an untested branch shipped broken. The second ruled
> out heterogeneous collections, which is the main reason to want trait objects
> at all -- the feature was close to decorative without them.
>
> **The mechanism was already present.** `coerce_to_trait_object` existed as a
> general helper and did exactly the right thing; it simply was not called at
> these sites, and both sites discarded the type information they needed:
>
> ```rust
> let (v, _) = self.lower_expr(arg)?;          // call argument
> let (list, _) = self.lower_expr(&args[0])?;  // list_push
> let (value, _) = self.lower_expr(&args[1])?;
> ```
>
> **Fixed** by capturing those types and calling the existing helper. Call sites
> need the callee's declared parameter types, so `fn_param_types` is built in
> the same loop as `fn_param_names` — from the **AST**, not read back from
> `IrModule`, because a callee defined later in the file has not been lowered
> when its caller is.
>
> No second site was needed for `list_get`: the element type survives, so
> `list_get(zoo, 0).speak()` dispatches through the vtable correctly.
>
> `tests/conformance/c08_trait_objects.iris` now covers all three coercion
> sites, a three-type heterogeneous collection, per-element dispatch in a loop
> (the failure a vtable bug would most plausibly produce is applying one impl to
> everything), and a trait object drawn back out of the list and passed on.

### Original report

A concrete value is coerced to a trait object at an annotated `val`, and nowhere
else.

```iris
record Cat { name: str }
trait Speaker { def speak(self) -> str }
impl Speaker for Cat { def speak(self) -> str { "meow" } }

def describe(s: dyn Speaker) -> str { s.speak() }

def main() -> i64 {
    val c = Cat { name: "tom" }

    val b: dyn Speaker = c        // works — the only path that does
    assert(b.speak() == "meow");

    assert(describe(c) == "meow");    // compiles, then at runtime:
    //   type error — DynCall on non-trait-object: Struct([Str("tom")])

    val xs: list<dyn Speaker> = list()
    push(xs, c);                      // compile error:
    //   type mismatch: dyn Speaker {...} vs %Cat -- ListPush
    0
}
```

| Form | Result |
|---|---|
| `val d: dyn Trait = concrete` | works |
| passing a concrete value to a `dyn Trait` **parameter** | compiles, fails at runtime |
| pushing a concrete value into `list<dyn Trait>` | fails at compile time |

The fat pointer is materialised by a `MakeTraitObject` coercion attached to
annotated let-bindings (`src/lower/mod.rs`, "Cohersion: if the binding is
annotated `dyn Trait`..."). Argument positions and collection elements have no
equivalent, so the concrete struct is passed through raw and `DynCall` finds no
vtable.

**A heterogeneous collection is the primary reason trait objects exist.** Until
`list<dyn Trait>` accepts concrete elements, `dyn` is close to unusable for the
thing it is for.

`tests/test_trait_object.iris` passes because it only ever uses the annotated
binding form, and it asserts nothing — it returns 0/10/20/30/40 as status codes.
Nothing runs it.

**Fix direction:** apply the same coercion wherever a `dyn Trait` type is
expected — argument positions, list/map element types, struct fields, return
positions — rather than only at let-bindings.

### 18b. …and there is no native implementation at all

Worse than the coercion gap: `dyn Trait` does not work in a native build in
*any* form, including the annotated-binding one that works interpreted.

```
use of undefined value '%v1'
  call void @iris_retain_kind(ptr %v1, i32 1)
```

The emitted IR shows the values simply missing:

```llvm
%v1 = getelementptr inbounds %Cat, ptr %struct_alloc1, i32 0   ; the Cat
call void @iris_retain_kind(ptr %v3, i32 1)                    ; %v3 undefined
%v5 = call i1 @iris_str_eq(ptr %v3, ptr %v4)                   ; used again
```

`%v2` and `%v3` — the trait object and the dispatched call — are never defined,
because both instructions are empty no-ops in the LLVM backend
(`src/codegen/llvm_ir.rs`):

```rust
IrInstr::MakeTraitObject { .. } => {}
IrInstr::DynCall { .. } => {}
```

Both declare a result (`result()` returns `Some`), so every SSA value they
should produce dangles. This is the partial-update defect the
`iris-compiler-change` skill describes: every required site updated except the
codegen one.

`tests/test_trait_object.iris` and `test_trait_object_return.iris` both fail
natively. Neither is run by anything, and neither asserts.

**Implementing this needs a vtable representation in the LLVM backend** — a
per-impl table of function pointers, a fat pointer `{ data, vtable }`, and
`DynCall` indexing it. That is real work, not a patch, and should be planned.

Until then: **trait objects are interpreter-only, and only via
`val x: dyn Trait = concrete`.** Do not describe `dyn Trait` as a working
feature.

**Status:** open.

---

## 19. Signed division miscompiled by strength reduction — **FIXED**

> **Fixed** in `src/pass/strength_reduce.rs` by removing the rewrite. Verified:
> both backends now agree at `-3`, and the division/modulo identity holds.

`StrengthReducePass` rewrote `x / 2^n` to `x >> n`. That identity holds only for
non-negative dividends. An arithmetic shift right **floors**; IRIS division
**truncates toward zero** (`wrapping_div`, matching the interpreter and the C
runtime):

```
-7 / 2  == -3      (truncate — correct)
-7 >> 1 == -4      (floor    — what the pass emitted)
```

**An optimisation silently changed the answer.** It fired only when the divisor
was a visible constant, so whether it applied depended on *inlining*:

```iris
def small_div(a: i64, b: i64) -> i64 { a / b }          // inlined, then reduced
def big_div(a: i64, b: i64) -> i64 { /* padded */ }     // stays a real call

small_div(0 - 7, 2)   ->  -4
big_div(0 - 7, 2)     ->  -3      DISAGREE
```

Both backends agreed on the wrong answer, so this was a pass defect, not a
backend one — and no single-backend check could have caught it.

It also broke the identity every integer division must satisfy, because `%` was
left truncating:

```
(-7 / 2) * 2 + (-7 % 2)  ==  -4*2 + -1  ==  -9     (should be -7)
```

Recovering the optimisation would need the usual bias correction
(`x + ((x >> 63) >>> (64 - n))` before shifting) or a proof that `x >= 0`.
Neither is worth doing: codegen emits LLVM `sdiv`, and LLVM performs this
reduction itself, correctly, whenever it is sound.

`tests/conformance/c09_numeric_widths.iris` carries the regression guard,
including both directions of the identity.

**How it was found:** an assertion in a conformance file that I wrote wrongly.
`(0-7)/2 == -3` failed, and the disagreement between the constant-folded and
runtime paths turned out to be the compiler, not the test.

---

## 20. Every diagnostic in a CRLF file was misplaced — **FIXED**

> **Fixed** in `src/preprocessor.rs` (byte-faithful line splitting) and
> `src/ir/instr.rs` / `src/lower/mod.rs` / `src/interp/mod.rs` (`Panic` carries
> its own source position). Verified: five separate cases now report the exact
> line *and* column.
>
> **The reported symptom was much narrower than the defect.** It looked like
> "assertion failures are off by one statement". The actual cause:
> `Preprocessor::process` iterated with `source.lines()`, which **strips the
> carriage return** of a CRLF terminator. Every kept line came out one byte
> shorter, the lexer computed spans against that shortened text, and diagnostics
> rendered against the caller's original source — so **every reported position in
> every CRLF file drifted by one byte per preceding line**. On Windows that is
> every file, and the drift grows with file length: a failure 60 lines down
> reports roughly a line early, one 200 lines down about three.
>
> That explains the whole family of "the span points at a comment / two lines
> early" observations recorded while writing the conformance corpus. They were
> not separate bugs.
>
> Two changes, both needed:
>
> 1. `split_inclusive` on the newline instead of `lines()`, preserving the
>    original terminator, so the preprocessor is byte-faithful for every line it
>    passes through.
> 2. `IrInstr::Panic` gained `span_byte: Option<u32>`. `span_table` is keyed by
>    `(block_id, instr_idx)` and **no optimisation pass maintains it**, so
>    const-folding the `ConstStr` holding a panic message shifted every later
>    index in the block and orphaned the entry. Carrying the position on the
>    instruction makes it immune to instruction motion.
>
> The general `span_table` staleness under optimisation remains — see the note
> below. Only `Panic` is currently immune.

### Original report

## 20a. Assertion failures point at the *previous* statement

```iris
def main() -> i64 {
    val a = 1
    val b = 2
    val c = 3
    assert(a == 1);
    assert(b == 2);
    assert(c == 99);   // line 7 — this is what fails
    0
}
```

```
error[E0406]: [runtime error] program panicked: assertion failed
 --> 6:20
   |
6 |     assert(b == 2);      <-- reported, and it SUCCEEDED
```

The span is off by one statement, so the diagnostic names a line that passed.
That is worse than no location: it sends the reader to correct working code.

Reproduced three times while writing the conformance corpus — once pointing at a
**comment** (`c07`), once two statements early (`c10`), and in the controlled
case above. It cost real time on each.

Assertions are the primary failure signal in every `.iris` test, so this is the
single most-encountered diagnostic in the language. Worth fixing before the
corpus grows to 125 asserting files, because every one of them will report the
wrong line when it breaks.

**Status:** fixed — see above.

---

## 20b. `span_table` is invalidated by every optimisation pass — **open**

Uncovered while fixing #20 and worth recording separately, because `Panic` is
now immune but nothing else is.

`IrFunction::span_table` is keyed by `(block_id, instr_idx)`. Every pass that
inserts or removes an instruction shifts those indices, and **no pass maintains
the table**:

| Pass | References `span_table` |
|---|---|
| `const_fold` | none |
| `opt` (DCE, CSE) | none |
| `strength_reduce` | none |
| `inline` | none |
| `copy_prop` | none |

So any diagnostic or debugger position derived from the table is unreliable in an
optimised function — it silently names whichever instruction happens to occupy
that index afterwards. This affects the DAP debugger's step/breakpoint mapping
(`src/debugger.rs`, `TraceEntry`) as well as runtime error locations.

**Fix directions:** carry the position on the instruction (as `Panic` now does)
for anything that reports a location; or key the table by result `ValueId`, which
is stable across index shifts; or remap in each pass, which is the most fragile
since every future pass must remember.

**Status:** open.

---

## 21. A generic instantiated at a generic type — **partially fixed, still non-deterministic**

> `wrap(wrap(5))` failed with `type mismatch: %Box__Box__i64 vs %Box__Box__i64`
> -- a message naming one type twice, because `try_unify` compared `IrType`
> structurally and two `Struct` values sharing a mangled name can differ in their
> `fields`.
>
> **Two fixes landed.** Nominal types now unify by name (`same_nominal_type`),
> since a monomorphised name already encodes its type arguments; and the
> diagnostic now says "two different types both named X" with the structural
> difference, instead of printing the same string twice.
>
> Two ordering sources were also made deterministic: the four-`HashMap` suffix
> scan in name resolution returned whichever key hashing happened to yield first
> (now: collect all, longest then lexicographic -- longest because a longer
> mangled name is the more specific instantiation), and generic struct templates
> were registered in `HashMap` order.
>
> **Still open.** The case remains non-deterministic: measured 6 of 12 before,
> 8 of 12 after, so at least one more ordering source exists. Same class as #17.
> `--emit ir` fails consistently while `--emit eval` succeeds about two thirds of
> the time, which is itself a clue -- the two paths do not run identical
> pipelines.

### Original report

`Box<Box<i64>>` does not monomorphise correctly.

```iris
record Box[T] { item: T }
def box_of[T](v: T) -> Box<T> { Box { item: v } }

def main() -> i64 {
    val inner: Box<i64> = box_of(3)
    val outer: Box<Box<i64>> = box_of(inner)
    val mid = outer.item
    assert(mid.item == 3);
    0
}
```

```
type error in function 'box_of__Box__i64'
  — type mismatch: %Box__Box__i64 vs %Box__Box__i64
```

Note the two sides are **the same name**. Two structurally different
`IrType::Struct` values carry that name — one with `item: Box__i64`, one whose
`item` is still the unsubstituted parameter — and the comparison is structural
while the message prints only the name. The inner type argument is not being
concretised, so `resolve_concrete_field` is not recursing through a type
argument that is itself generic.

Going through a generic *function* instead of direct field access gives the
other half of the same defect:

```
expected '%T' but found 'i64'      // unbox(unbox(outer))
```

Single-level generics are fine, at any number of distinct instantiations —
`Box<i64>`, `Box<str>`, `Box<Config>`, `Box<list<i64>>` all work (see
`tests/conformance/c12`). Only generic-in-generic fails.

**Two consequences.** Container-of-container is ordinary — a `Box<list<T>>` is
fine but a `Pair<Box<T>>` is not — so this bites as soon as abstractions are
composed. And it blocks the higher-kinded types the project already claims,
since `Wrapper<Box, i64>` is exactly this shape.

**Secondary finding (DX):** a type-mismatch diagnostic that prints identical
text on both sides of "vs" is unactionable. When two `IrType`s differ
structurally but share a name, the message should show the differing fields.

**Status:** open.

---

## 22. Generic instantiated at a container type breaks natively — **open**

`Box<list<i64>>` works interpreted and fails to build.

```iris
record Box[T] { item: T }
def box_of[T](v: T) -> Box<T> { Box { item: v } }
def unbox[T](b: Box<T>) -> T { b.item }

def main() -> i64 {
    val bl: Box<list<i64>> = box_of(list())
    push(unbox(bl), 5);
    assert(list_len(unbox(bl)) == 1);
    0
}
```

```
error: base element of getelementptr must be sized
  %fgep1_0 = getelementptr inbounds %Box__list, ptr %b, i32 0, i32 0
```

The emitted module *declares*:

```llvm
%Box          = type { ptr }
%Box__T       = type { ptr }
%Box__list_i64 = type { ptr }
```

but the GEP references **`%Box__list`** — the name truncated at the underscore.
LLVM therefore treats it as an undeclared opaque type, which has no size.

The mangled name for a container type argument contains an underscore
(`list<i64>` → `list_i64`), and something on the use-side path splits or rebuilds
the name on `_` and keeps only the first segment. Type arguments whose mangling
is a single token (`i64`, `str`, a plain record) are unaffected, which is why
every other instantiation in `tests/conformance/c12` is fine.

Related to #21 but a distinct mechanism: #21 is the type argument not being
substituted, this is the substituted name being mangled inconsistently between
declaration and use.

**Status:** open.

---

## 23. FFI out-parameters — **IMPLEMENTED**

> Not previously numbered, but recorded across several entries as *the* reason
> ROS 2 subscriptions could not read a payload.

A large fraction of C APIs return their result through a pointer argument:

```c
int64_t iris_rcl_take_twist(int64_t sub, double* out);      /* writes 6 doubles */
int64_t iris_rcl_take_string(int64_t sub, char* buf, int32_t max);
```

None was callable from IRIS. `ros2_bridge.c` has implemented
`take_float64` / `take_int64_val` / `take_string` / `take_vector3` /
`take_twist` / `take_pose` since the bridge was written, and every one was
unreachable — a subscriber could observe only `wait_for_message`'s bool.

**The gap was never the calling convention.** `ffi_dispatch_i64` already passes
an array of `int64_t` slots, and on every supported target a pointer fits in
one. What was missing was a way for IRIS to *own* memory and name its address.

Added: `ffi_out_new` / `ffi_out_free`, typed indexed readers
(`ffi_out_get_f64` / `_i64` / `_str`) and setters for in-out parameters, in the
runtime, the lowerer, codegen and the interpreter, plus `std.ffi` wrappers.
Indexed access matters because one out-pointer often receives several values —
a `Twist` is six doubles.

Each cell carries a 16-byte header holding a magic word and the payload length.
That gives bounds-checked reads (an out-param the callee never filled reads as
zero, not garbage) and an idempotent `out_free`. Idempotence is not a nicety:
the error path of a foreign call typically frees on the way out, and the first
implementation corrupted the heap (`exit 0xC0000409`) on a double free while the
interpreter silently tolerated it. The two backends must agree, so both now do.

`src/stdlib/ros2.iris` gains `take_f64`, `take_i64`, `take_str`,
`take_vector3`, `take_twist` and `take_pose`, each returning `option` so that
"no message" and "a message whose value is zero" stay distinguishable.

**Not yet verified against a live ROS 2 installation** — that needs one
present. What is verified on both backends: the cell round trip
(`tests/conformance/c13`), and that every `take_*` returns `none()` safely when
the middleware is absent, which is the error path a real deployment hits when
it goes down.

Still open for ROS 2: the publish path scales f64 by `1e8` into an i64
fixed-point trampoline, and tf2, QoS, executors, services, actions and
lifecycle nodes remain absent.

---

## 24. `par for` was a data race by construction — **FIXED**

`iris_list_push` grew the buffer with `xrealloc` and advanced `len++` with no
synchronisation at all, so two threads could lose an update or one could move
the buffer while another wrote through the stale pointer:

```iris
val shared: list<i64> = list()
par for i in 0..2000 { push(shared, i); }
```

**It produced the right answer on all eleven runs of a probe.** That is the most
dangerous possible result, and two accidents were hiding it: every push passes
through `iris_retain`, which takes a *global* refcount mutex and serialises most
of the window; and `iris_par_for` created **one OS thread per iteration**, so on
a 2-core box thread startup dominated and iterations barely overlapped. Both
disappear on a larger machine.

Three changes, at three levels:

**Runtime — the collection primitives are locked.** `list_push`, `list_get`,
`list_set`, `list_pop`, `map_set`, `map_remove` now take a global collection
mutex. `list_get` is locked too: an unsynchronised read can observe `data`
mid-realloc and follow a freed pointer. A single global lock rather than
per-list because the refcount mutex is already taken on every element
operation, so the serialisation exists regardless; per-list is the upgrade path
if profiling ever shows it matters. The mutex is non-recursive, so
`iris_map_set` stringifies its key *before* taking it — `iris_value_to_str`
walks a list value and would otherwise re-enter `iris_list_get` and deadlock.

**Runtime — `par for` is a worker pool.** `min(hardware_threads, n)` workers
striding the iteration space, instead of a thread per index. `par for i in
0..2000` created two thousand OS threads; at ~1 MB of reserved stack each that
is an address-space problem before it is a performance one. Striding rather
than contiguous blocks keeps the split even when body cost varies with the
index.

**Language — mutating a captured collection is now a compile error.**

```
error[E0108]: `par for` body mutates the captured collection `shared` via `push`.
Iterations run concurrently, so the result depends on thread scheduling.
Use `atomic` for a shared counter, or build a per-iteration value and combine
after the loop.
```

Locking makes the race survivable; it does not make it *correct*, because the
result is still order-dependent. A language whose selling point is a statically
verifiable autonomy layer should not admit a race through the front door, and
`IrInstr::ParFor` already enumerates its captures, so the check costs nothing.

A new `LowerError::Rejected` variant carries it. `Unsupported` was wrong: its
message promises the construct may arrive later, and a concurrency violation is
not a missing feature.

**Limit, stated plainly:** the check catches a mutating builtin applied directly
to a captured name. Mutation reached through a user-defined function is not
detected — that needs interprocedural analysis. False negatives, never false
positives.

**Still open:** `list_sort`, `list_concat`, `list_slice`, `map_keys` and
`map_values` are not yet locked. They are read-mostly and not reachable from the
rejected pattern, but they are not proven safe either.

---

## 25. Deep recursion crashed the interpreter instead of erroring — **guard FIXED, cause open**

```iris
def sum_to(n: i64, acc: i64) -> i64 { if n == 0 { acc } else { sum_to(n - 1, acc + n) } }
def main() -> i64 { println(to_str(sum_to(400, 0))); 0 }
```

```
thread 'iris-compile' has overflowed its stack
```

A hard process abort, not a diagnostic. Native builds the same program and runs
it fine — another backend divergence, and this one kills the process.

**The guard existed and could never fire.** `InterpOptions.max_depth` and the
"call depth exceeded" error are both present, but:

| | Was | Now |
|---|---|---|
| eval path | hardcoded `max_depth: 5_000` | honours the caller |
| `compile_ast_inner` | took `_max_depth` (discarded) | threaded through |
| CLI default | 500 | 250 |
| Real limit | **~350 frames** | unchanged |
| Error hint | *"use `--max-steps`"* | `--max-depth` |

Every layer was set above the depth the stack can actually take, so the process
died before the check was reached. That is the second guard found dead by
construction in this codebase — the effect-subsumption check (#20's neighbour)
was the first. A guard nobody has watched fire is a guard that does not work.

`--max-depth` was also accepted by the CLI and thrown away, and the error it
raises named the wrong flag.

**Cause still open.** The interpreter consumes a Rust stack frame per IRIS call
frame — 64 MiB / ~350 ≈ **190 KB per call**, which is enormous. `Interpreter::new`
is invoked recursively for each call rather than pushing onto an explicit stack.
Until that changes, the depth limit is a property of the host stack rather than
of the language, and native and interpreted programs disagree about which
programs are valid.

**Fix direction:** an explicit heap-allocated frame stack in the interpreter, so
IRIS recursion depth is bounded by memory rather than by the Rust stack. Also
worth trimming the frame: 190 KB suggests large values are being copied per
call.

**Status:** guard fixed and verified; underlying frame cost open.

---

## 26. Native struct/tuple equality returns `true` for unequal values — **FIXED for records; tuples now rejected**

> **Root cause.** Records are `ptr` in LLVM, and the comparison path treated
> "both operands are `ptr`" as evidence of a *string*, so `==` on two records
> called `iris_str_eq`, which `strcmp`s the raw struct bytes.
> `P { x: 1, y: 2 }` and `P { x: 1, y: 3 }` both read as the one-byte string
> `""` — the first field followed by its zero padding — and compared
> **equal**. Not unreliable: the exact opposite of the truth, silently.
>
> It also explains why the first check tried during #8 appeared to pass. It did
> not work; it returned `true` for everything whose first field matched.
>
> **Fix.** The `ptr`-means-string heuristic now excludes aggregates, and records
> compare field by field: `icmp`/`fcmp` for scalars, `iris_str_eq` for `str`
> fields, and recursion for nested records. Pointer identity would have been
> just as wrong in the other direction — two separately built records with
> identical fields would compare unequal, while the interpreter compares by
> value.
>
> With native correct, the interpreter's `Struct` arms were enabled, so both
> backends now agree.
>
> **Tuples are not fixed.** They have no named LLVM type to index, so there is
> no structural comparison to emit; native rejects `==` on tuples at compile
> time and the interpreter still rejects it at runtime. Both refuse, which is
> the honest state — a field type with no comparison is refused explicitly
> rather than guessed at, because a wrong answer here is invisible.
>
> Regression test: `tests/conformance/c20_record_equality.iris`, whose cases are
> chosen to fail a `strcmp` implementation — records differing only in a
> trailing field, and records sharing a leading field.

### Original report

```iris
record P { x: i64, y: i64 }
P { x: 1, y: 2 } == P { x: 1, y: 3 }
```

| | Result |
|---|---|
| native | **`true`** |
| interpreter | `false` (via `PartialEq`, but the binop arm is not wired — see below) |

Native reports two structs with different contents as equal, and correspondingly
`!=` as `false`. A silent wrong answer, not a crash.

Found while fixing #8. Enum equality was added to the interpreter and verified
identical on both backends; struct and tuple equality was **deliberately left
out**, because `IrValue` already implements `PartialEq` for them and wiring the
arm would have made the interpreter structurally correct while native stayed
wrong — replacing one wrong answer with a backend disagreement, which is worse
to debug.

**Fix both together:** native needs a real field-wise comparison (or a runtime
helper) for aggregates, and the interpreter needs the two binop arms. The
interpreter side is two lines and is marked in place with a pointer here.

**Status:** open.

---

## 27. Loop-carried i64 accumulator unifies with an f64 body — **open**

Reproduction: `tests/conformance/c21_loop_carried_type.iris` (checked in, fails).

```
error[E0202]: type mismatch: i64 vs f64 -- Br arg %N to param %M
```

An `i64` accumulator carried across a loop whose body compares two `f64` values
has its block-parameter type unified with the comparison. Deterministic -- 10 of
10 compiles fail identically, on both backends.

It needs the whole combination: dropping the asserts from the first loop, or the
`random()` call, or the trailing loop, each make it pass. This is the
type-inference-after-lowering weakness in `docs/architecture-vs-rustc.md` --
inference runs on the CFG, after the source shape is gone, so a loop-carried
accumulator has nothing anchoring its type except the values flowing near it.

**Workaround:** keep the accumulator's arithmetic in a separate function, or
avoid mixing an integer accumulator with float comparisons in one loop body.

**Status:** open.

---

## 28. A `str`/`i64` mismatch is reported as `option<i64>` vs `i64`, at a blank line — **open, diagnostics**

Writing `val rc = shell(cmd); if rc == 0` -- where `shell` returns `str` -- gives

```
error[E0101]: type mismatch — expected 'option<i64>' but found 'i64'
 --> 3:1
   |
 3 |
   | ^
```

Neither type in the message is a type in the program, and the span points at a
blank line in a *different file* from the error. The same misreporting appeared
for `find`, which returns `option<i64>`: the message named the option but the
caret landed on an unrelated line, and moving unrelated statements moved the
reported location.

Cost is real: two genuine one-line mistakes in `std.serial` (assuming `shell`
returned an exit code, and assuming `find` used a `-1` sentinel) each took
several bisection rounds to locate, because the diagnostic pointed away from
both. Spans are invalidated by the optimisation passes (#20b), which is likely
the same root cause.

**Status:** open. Fixing #20b probably fixes the span half; the *type* half
needs the message to name the types the user actually wrote.

---

## 29. `eval_rule` silently ignored four of six operators — **FIXED**

> `AutonomicRule` carries an `operator: str`, and `eval_rule` understood only
> `">"` and `"<"`. Every other operator fell through to `false`, for every
> input. A rule written with `">="` was therefore not a strict rule -- it was a
> rule that **never fired**, while still appearing in the policy, counting
> toward the rule set, and looking correct in review.
>
> This is the fourth instance of the project's most expensive pattern: a guard
> that exists and cannot fire (after effect subsumption, the recursion limit,
> and use-before-def). It is also the most dangerous instance, because the
> others were compiler internals and this one is a user-facing policy
> primitive -- in an autonomic system, a safety rule that quietly never fires is
> worse than one that fails loudly.
>
> **Fixed:** all six comparisons (`>` `<` `>=` `<=` `==` `!=`) are handled, and
> an unrecognised operator now **panics** rather than returning `false`. A typo
> in a policy is a programming error, and the safe direction for a programming
> error is loud.
>
> Asserted on both sides of every boundary in `tests/test_ais_primitives.iris`,
> since off-by-one at a threshold is the entire difference between `>=` and `>`.

---

## 30. `adaptive_uncertainty_bayes_update` was not a Bayesian update — **FIXED**

The C implementation computed a posterior from the accumulated error
statistics and then returned `posterior_mean + observation`. With no
accumulated data that is `prior_mean + observation`, where a conjugate update
with no data must return the **prior mean**; the `observation` argument was
never folded into the update at all, and the result grew without bound as
observations arrived.

This value is what an adaptive system uses to decide whether it is confident
enough to act, so a posterior that is not a posterior is a live hazard rather
than a cosmetic error.

**Fixed:** proper normal-normal conjugate update treating `observation` as one
new datum on top of the accumulated statistics. Asserted in
`tests/test_adaptive.iris` against the defining property the old version
violated -- the posterior must lie **between** the prior mean and the
observation -- plus the confident-prior and uncertain-prior limits.

---

## 31. Effect handler parameters were untyped — **FIXED**

> **Root cause.** `lower_handler_arm` looked up the extern signature and built
> `lifted_params` with the correct types -- and then added the entry block's
> params as `IrType::Infer` anyway, discarding what it had just computed.
> Inference defaulted them to `i64`, so a handler that *used* its parameter
> received a `str` bound as an integer.
>
> Two symptoms, one cause:
>
> | | before | after |
> |---|---|---|
> | `concat(s, p)` in a handler, native | **invalid LLVM IR** -- `call ptr @iris_str_concat(ptr %v2, ptr %p)` with `%p` defined `i64` | works |
> | `concat(s, p)` in a handler, interpreted | worked | works |
> | `"s:" + p` in a handler | **silently evaluated to `0`** | compile error (see below) |
>
> **`test_resume_handler.iris` passed on both backends throughout**, because its
> handler ignores its argument. Every handler in the tree did. So the algebraic
> effect system -- a headline feature -- could not use handler arguments
> natively at all, and nothing said so.
>
> A second fix went in alongside: `ResumeCont` emitted `trunc i64 %x to i64`
> when the handler result was already `i64`, which LLVM rejects as an invalid
> cast. Only reachable once a handler used its parameter.
>
> The first attempt at this fix indexed `lifted_params` by the arm-parameter
> position, but the continuation is `insert`ed at index 0, so every first
> parameter received the continuation's `weak_ref<_>` -- which is where the
> `expected 'str' but found 'weak_ref<_>'` message came from. Reading the types
> straight from the extern signature removes the offset entirely. `+` and
> `concat` both work in handler bodies now, on both backends, and
> `tests/test_effect_system.iris` asserts both.

### Original report

```iris
extern def rf(path: str) -> str
val c = handle { rf("cfg") } with { rf(p) -> resume(v) => v("s:" + p) };
// interpreter: c is I64(0), then `c == "s:cfg"` fails as CmpEq on I64 and Str
// native:      "arithmetic (Add) on a heap-represented value is not supported"
```

`concat("s:", p)` works, and `"s:" + p` works fine **outside** a handler. Handler
parameters carry no declared type, so they default to `i64` and `+` lowers as
integer addition rather than string concatenation.

Another instance of the type-inference-after-lowering weakness in
`docs/architecture-vs-rustc.md`: `p` is a handler binding with nothing to anchor
its type to by the time inference runs.

**Workaround:** use `concat` inside handler bodies.

**Status:** open. The fix is to give handler parameters their declared types from
the `effect` declaration at lowering time.

---

## 32. Unresolved externs returned a typed zero instead of failing — **FIXED**

`dispatch_extern` ended with:

```rust
// Return a zero value of the declared return type so tests can verify the call happened.
```

A test that accepts a fabricated answer verifies nothing. The whole of
`std.adaptive` -- 36 functions over `iris_adaptive_*` -- ran this way in the
interpreter: `adaptive_new` returned handle `0`, `adaptive_name` an empty
string, `adaptive_n_params` `0`, with no diagnostic, while the identical program
was correct natively.

Worse than an ordinary silent default, because `--emit eval` **falls back to the
interpreter when a native build fails**. That turned "the build broke" into
"every extern returns 0" -- a wrong answer wearing the costume of a successful
run.

**Fixed:** an extern with no builtin implementation and no resolvable dynamic
symbol is now an error naming the symbol. Blast radius across the 132-file
corpus was two files, both of which were depending on the fabricated zero.

---

## 33. Every native FFI call carrying arguments crashed — **FIXED**

`iris_ffi_call_i64(void* handle, const char* name, int64_t* args, int nargs)` --
and its `_f64`, `_str` and `_void` siblings -- take a **pointer to an array** of
arguments plus a count. Codegen passed the arguments flat:

```llvm
declare i64 @iris_ffi_call_i64(ptr, ptr, ptr, i32)
call   i64 @iris_ffi_call_i64(i64 %h, ptr %name, i64 %pub, i64 %val)
```

so the callee read the first user argument as the `args` pointer and the second
as `nargs`, then dereferenced it. **Any FFI call with arguments crashed
natively** with an access violation, while the interpreter -- which marshals
correctly -- returned the right answer.

**Found by running against a real ROS 2 Humble installation.** `rcl_init`, node
creation and publisher creation take no arguments and all worked; the first
two-argument call, `iris_rcl_publish(handle, value)`, segfaulted. Nothing in the
tree exercised an FFI call with arguments on the native backend, so this had
never been observed.

**Fixed** by allocating an `[N x i64]`, storing each argument into it -- doubles
bitcast, pointers narrowed, bools zero-extended, so nothing is passed in the
wrong register class -- and passing the array pointer with the count.

This is the **third** instance of the same bug class in `emit_instr_ir`, after
`iris_select` (channel count not prepended) and `json_stringify` (value not
boxed). In each case a runtime signature is declared a few hundred lines below
the call site and nothing checks one against the other. **A check that every
emitted call matches its own `declare` would have caught all three**, and is
worth more than fixing them one at a time.

**Verified after the fix:** a native IRIS binary creates a ROS 2 node and
publisher and publishes 20 messages, matching the interpreter.

---

## ROS 2 support — regraded 2026-08-16

Previously graded on inspection alone. With ROS 2 Humble installed and the
bridge built, the following are now **Verified** on both backends:

| | Status |
|---|---|
| `rcl_init` | Verified -- returns 0 against real rcl |
| node creation | Verified -- real node handle |
| publisher creation | Verified -- real publisher handle |
| publish (`std_msgs/Float64`) | Verified -- 20-message burst, both backends |

Still **not** verified: subscriptions and the six `take_*` payload functions, QoS,
services, actions, tf2, lifecycle nodes. The 2/10 rating in
`docs/autonomy-stack-assessment.md` covered the whole stack and remains broadly
right; what changed is that the publish path is now demonstrated rather than
assumed.

**Build note.** The bridge links only against MSVC-built ROS 2 binaries. ROS 2's
Windows headers put `__attribute__((dllimport))` on enums under `__GNUC__`, so
the MinGW toolchain cannot compile them; and the MSVC BuildTools install on this
machine has headers but no CRT import libraries, so `memcpy` and `_fltused` were
supplied by a two-function shim. Reproduced in `docs/ros2-build.md`.

---

## 34. `std.adaptive` reports risk numbers it never measures — **FIXED**

> **Fixed** on 2026-08-20. All three fabrications are gone; every field of
> `RiskMetrics` is now measured in the runtime and read back through its own
> accessor.
>
> The fabrication was in two layers, which is why it survived. The C side never
> tracked a maximum or an error count at all (`m.max_error = 0.0`, and
> `m.errors` was set to `1` by `if (m.last_risk > 0.5)` — a flag, not a count).
> The IRIS side then papered over the gap: `adaptive_get_risk` built its tuple
> as `(err, err, obs, obs, err, 1.0 - 1.0/(1.0 + abs(err)))`, substituting the
> mean for the max, the observation count for the error count, and inventing a
> confidence from the mean. Every number had a plausible value, which is exactly
> what made it dangerous.
>
> `IrisAdaptiveState` now carries `max_err` and `err_count`, both updated in
> `iris_adaptive_record_error_impl` (`err_count` counts observations exceeding
> *this state's* risk threshold), and four new accessors —
> `iris_adaptive_max_error`, `_error_count`, `_last_risk`, `_confidence` — give
> the stdlib real values to return.
>
> **`adaptive_is_unsafe` ignored the threshold entirely**, using a hardcoded
> `0.5`, so `adaptive_set_risk_threshold` had no effect on the one guard the
> module exists to provide — a caller set a threshold, saw a guard, and got a
> guard answering a different question. It reads `state->risk_threshold` now,
> clamped to `(0, 1]` since `last_risk` is itself clamped to 1.0 and a threshold
> above that would make the arm unreachable.
>
> Verified: errors of 1, 3, 2 give mean 2 and max 3 (previously both 2); errors
> of 0.0 and 0.6 give a risk of 0.724, which reads safe at a threshold of 0.9
> and unsafe at 0.05 (previously identical at both).
>
> `tests/test_adaptive.iris` had both defects **pinned as KNOWN-WRONG** so that
> fixing them would fail the test rather than let the fabrication survive
> another release. It did fail, and the assertions now pin the real behaviour —
> including `risk_max_error(m) != risk_mean_error(m)` and `risk_errors(m) !=
> risk_observations(m)`, so the two can no longer be the same value by
> construction.
>
> The guard's second arm (`confidence < 0.3`) is unchanged and pre-existing; the
> test's sample is chosen to keep it quiet so the threshold arm is the only
> thing deciding.

### Original report

Three separate fabrications in the module whose entire purpose is deciding when
an adaptive system is safe to act:

| Reported | Reality |
|---|---|
| `risk_max_error` | The C side sets `m.max_error = 0.0` and never updates it. `adaptive_get_risk` then builds its tuple as `(err, err, obs, obs, err, ...)`, substituting the **mean** for the max. |
| `risk_errors` | Substituted with `observation_count` -- the error count is not tracked separately. |
| `adaptive_is_unsafe` | `return m.last_risk > 0.5 \|\| m.confidence < 0.3;` -- it **ignores `risk_threshold` entirely**. `adaptive_set_risk_threshold` has no effect on it. |

The last is the serious one. A caller sets a threshold, sees a guard, and gets a
guard answering a different question. That is a guard that cannot fire for the
reason its user believes -- the fifth instance of that pattern in this tree,
after effect subsumption, the recursion limit, use-before-def and `eval_rule`.

`tests/test_adaptive.iris` **pins the current behaviour deliberately**
(`risk_max_error(m) == risk_mean_error(m)`, and a tightened threshold changing
nothing) so that fixing the module fails the test and forces the assertions to be
updated, rather than the fabrication surviving another release.

**Found by writing the first test that asserted this module's numbers rather
than watching it not crash** -- and found late, because that test was committed
before it had ever been run. It failed on the very first honest execution.

**Status:** open. Needs `max_error` genuinely tracked in `IrisAdaptiveState`, an
error counter distinct from the observation count, and `is_unsafe` to consult
the threshold the caller set.

---

## 35. There is no tail-call optimisation — **FIXED**

> There is now. `src/pass/tail_call.rs` rewrites a self-call whose result is
> returned into a jump back to a loop header, so the recursion becomes a loop
> before either backend sees it.
>
> Done as an **IR pass** rather than per-backend, so the interpreter and native
> codegen get it from one implementation and cannot drift apart — and the
> transformation is visible in `--emit ir` rather than having to be trusted:
>
> ```text
>   entry0(n, acc):       br tail_header(n, acc)
>   tail_header(n, acc):  condbr ...
>   else2():              br tail_header(n - 1, acc + n)    <- was `call @sum_to`
> ```
>
> The entry block cannot be a branch target (LLVM rejects it), so the body moves
> into a fresh header and the entry block becomes a single jump into it.
>
> **Measured:** `sum_to(50000)` runs at the **default** depth limit on both
> backends — it previously needed `--max-depth 100000` and stopped at 250.
> Natively, 1,000,000 recursions complete outright. Interpreted, the same depth
> needs only a raised `--max-steps`, which is a runaway-loop guard rather than a
> stack limit; that distinction is the proof it is constant-stack.
>
> **Mutual recursion is now handled too**, by merging rather than by a returned
> thunk. The classic trampoline needs a tagged return value, which would force
> every mutually-recursive function to change its return type — unacceptable
> when `is_even` must keep returning `i64`. So the SCC is merged into one
> function with a leading selector parameter, and the calls between its members
> become branches:
>
> ```text
>   __tramp$is_even$is_odd(__sel: i64, n: i64) -> i64 {
>     tramp_dispatch00(%0 __sel, %1 n):  condbr %0 == 0, bb1(%1), bb5(%1)
>     is_even$else3():                   br bb5(%10)   <- was `call @is_odd`
>     is_odd$else7():                    br bb1(%19)   <- was `call @is_even`
>   }
>   def is_even(n) { __tramp$is_even$is_odd(0, n) }    <- thin forwarder
> ```
>
> The originals survive as forwarders, so non-tail calls, function values and
> anything referring to them by name keep working; only the depth changes.
>
> **Measured:** `is_even(100000)` and `sum_even(100000, 0) == 5000050000` both
> run interpreted, where they previously stopped at "call depth exceeded 250".
> They already passed *natively* before this — because clang applied its own
> tail-call optimisation to the emitted IR. That is the more interesting half of
> the bug: the two backends disagreed about how deep a program could go, and the
> native pass was luck rather than a property of the language.
>
> **Limits, deliberately:** the members of an SCC must share one parameter list
> (same arity, same types) and one return type — that is what lets a single
> merged signature serve all of them. Where they differ, the merged function
> would need a synthesised dummy for every slot the entering member does not
> supply, and for `str`, records or lists there is no dummy to synthesise; such
> an SCC is declined rather than merged wrongly (asserted by
> `mismatched_signatures_are_declined_but_still_correct`). A call whose result is
> used before being returned — `n * fact(n - 1)` — is not a tail call in the
> first place and is untouched.
>
> **Tested by** `tests/tail_call_elimination.rs` (10 tests, in the Rust suite so
> CI actually runs them) and `tests/test_tco_mutual.iris`.

### Original report

`tests/test_tco.iris` was titled "Test tail-call optimization (TCO) in the
interpreter" and commented "deep recursion that would overflow the stack without
TCO". It demonstrated no such thing.

`sum_to(50000)` runs only with `--max-depth 100000`. It genuinely builds 50,000
interpreter frames; at the default limit it stops with
`call depth exceeded 250`. Nothing is being optimised — the test passed
historically because the depth limit used to be 5,000 and the recursion fitted
under it.

There is no `musttail` in codegen either, so the native backend does not do it
in the compiled path.

`iris-claims` already listed TCO as at risk of overstatement, noting the test
"says in the interpreter". That caveat was too generous: the interpreter does
not do it either.

The test now asserts tail-recursive *arithmetic* at depths inside the default
limit, which is what it can honestly check, and says plainly what it does not
show. A real TCO test would recurse unboundedly in constant stack.

**Status:** open as a feature. Closed as a claim -- do not describe IRIS as
having TCO.

---

## Test-suite audit, 2026-08-16

Of 154 `.iris` files under `tests/`, 33 asserted anything. Converting them
surfaced something the count alone hid: **20 of the 121 non-asserting files do
not run at all.**

| Category | Count | Notes |
|---|---|---|
| Expected failures | 6 | borrow, move, exhaustiveness, strict-effects -- correct behaviour, need an expected-failure list rather than assertions |
| Needs a type annotation (#14) | 5 | `heap_new`, `set_new`, `container_count` -- a type parameter appearing only in the return type |
| Real syntax/feature gaps | 6 | `where` refinement clauses, two module forms, `par_map`, `nursery` arity |
| Other | 3 | no `main`; a genuine `CmpLt on Struct and Str` type bug; and #35 |

Two of these had been broken since they were written. `test_heap_min.iris` and
`test_generic_mod.iris` both called `heap.heap_new(0)` with an argument when the
signature takes none, so neither had ever compiled. Nothing globs `tests/*.iris`,
so a file that could not be parsed sat in the tree indistinguishable from a
passing one — which is the same failure as a test that asserts nothing, one level
further down.

---

## 36. Higher-kinded type parameters could not be inferred from arguments — **FIXED**

> ```iris
> def ident[F[_], T](c: F<T>) -> F<T> { c }
> ident(some_list)   // was: "cannot infer type parameter T -- no argument mentions it"
> ```
>
> **Two causes, both in the monomorphiser.** `extract_from_ast_type` had no
> case for a higher-kinded parameter -- and its caller built the list of
> bindable names with a filter keeping only `AstGenericParam::Type`, dropping
> every `Hkt` entry, so `F` was excluded from the names it was permitted to bind
> at all. Substitution had the mirror problem: rebuilding `F<T>` only knew how
> to form a user record name like `Box__i64`, so a bound `F` naming a builtin
> constructor produced a struct defined nowhere -- the same shape as #14.
>
> A constructor now binds to a nameless-field struct marker (the convention
> already used for unsubstituted type parameters) and `F<T>` rebuilds
> `list<T>`, `option<T>` or `map<K,V>` directly. Decomposition handles list,
> option, map and monomorphised user records.
>
> Verified in `tests/test_hkt_inference.iris`: inference over list and option,
> at `i64` and `str`, with the returned `F<T>` rebuilt correctly and earlier
> instantiations undisturbed. 8 of 8 interpreted, 4 of 4 native.
>
> **Two limits remain, and neither is an HKT defect:**
> - A multi-parameter HKT record (`W[F[_], A]`) works most of the time but is
>   still non-deterministic through the mangled-name ordering in #21.
> - Traits accept no generic parameters at all -- `trait Container[T]` is a
>   parse error exactly like `trait Mappable[F[_]]`. Type classes over a
>   constructor need **generic traits**, which is a separate feature.

### Original report

```iris
def passthrough[F[_], T](c: F<T>) -> F<T> { c }

val lst: list<i64> = list();
passthrough(lst)
// error: cannot infer type parameter `T` for `passthrough` -- no argument
//        mentions it, so annotate the binding
```

The parameter type `F<T>` plainly mentions `T`; the compiler cannot decompose a
concrete `list<i64>` into `F = list, T = i64`. The diagnostic is also wrong
about the cause, and suggests an annotation that cannot help -- `passthrough`
returns `F<T>`, but `container_count` returns `i64`, so there is no binding
whose annotation would resolve anything (#28 again).

**What does work** is declaration plus *explicit* instantiation:

```iris
record Wrapper[F[_], A] { value: F<A> }
val w: Wrapper<Box, i64> = Wrapper { value: Box { item: 100 } }   // fine
```

`tests/test_hkt.iris` uses exactly that form, which is why HKT was graded
"Actually supported -- safe to claim". The grading needs narrowing: **HKT is
supported for declaration and explicit instantiation, not for inference.** A
function generic over a container cannot be called without spelling out its type
arguments, which is most of the reason to want HKT.

`tests/test_box_test.iris` claimed to demonstrate the inference case and had
never compiled. It now asserts the forms that work and documents the gap.

**Status:** open.

---

## 37. Traits accepted no generic parameters — **FIXED**

`trait Container[T]` was a parse error, exactly like `trait Mappable[F[_]]`.
`parse_trait_def` went straight from the trait name to `{`. There were no type
classes of any kind — so the "no type classes over a constructor" limit recorded
when rating HKT was not an HKT gap at all, but a missing feature underneath it.

**Fixed.** Traits parse `[params]` using the same parser functions and impls
already use, so `F[_]` is accepted wherever `T` is. Impls gained the matching
half: `impl Container[i64] for IntBox` parses its trait arguments.

`impl Mappable[list] for ListHolder` needed one special case — `list` binds `F`
to a *constructor*, which is not a complete type, and `parse_type` would demand
`list<...>`. A bare name followed by `,` or `]` is read as a constructor.

Dispatch needed no change: it already keys on the concrete type and method name,
and an impl supplies concrete signatures. The trait's parameters are interface,
not dispatch.

Asserted in `tests/test_generic_traits.iris`, both backends.

---

## 38. An impl target cannot be a generic type — **FIXED**

> ```iris
> impl Sized2 for list<i64>   { def sz(self) -> i64 { list_len(self) } }
> impl Sized2 for option<i64> { def sz(self) -> i64 { if is_some(self) { 1 } else { 0 } } }
> ```
>
> **Three changes, and the third was the hidden one.**
>
> 1. *Parser* — `for` accepted only a bare type name. It now parses type
>    arguments, building `list<T>`, `option<T>` and `map<K,V>` as their own AST
>    forms.
> 2. *Lowering* — dispatch resolved the bare name, so `list` became
>    `IrType::Infer` and matched nothing. It now lowers the whole target, and
>    the mangled name includes the arguments, or two impls of one trait at
>    different element types would collide on one symbol.
> 3. *Method-call lowering* — **the receiver had to be a struct.** An impl on a
>    container was registered in `trait_dispatch` and then never consulted; the
>    call was rejected before reaching it. This is why the first two changes
>    alone produced no visible effect.
>
> `val n: option<i64> = none()` produces `option<_>` — the annotation does not
> reach the `none()` call — so dispatch tries an exact match first and falls
> back to the constructor alone, but only when exactly one candidate matches.
> Impls for `option<i64>` and `option<str>` therefore cannot silently resolve to
> whichever was registered first.
>
> Asserted in `tests/test_impl_on_container.iris`, both backends.
>
> **Generic element types now work too.** `impl[T] Sized2 for list<T>` was being
> caught by the blanket-impl path, which enumerates the concrete types
> satisfying a trait bound — a bound it does not have — so it found none and
> emitted nothing. But a generic *target* is not a blanket impl: the parameter
> belongs to the container, not the impl, and the body is usually indifferent to
> it (`list_len` works for any element). It is now registered **once** against
> `list<_>`, and dispatch treats a type-parameter marker (a struct with no
> fields, the lowerer's existing convention) as a wildcard.
>
> The widening is safe because it is consulted only after an exact match fails,
> and only when exactly one candidate matches — it can rescue a lookup that
> would otherwise fail, never redirect one that would have succeeded.
>
> `src/stdlib/container.iris` is the result: `Sized` and `Countable` across
> `list` and `option`, each impl written once. Asserted in
> `tests/test_container.iris`, both backends.

### Original report

```iris
impl Sized2 for list<i64> { ... }          // parse error: expected '{', found '<'
impl[T] Sized2 for Wrap<T> { ... }         // same
impl Sized2 for Wrap { ... }               // parses, then: undefined function 'Wrap__i64_...'
```

`parse_type_name_str` accepts only a bare type name after `for`, and the third
form parses but is never monomorphised per instantiation.

**This is what blocks a container-generic standard library.** With #36 and #37
in place, a function generic over `F` can *consume* an `F<T>` and *return* the
same `F<T>`, but it cannot construct an `F<B>` — there is no way to give `list`
and `option` a shared interface, because neither can be the target of an impl.
So `map`, `filter` and `fold` still have to be written once per constructor.

Fixing it needs two things: an impl target that may carry type arguments, and
monomorphisation of impl methods per instantiation of the target — the machinery
that already exists for blanket impls, applied to the target rather than to the
impl's own parameters.

**Status:** open. This is the remaining step for "generic functions over
containers" in the useful sense.

---

## 39. Effect clauses are dropped when an impl method is mangled — **open**

`src/stdlib/container.iris` declares `effect alloc` on every method that calls
`list_len`, and on the trait declarations too. Compiling anything that brings it
prints:

```
error[E0302]: function `list_len` requires effect `alloc`
              but caller `container__Sized__list__size` has effects `pure`
```

The clause is present and correct in the source; it does not survive the rename
into a module-prefixed, trait-mangled function, so the checker sees every impl
method as `pure`.

Currently warnings only, so programs still run. **Under `--strict-effects` they
would be build failures**, which makes this a blocker for the one claim the
effect system exists to support: that a control path can be proven to allocate
nothing. Any trait method would fail that proof regardless of what it does.

**Status:** open. The fix belongs wherever impl methods are renamed — the
`renamed` copy carries `params` and `return_ty` through substitution but not
`effects`.

---

## 40. ROS 2 subscriptions returned nothing, or the wrong thing — **FIXED**

Three defects, all found by running against a live ROS 2 Humble install.

**(a) Inverted success convention.** All six `take_*` functions tested
`rc == 0` while the bridge returns **1** on success. A successful take produced
`none()`; a *failed* take produced `some()` wrapping whatever was in the
out-cell — so a subscription could report a reading it never received. These
were the six functions graded *Present, not Verified* earlier in the session.

**(b) Wrong register class.** `iris_rcl_publish_float64` was declared to take a
`double`, but `iris_ffi_call_i64` dispatches through an all-integer function
pointer, so on x86-64 Windows the value was read from XMM1 rather than RDX. The
IRIS side already scaled by 1e8 to work around exactly that, and the C side
never unscaled it. Published `42.5` arrived as `1.00138e-307`.

**(c) The interpreter passed strings as null.** `i64_arg` returns 0 for a
`Str`, so `node_create(ctx, "probe", "")` was called with two null pointers.
Zero-argument calls worked, which made it look like the interpreter could not do
FFI at all rather than that it could not pass strings. Interpreter-side mirror of
#33; both backends now marshal identically.

**Verified** end to end on both backends — publish, wait, take for `Float64`,
`Int64` and `String`, asserting the values. `examples/10_ros2/roundtrip.iris`.

---

## ROS 2 — regraded 2/10 → 4/10, 2026-08-16

| | Status |
|---|---|
| `rcl_init`, node, publisher, subscription | Verified, both backends |
| publish + take: `Float64`, `Int64`, `String` | Verified, values asserted |
| `Vector3`, `Twist`, `Pose` | Present, untested |
| QoS profiles | Absent |
| Services (request/response) | Absent |
| Actions | Absent |
| tf2 transforms | Absent |
| Lifecycle nodes | Absent |
| Executors, callback groups, timers | Absent |
| Parameters | Absent |

**IRIS can now sense and command a topic. It cannot drive a robot.** The gap to
10/10 is a robotics stack, not test coverage — services, actions, tf2 and
lifecycle are each substantial features, and several need more than one node to
verify meaningfully.

---

## 41–44. Five FFI and HTTP defects found by exercising the modules — **FIXED**

Taking `std.ffi` and `std.http` to full coverage meant calling every function
against something real: a purpose-built test library, a live Python, and a local
echo server. That found five defects, none visible from reading the code.

**#41 — `lib_open` returns different sentinels per backend.** A missing library
gives `-1` interpreted and `0` natively. `std.ros2` guarded only on `== -1`, so
natively a failed open passed the guard and every later call failed separately,
far from the cause. Guards now reject both.

**#42 — `std.http`'s client wrappers were unbounded recursion.**

```iris
pub def http_get(url: str) -> str effect net { http_get(url) }
```

A self-call, not a forward. It worked only while a builtin silently beat a
same-named user function; once that was fixed (#15) every call recursed to the
depth limit. The four wrappers added nothing — the builtins are already
available — and are deleted.

**#43 — `http_request` dropped its body unless the method was `POST`.** The
interpreter attached a body only for that exact verb, so `PUT` and `PATCH` sent
a well-formed request with no payload; the server answered 200 and the caller
could not tell. Now keyed on whether there *is* a body.

*Still open natively:* `iris_http_request` delegates to `iris_http_post`, whose
request line hardcodes `POST`, so a native `PUT` arrives as a `POST`. The
interpreter builds its own request line and is correct. The arity mismatch that
crashed it — a four-parameter `declare` called with three arguments, the
**fourth** instance of that class after `iris_select`, `json_stringify` and #33 —
is fixed.

**#44 — native `py_call1` did not quote its argument.** It generated
`basename(/a/b/c.txt)`, a `SyntaxError`. The interpreter quoted it, so the
backends disagreed on every non-numeric argument.

**Also fixed, both in the interpreter's FFI:** a `f64`-returning call read RAX
instead of XMM0 (`call_f64` returned `6.95e-310` instead of `4.5`), and the
`rust_call_*` family was left out of #33's argument-array packing, so
`rust_i64` and `rust_f64` crashed natively. The second is the more instructive:
fixing that class by enumerating names got it wrong twice.

`std.ffi` is now **27/27** covered, `std.http` **12/12**, both backends.

---

## Verified working

Confirmed correct by running and asserting output:

- Labelled `break` / `continue` on `loop`, `for`, `while`
  (`examples/01_basics/labeled_loops.iris`)
- Struct patterns, or-patterns, range patterns, guards
  (`examples/04_types_and_traits/struct_patterns.iris`)
- Extension methods, including chaining and on user records
  (`examples/02_functions/extensions_and_defaults.iris`)
- Default record fields, including partial override (same file)

---

## 45. The LLVM C API emits an object that will not link — **open**

`codegen::llvm_c_api::compile_llvm_ir_to_object` returns `Ok` for
`x86_64-pc-windows-gnu`, but the object it produces is rejected at link time.
This is the last thing standing between IRIS and a clang-free `iris build`.

Making it the preferred path was tried on 2026-08-16 and **reverted the same
day**, because the failure did not surface: `iris build` exited **0 and produced
no binary at all**. Silently building nothing is a far worse failure than
requiring a compiler, so clang stays the default.

It remains available opt-in via `IRIS_NO_CLANG=1`, for anyone diagnosing it.

Two things need fixing, in this order:

1. **The link failure must surface.** Exiting 0 with no output is the real
   defect; the missing object format is only the trigger. Until this is fixed,
   any further attempt at this path can fail the same silent way.
2. **The object itself** — most likely a target-triple or object-format
   mismatch between what the C API is told to emit and what the MinGW linker
   expects.

Note the C-shim half of the clang dependency **is** gone: `iris_runtime.c`,
`onnx_shim.c` and `iris_ml_kernels.c` are now shipped as prebuilt objects
(`src/runtime/prebuilt/x86_64-pc-windows-gnu/`), so a build prints
"using prebuilt runtime objects … (no C compiler required)". Only `.ll` → `.o`
and the link still need clang. Do not describe IRIS as clang-free.

---

## 46. `set_result` silently skipped 34 of 134 instructions — **FIXED**

> **Fixed** on 2026-08-16 in `src/pass/inline.rs`. The catch-all is gone: every
> variant is now listed, so adding an `IrInstr` fails the build here instead of
> silently doing nothing.

`set_result` renumbers an instruction's result when a function's body is copied
into another. It matched 100 of the 134 `IrInstr` variants and ended in
`_ => {}`. Five of the 34 it skipped — `TapeRecord`, `Backward`, `TapeGrad`,
`SparseNnz`, `ResumeCont` — **do** define a result, so a callee containing any of
them kept the *callee's* `ValueId` after being inlined into a caller that numbers
its values independently.

`nnz` in a one-line helper was enough to trigger it:

```iris
def count_nz(a: [i64; 6]) -> i64 { val s = sparsify(a); nnz(s) }
```

```text
error[E0202]: [after inline] main: block0 instr17 operand[0] = %28
              not defined in function 'main'
```

So any function calling a small helper that used `nnz`, the autodiff tape, or
`resume` failed to compile. `InlinePass` inlines single-block callees of ten
instructions or fewer, which is exactly the shape such helpers take.

Found while building the mutual-recursion trampoline (#35), which renumbers
values the same way and would have inherited the identical hole.

**This is the fifth defect of this shape** — an enumeration a human has to
remember to extend, which was wrong every time it mattered (see #33, #42, and
the `iris_select` / `json_stringify` argument-count mismatches). The pattern is
now explicit in the fix: prefer an exhaustive match that breaks the build over a
catch-all that absorbs the next variant silently. `apply_replacements` in
`src/pass/opt.rs` was checked at the same time and is genuinely exhaustive
(134/134, no catch-all).

---

## 47. Every thread reserves 12.6 MB for an autodiff tape it never uses — **FIXED**

> **Fixed** on 2026-08-16. The tape is now allocated lazily in 4096-node chunks
> instead of as a fixed thread-local array, and is freed on thread exit via a
> pthread TLS destructor (`spawn` calls the user function directly, so there is
> no wrapper to free from; the destructor is the one hook covering every thread
> the runtime creates, including `par_for` workers).
>
> **Chunked rather than a single growable block on purpose:** nodes reference
> their parents by *pointer*, so reallocating one block would invalidate every
> parent pointer already recorded — corrupting the graph silently rather than
> failing. Chunk addresses are stable for the life of the tape. The node ceiling
> and wrap-around behaviour are unchanged.
>
> **Measured, same binaries and method as the original report:**
>
> | | before | after |
> |---|---|---|
> | `.tls$` in `iris_runtime.o` | 12,747,312 | **164,688** (−98.7%) |
> | `iris_runtime.o` | 12,910,081 | **329,192** |
> | a built IRIS binary | 13,035,008 | **453,120** (−96.5%) |
> | peak RSS, 4 threads | 77.4 MB | **6.1 MB** |
> | peak RSS, 64 threads | 808.1 MB | **16.6 MB** (−97.9%) |
>
> Per-thread cost is now ~175 KB, down from 12.2 MB — a 70× reduction. What
> remains is `handler_frames`, a 164 KB thread-local array of the same shape;
> worth the same treatment, but it is 1.3% of what this was.
>
> **Correctness verified, not assumed:** `tests/autodiff_tape_chunks.rs` asserts
> gradients through chains of 100, 4095, 4096, 4097 and 5000 nodes — the last
> spanning two chunks, which is precisely where a relocating allocator would
> produce a wrong gradient rather than a crash.
>
> Two **pre-existing** autodiff defects surfaced while verifying this, both
> confirmed identical on the compiler built immediately beforehand: see #48
> (a second `backward()` silently corrupts the first tape's gradients) and #49
> (a tape handle does not survive a function call or a `var`).

### Original report

The reverse-mode autodiff tape is a **statically sized thread-local array**:

```c
#define IRIS_TAPE_ARENA_SIZE 131072
static IRIS_THREAD_LOCAL IrisTapeNode  iris_tape_arena[IRIS_TAPE_ARENA_SIZE];
static IRIS_THREAD_LOCAL IrisTapeNode* iris_topo_buffer[IRIS_TAPE_ARENA_SIZE];
```

`IrisTapeNode` is 88 bytes, so that is 11.5 MB + 1 MB per thread, plus 160 KB of
handler frames — and the object file confirms it exactly:

```text
$ size -A iris_runtime.o
.tls$    12747312        <- 12.7 MB of thread-local storage
.text       86896
```

Every thread pays it whether or not it ever calls `grad`. **Measured**, with a
program whose threads only sleep and send an integer:

| threads | peak working set |
|---|---|
| 4  | 77.4 MB |
| 64 | 808.1 MB |

That is ~12.2 MB per thread by linear fit, against 12.58 MB predicted — this is
committed working set, not merely reserved address space. It also means **every
IRIS native binary is at least 13 MB**; the `tls_cost.iris` test binary, which
spawns threads and adds numbers, is 13,035,008 bytes.

Consequences, in the order they matter:

1. **Concurrency does not scale.** A few hundred `spawn`s exhausts memory on an
   ordinary machine. This undercuts the concurrency story directly.
2. **Embedded and robotics targets are out of reach.** The stated direction
   includes ESP32 and microcontroller-class hardware, where 12.6 MB per thread
   is not a tuning problem but a disqualification.
3. **It contradicts the memory positioning.** IRIS is pitched on deterministic,
   bounded-latency memory; a fixed 12.6 MB per-thread reservation is neither
   deterministic in aggregate nor bounded in any useful sense.

**Do not describe IRIS's concurrency or memory footprint without this number**
until it is fixed.

The fix is to allocate the tape lazily on first autodiff use and grow it from
the heap, rather than reserving the worst case per thread up front; the arena
size should also be tunable rather than a compile-time constant. This is a
runtime change with its own verification needs, so it is recorded rather than
bundled into the tail-call work that found it.

**Found while** deciding whether to commit the prebuilt runtime objects for #45
— a 12.9 MB `iris_runtime.o` was the symptom that led here. Those objects are
therefore **deliberately left uncommitted**: baking a 12.9 MB artifact into git
history is not worth it when the fix should shrink it by roughly 98%.

---

## 48. A second `backward()` silently corrupts the first tape's gradients — **FIXED**

> **Fixed** on 2026-08-16 by pinning leaves. Leaf nodes — the only nodes a
> program holds a handle to after a pass returns — now live in their own region
> that `iris_backward` never recycles. Intermediates still reset, which is what
> keeps a training loop from growing without bound.
>
> `dx` now reports `0` for a pass `x` took no part in, rather than borrowing
> `y`'s answer. Zero is the correct partial derivative here — `x` is not in
> `y2`'s graph — and gradients remain valid for the most recent pass, which is
> the existing epoch semantics.
>
> Exceeding the leaf budget returns NULL (a zero gradient) rather than recycling
> a live leaf: a missing answer, never another leaf's answer reported as yours.
>
> **Tested by** `a_later_tape_does_not_alias_an_earlier_leaf`,
> `the_second_pass_gradient_is_still_correct`,
> `sequential_passes_each_report_their_own_gradient` and
> `multiple_live_leaves_get_distinct_gradients` in
> `tests/autodiff_tape_chunks.rs`.

### Original report

`iris_backward` resets the arena index to zero when it finishes ("deterministic
memory reuse in the next training step"). The next `tape(...)` therefore hands
back **the same node address** the previous leaf is still holding, so the older
handle silently starts reading the newer node's gradient.

```iris
val x = tape(2.0);  val fx = x * x;   val _  = backward(fx);   // d/dx = 4
val y = tape(3.0);  val y2 = y * y;   val _2 = backward(y2);   // d/dy = 6
println("dx=" + to_str(grad(x)) + " dy=" + to_str(grad(y)));
```

```text
dx=6 dy=6        <- dx should be 4
```

No error, no crash — just a wrong number, and one that looks plausible. Any
program running more than one backward pass and reading a gradient from before
the last one gets silently wrong answers. That is the shape of an optimiser loop
that keeps per-parameter handles across steps, which is exactly what the ML
story requires.

The epoch machinery already exists to detect this — `iris_tape_grad` returns 0
when `grad_epoch` is stale — but reuse defeats it, because the recycled node
carries the *new* epoch. A fix has to either stop reusing addresses while
handles are live, or give each node an identity the handle can check.

**Pre-existing**, and confirmed as such: identical output from the compiler
built immediately before the #47 chunked-tape change.

---

## 49. A tape handle does not survive a function call or a `var` — **FIXED**

> **Fixed** on 2026-08-16, in three parts. The cause throughout was that a tape
> handle had no IR type: `TapeRecord`'s result was typed as the *primal*
> (`f64`), so it emitted as `double` and stopped being a handle the moment it
> crossed a block boundary. `IrType::TapeRef` now exists as an opaque pointer
> type, added by declaring the variant and letting the compiler enumerate the
> eight sites that needed an arm.
>
> **Loops.** A taped loop variable carries its handle across the back-edge as an
> extra block parameter beside its primal, in both `while` and `for`.
>
> **Function calls.** A handle cannot be passed in an `f64` parameter, and
> multi-value returns are unsupported (`Return` emits `values[0]` only), so it
> cannot come back out either. A call carrying a taped argument is therefore
> lowered *inline*. `tape_nodes` is keyed by `ValueId`, so binding the callee's
> parameters to the caller's argument values carries the mapping across for
> free — no specialised function, no new instruction, no runtime change.
>
> Inlining is gated by a **whitelist** of expression and statement forms rather
> than by scanning for `return`. A `return` is reachable only through a block,
> and blocks appear in just two expression forms — but finding those means
> recursing through every variant, and the nearest existing walker covers 33 of
> roughly a hundred. Missing one branch would inline a body whose `return`
> terminates the *caller*: a miscompile. The whitelist inverts the failure — an
> unrecognised form declines the inline and the program gets the same error it
> did before. Recursive taped calls are declined by a guard stack.
>
> **`if` expressions.** Found while testing the above and confirmed to fail with
> no helper and no inlining involved, so it was its own defect: a taped `if`
> merges its arms through a block parameter, dropping the handle exactly as a
> loop back-edge did. Branch arguments are emitted before the sibling arm has
> been lowered, so `IrFunctionBuilder::append_br_arg` extends them once both
> arms are known. Threaded only when *both* arms are taped — a one-sided graph
> would silently drop the other path's gradient.
>
> So a real training step now works end to end:
>
> ```iris
> def mse(pred: f64, target: f64) -> f64 { val d = pred - target; d * d }
>
> val w = tape(1.0);
> var total = tape(0.0);
> for s in 1..5 { total = total + mse(w * to_f64(s), 0.0); };
> val _ = backward(total);      // loss 30, grad(w) 60 — both asserted
> ```
>
> **Tested by** 22 tests in `tests/autodiff_tape_chunks.rs`. The three that
> matter most are the declines: a helper containing `return` must not be inlined
> (asserted by its caller running to completion), a recursive taped call must
> terminate, and untaped calls and loops must be bit-for-bit unaffected, since
> the threading touches every loop and every call the lowerer emits.

### Original report

Reverse-mode AD works only on straight-line code inside a single function.
Passing a taped value to a function, or accumulating into a `var`, loses the
handle and native codegen rejects the program:

```iris
var acc = tape(0.0);
while i < 10 { acc = acc + z * z; i = i + 1; };
val _ = backward(acc);
```

```text
error[E0300]: reverse-mode AD backward requires a lowered tape handle;
              use tape(...) on leaf values before calling backward(...)
```

The message is misleading: `tape(...)` *was* called on the leaf. What actually
failed is that the handle did not survive the reassignment.

This rules out the ordinary shapes of a training loop — accumulating a loss over
a batch, or computing it in a helper — so reverse-mode AD cannot currently
express a loop-accumulated loss. Forward mode (dual numbers) is unaffected.

**Do not describe reverse-mode autodiff as usable for training** without this
limit stated alongside.

**Pre-existing**, confirmed against the pre-#47 compiler.

---

## 50. The prebuilt runtime generator never ran, and its hash tracked the checkout — **FIXED**

> **Fixed** on 2026-08-16. Two independent defects in the same mechanism, both
> of which made the shipped prebuilt objects useless without ever failing
> loudly. The build printed
> `prebuilt runtime for x86_64-pc-windows-gnu is stale (built from different C
> sources) — falling back to compiling it`, which reads like a warning about
> *those* objects rather than a defect in the machinery that produced them.

**The generator never re-ran.** `IRIS_GENERATE_PREBUILT` had no
`cargo:rerun-if-env-changed` declaration, so setting it did nothing: cargo
considered the build script fresh and skipped it. The generator only ran when
some *other* build-script input happened to change in the same invocation.

Every "regeneration" after the first was therefore a no-op. The objects
committed alongside the #48 leaf-pinning fix were built from the *pre-#48* C
sources — a runtime without the fix. Nothing announced this; the objects were
simply never rebuilt. It was caught only because the hash check refused them,
which is the one part of this mechanism that worked as designed.

**The hash described the checkout, not the content.** `runtime_sources_hash`
read raw bytes of text files that git rewrites on checkout (`core.autocrlf`).
The working tree ends up with *mixed* line endings — some files rewritten,
some not — so the recorded hash matched neither the LF nor the CRLF form of any
committed revision:

```text
recorded          5c50191b539dd4dd
on-disk (mixed)   6096bb856166d794
HEAD as LF        5f0dc3ff0a95f4dc
HEAD as CRLF      ee064f4487b65ae0
```

Even with the generator fixed, a set generated on one machine would be rejected
on any machine with a different `autocrlf` setting — which defeats the entire
purpose of shipping the objects, since the point is that a *user* needs no C
compiler. The hash now normalises CRLF to LF before hashing, and
`.gitattributes` pins the runtime sources to LF so the compiled objects are
reproducible too.

**Consequence for anything claimed earlier:** a suite run that reported
"using prebuilt runtime objects (no C compiler required)" was accurate for that
run, but the claim that this made the result *reproducible from a clean
checkout* was not — on a fresh clone the hash would not have matched and the
build would have fallen back to compiling. Do not describe IRIS as
C-compiler-free until a clean clone has been observed accepting the prebuilt
set.

---

## 51. Three corpus files crash natively; a `ptr` result is printed as `i64` — **FIXED**

> **Fixed** on 2026-08-17. All three files run and assert, and are off
> `KNOWN_BROKEN`. It was not one defect but five, each found by fixing the one
> in front of it.
>
> **1. `Print` chose its helper from the IR type.** An `option<i64>` is a boxed
> value emitted as `ptr`, so selecting `iris_print_i64(i64 …)` produced invalid
> IR; clang rejected the module and the eval fallback then crashed, which is why
> the symptom was a segfault rather than a type error. `println` on *any* option
> was broken natively, not just the three constructs reported. A value emitted
> as `ptr` is now boxed and passed to `iris_print`, which dispatches on the tag
> and formats exactly as the interpreter does. `Str` keeps its own helper — it is
> a `ptr` but a raw `char*`, not a boxed value.
>
> **2. `to_str` had the identical defect**, in the sibling function: it
> `ptrtoint`-ed the option and printed the address (`find: 81260032` against the
> interpreter's `find: some(6)`).
>
> **3. `iris_value_to_str` handled no containers at all** — only BOOL/F/I/STR/
> UNIT — so even a correctly boxed option fell to its `<val:N>` default. It now
> formats `some(…)`, `none`, `ok(…)` and `err(…)`, quoting nested strings so
> `err("bad")` is distinguishable from a variant named `bad`.
>
> **4. `GetVariantTag` passed an `i64` to a `ptr` parameter.** A data-free enum
> variant is emitted as a bare tag, and the call did not coerce — though the
> surrounding code was already emitting the `inttoptr` for the same value two
> lines later.
>
> **5. `ProcessExit` was not treated as a block terminator.** It emits
> `unreachable`, but only `Panic` was registered as doing so, so `exit(1)` inside
> an `if` arm produced a block with two terminators. Clang did not diagnose that
> — it crashed outright with a bare stack dump, at every optimisation level
> including `-O0`. Both sites now share one `ends_block_with_unreachable`
> predicate rather than two matches that can drift.
>
> Two further defects surfaced while asserting the fixed files and are recorded
> separately: #60 (method-call sugar drifting from its builtin) and #61 (native
> `json_stringify`).

### Original report

Found by the corpus gate added for #4, on its first native run. The interpreter
sweep had reported these files as passing, because `IRIS_FORCE_INTERP=1`
bypasses codegen entirely.

`test_methods.iris`, `test_move.iris` and `test_pattern_guards.iris` are killed
by a signal (exit 139) under `--emit eval`. For `test_methods` the cause is
visible: native codegen emits invalid LLVM IR, clang rejects it, and the
*interpreter fallback then crashes* rather than recovering — so two defects
stack, and the second hides the first.

```text
module.ll:436:33: error: '%v47' defined with type 'ptr' but expected 'i64'
  call void @iris_print_i64(i64 %v47)
```

Three constructs reproduce the bad IR, all the same shape — a pointer-typed
result passed to the `i64` print helper:

```iris
def main() -> i64 { val s = "hello world"; println(s.find("world")); 0 }   // option<i64>
def main() -> i64 { val x = some(42); println(x.map(|v: i64| v * 2)); 0 }  // option<i64>
def main() -> i64 { val c = channel(); c.send(99); println(c.recv()); 0 }  // chan payload
```

`s.index(0)`, `m.len()`, `m.contains(..)`, `l.get(..)` and `unwrap_or(..)` are
all fine, so this is not "method calls are broken" — it is `Print` choosing its
helper from a type that disagrees with the value's *emitted* type. That is the
same class as the `emitted_types` mismatches fixed under #6 and the
`OptionUnwrap`/`MapGet` eager-unboxing work.

**The fallback crash is the more serious half.** `--emit eval` is documented as
building natively and falling back to the interpreter; a fallback that
segfaults removes the safety net exactly when it is needed.

**Also found:** `tests/test_adaptive.iris` exits 0 natively and 1 interpreted —
the same program, two answers. Recorded here rather than under #34, which is
about the values `std.adaptive` reports rather than about a divergence.

---

## 52. The corpus gate covers only the native backend — **FIXED**

> **Fixed** on 2026-08-17. `tests/iris_corpus.rs` gained
> `the_two_backends_agree`, which runs every corpus file both natively and under
> `IRIS_FORCE_INTERP=1` and asserts the output matches, plus
> `the_known_divergent_list_is_accurate` to keep the exemption list honest.
>
> Asserting the backends *agree* is stronger and cheaper than asserting each
> passes: it catches a regression in either, and it immediately caught `to_str`
> on an option printing a raw address natively while the interpreter printed
> `some(6)` (#51).
>
> The first sweep reported **all 122 files divergent**, which was the check
> working: `iris_codegen:` progress lines were going to *stdout*, so every
> native run began with a line the interpreter never emits. Moving them to
> stderr (#62) took the count to five real divergences, of which one was fixed
> and four are listed with reasons: #34, #55, #61, and `test_quick_wins`
> producing no output under the interpreter.
>
> Cost: the gate now takes roughly twice as long, since each file runs twice.
> That is the price of covering both backends, and it is worth it — every
> hand sweep before this used `IRIS_FORCE_INTERP=1` and so could not see a
> native crash at all.

### Original report

`tests/iris_corpus.rs` runs every `.iris` file through `--emit eval`, which
builds natively. It does **not** also run them under `IRIS_FORCE_INTERP=1`, so
an interpreter-only failure still escapes CI — and #51 shows the two backends
genuinely disagree, in both directions.

Doing both doubles the gate's runtime (currently ~6 minutes for 139 files).
Worth doing, and worth doing as a *divergence* check — asserting the two
backends agree is a stronger and cheaper statement than asserting each
separately passes.

---

## 53. `--emit eval` discards `main`'s return value, so half the documented test idiom does nothing — **FIXED**

> **Fixed** on 2026-08-17. `--emit eval` now exits with `main`'s return value,
> as a built binary already did. Both eval paths document the same output shape
> — printed lines first, then the returned value last — so the final line is the
> value; a non-integer return leaves the exit code at 0.
>
> The immediate effect was to make one silently-failing file visible:
> `test_slice_patterns.iris` returned **16** from `main` (the sum of seven arm
> indices) and had been exiting 0 regardless. It now asserts each helper and
> returns 0.
>
> Every other non-zero exit in the corpus belongs to a file already listed in
> `MUST_FAIL` or `KNOWN_BROKEN`, which is the expected shape: 18 of 139
> non-zero, 17 of them intentional.
>
> This also makes the corpus gate's exit-code check mean something. Before,
> `every_iris_test_runs` could only catch compile failures and crashes; a test
> that detected a wrong answer and reported it through a return code passed.

### Original report

A built binary propagates `main`'s return value as its exit code. `--emit eval`
does not: it prints the value and exits 0 regardless.

```text
def main() -> i64 { println("about to fail"); 1 }

$ iris build rc.iris && ./rc.exe ; echo $?      -> 1     correct
$ iris --emit eval rc.iris ; echo $?            -> 0     value printed, exit 0
```

The consequence is not cosmetic. Both `iris-write-code` and issue #4 tell test
authors:

> Either `assert(...)`, or return a non-zero code on mismatch.

**The second half does not work.** Under `--emit eval` — which is how the corpus
is run — a test that detects a mismatch and returns 1 exits 0 and passes. Of the
80 convertible corpus files, 34 use exactly this idiom, and every one of them
has been silently passing whatever it computed. A failing `assert` exits 1; that
is the only mechanism that actually fails a run.

This also caps what the corpus gate added for #4 can detect: `every_iris_test_runs`
checks the exit code, so it catches compile failures and crashes but *cannot*
catch a wrong answer in a file that reports through a return code.

The guidance has been corrected to say `assert(...)` and nothing else, and the
34 files are being converted. The underlying fix is to make `--emit eval` exit
with `main`'s value, as the built binary already does. That is deliberately not
bundled here: several corpus files return a computed value from `main`
(`test_const_generic` returns 156), so propagating the exit code turns them into
failures that must be fixed at the same time — real work, with its own
verification, rather than a one-line change.

---

## 54. A non-exhaustive `when` on an option is accepted and silently yields 0 — **FIXED**

> **Fixed** on 2026-08-17, for `option` and `result` alike. The covered arm now
> yields its value: `when o { some(x) => x }` on `some(42)` returns 42.
>
> The cause was a type-safety shortcut. A partial match makes the two branches
> disagree in type — the present arm yields its own type, the missing one has
> nothing — so `lower_option_when` forced **both** to unit to keep the merge
> block well-typed. Correctness was traded for typability, and the trade was
> silent: every input returned 0, including the ones the match covered.
>
> Only the *missing* branch now gets a default, of the same type as the arm that
> is present, so the merge stays consistent and the covered arm keeps its value.
> `lower_result_when` had the identical shortcut and the identical fix.
>
> **Still open:** exhaustiveness is not *enforced* — a missing arm is accepted
> rather than rejected. That is a separate decision about whether IRIS requires
> total matches, and it now fails safe: the covered case is right and the
> uncovered case yields a typed default rather than silently poisoning both.

### Original report

```iris
def test_option_non_exhaustive() -> i64 {
    val o = some(42)
    when o {
        some(x) => x,
        // no `none` arm
    }
}
```

This compiles, runs, and returns **0**. Two defects, and the second is the
dangerous one:

1. `ExhaustivePass` does not reject the missing `none` arm. `tests/test_option_non_exhaustive.iris`
   exists precisely to check that it does.
2. Worse, the match does not even take the arm that *does* apply. `o` is
   `some(42)`, so the result should be 42. `lower_option_when` computes
   `partial = some_arm.is_none() || none_arm.is_none()` and, when partial,
   yields the unit value instead of the arm's value — so a match missing one arm
   silently returns 0 for **every** input, including the ones it covers.

That is the exact failure mode #4 was raised about: a feature that silently
returns 0 while its test reports success. This file has been in the corpus
reporting success the whole time, because it asserted nothing and because
`--emit eval` ignores `main`'s return value (#53).

The file now asserts the correct answer (42) and is listed in `MUST_FAIL` in
`tests/iris_corpus.rs`, so the corpus gate stays honest: the file is expected to
fail today, and whichever way it is fixed — rejecting the match, or evaluating
the covered arm — it will keep failing until then and can be moved out of the
list when it passes.

The same `partial` logic exists in `lower_result_when`, and **it behaves
identically** -- confirmed:

```iris
val r: result<i64, str> = ok(42);
when r { ok(x) => x, }        // yields 0, not 42
```

So both option and result matches silently discard the covered arm's value when
any arm is missing. Any program that pattern-matches without covering every case
gets 0 rather than a compile error or the right answer.

---

## 55. The value returned by `weak_upgrade` cannot be used — **FIXED**

> **Fixed** on 2026-08-17. It was four independent defects stacked, which is why
> the payload was unreadable on both backends at once.
>
> **1. The interpreter dropped its own target immediately.** `weak_ref` built an
> `Arc`, downgraded it, and returned the `Weak` — leaving the `Arc` to be
> dropped at the end of the arm. The reference dangled instantly, so every
> `weak_upgrade` answered `none`. A registry now holds the strong owner.
>
> **2. The referent's type was never carried.** `weak_ref` recorded
> `WeakRef(Infer)` and `weak_upgrade` returned `option<Infer>`, so the
> downstream `unwrap` defaulted to `i64` and a weak-referenced list came back as
> an integer — `iris_list_len(ptr %v)` handed an `i64`, which is invalid IR.
> Both now carry the real type.
>
> **3. Codegen passed an unboxed container.** `iris_weak_ref(IrisVal* val)`
> stores a boxed value and hands the same box back inside the option; the raw
> container pointer meant the unwrap read a garbage tag.
>
> **4. The boxed value was not refcount-tracked.** `iris_weak_ref_new` only
> registers a weak reference whose target is already in the refcount table, so a
> freshly boxed value was dead on arrival and the upgrade returned `none`. The
> box is now retained.
>
> `tests/test_weak_ref.iris` now reads the payload and asserts its contents on
> both backends — the assertion the file could not previously make.
>
> **Two limits, stated:**
>
> * The referent's type must be known. `val t = list()` leaves it `Infer` and
>   the payload still comes back as an integer; annotate it. Same inference gap
>   as #14.
> * A weak reference never observes its target being collected, so `weak_alive`
>   is always true in the interpreter. Modelling collection needs object
>   identity the interpreter does not have — values are cloned, so there is no
>   single owner to outlive. **Do not describe weak references as observing
>   liveness.**

### Original report

`weak_ref` and `weak_alive` work. `weak_upgrade` returns a `some(...)`. But the
payload inside it cannot be touched:

```iris
val t = list(); list_push(t, 42);
val w = weak_ref(t);
val o = unwrap(weak_upgrade(w));

list_len(o)        // segfault, both backends
list_get(o, 0)     // segfault, both backends
type_of(o)         // error
list_len(t)        // fine -- the original handle is unaffected
```

So a weak reference can be created, checked for liveness and upgraded, and then
the result is unusable — which is the only reason to upgrade one.

`tests/test_weak_ref.iris` reported success throughout, because the only thing
it did with the upgraded object was pass it to `type_of` inside a `when` arm and
print the result. It never indexed it, never took its length, and asserted
nothing. Adding `assert(list_len(obj) == 2)` while converting it for #4 is what
surfaced this.

**Any program using `weak_ref` crashes the native backend.** Even without
touching the payload, a `weak_ref` + `weak_upgrade` sequence dies with
`0xc0000005` (access violation) and only completes because `--emit eval` falls
back to the interpreter. So weak references have never worked natively at all;
the fallback has been hiding it, and `tests/test_weak_ref.iris` reported success
throughout because it asserted nothing.

Reading the payload (`list_len`, `list_get`) crashes the interpreter too, so
there are two layers: a native crash for any use, and a representation problem
for the upgraded value on both backends.

The test now covers the surface that survives and states this limit, so the file
is honest rather than silently passing.

**Do not describe weak references as usable** until the payload can be read.

---

## 56. Comparison operators do not dispatch to their `Ord` impl — **FIXED**

> **Fixed** on 2026-08-17. `op_trait_method` mapped only the five arithmetic
> operators and sent everything else to a `_ => None`, so `<`, `<=`, `>`, `>=`,
> `==` and `!=` never looked for a trait impl at all. The dispatch machinery was
> right there and correct; it was simply never reached.
>
> All six now map (`Eq::eq`/`ne`, `Ord::lt`/`le`/`gt`/`ge`), and the match is
> exhaustive rather than defaulting — `And`/`Or` are listed as deliberately
> non-overloadable, since a method call would evaluate the right operand that
> short-circuiting exists to skip.
>
> Verified against the report's own table: every row now matches the impl.
>
> This is the **seventh** defect in this codebase caused by a `_` arm quietly
> absorbing a case that needed handling. The others: `set_result` (#46),
> `iris_select`, `json_stringify`, FFI argument packing (#33), `rust_call_*`,
> and `ProcessExit` as a terminator (#51).

### Original report

`impl Ord for V` defines `lt`/`le`/`gt`/`ge`. Calling the methods directly gives
the right answers; using the operators does not.

```iris
record V { x: i64 }
impl Ord for V {
    def lt(self, other: V) -> bool { self.x < other.x }
    def le(self, other: V) -> bool { self.x <= other.x }
    def gt(self, other: V) -> bool { self.x > other.x }
    def ge(self, other: V) -> bool { self.x >= other.x }
}
val a = V { x: 1 };  val b = V { x: 1 };  val c = V { x: 2 };
```

| expression | result | correct |
|---|---|---|
| `a < c` | **false** | true |
| `a <= b` | **false** | true |
| `a <= c` | **false** | true |
| `c >= a` | **false** | true |
| `a >= b` | true | true |
| `a.le(b)` | true | true |

The direct call is right, so the impl and the method dispatch are fine; the
operator forms are not reaching them. The answers are not uniformly `false`
either — `a >= b` is correct and `a < d` happens to be correct in
`tests/test_tier2.iris` — which suggests the operands are being compared some
other way rather than the impl simply being ignored.

`==` and `!=` **do** dispatch correctly to `impl Eq` in the same file, so this is
specific to the ordering operators.

This was invisible because `tests/test_tier2.iris` guarded each comparison with
`if !(a <= c) { println("FAIL: le"); return 1 }` and returned a code that
`--emit eval` discards (#53). The file printed "FAIL: le" on every run and
still exited 0. It is listed in `KNOWN_BROKEN` now that its checks are real.

Related: #26 records that tuples cannot be compared with `==` on either backend.

---

## 57. `map_entries` crashes the native backend — **FIXED**

> **Fixed** on 2026-08-17. Two mismatches against the same runtime signature,
> on both sides of the call.
>
> `iris_map_entries(IrisVal* map_val)` takes a **boxed** map and unboxes it
> internally; codegen passed the raw `IrisMap*`, so the unbox read a tag that
> was never written (`unbox_map type mismatch`). It also *returns* a boxed list,
> while the IR type is `list<...>` and every consumer (`list_len`, `list_get`)
> expects the raw `IrisList*` — so fixing only the argument moved the failure
> from a tag mismatch to an access violation.
>
> Both sides are now handled: the argument is boxed, the result unboxed.
> `map_entries` agrees across backends, and `tests/test_map_entries.iris` passes
> natively rather than only through the interpreter fallback.

### Original report

```iris
def main() -> i64 {
    val m = map();
    map_set(m, "a", 1);
    println(to_str(list_len(map_entries(m))));
    0
}
```

```text
runtime error (exit code 0xc0000409): iris: unbox_map type mismatch (tag=2078172688)
```

The program then completes only because `--emit eval` falls back to the
interpreter, where it is correct. `map_set`, `map_get`, `map_keys`,
`map_values`, `map_contains`, `map_len` and `map_remove` are all fine natively —
this is specific to `map_entries`, which returns a list of key/value pairs and
evidently unboxes the map with the wrong tag.

`tests/test_map_entries.iris` passes, including the assertions added for #4,
because the fallback hides it. Found while writing the map examples.

Prefer `map_keys` plus `map_get` when iterating a map in code that must run
natively.

---

## 58. The LSP server exits 1 when the editor closes stdin — **FIXED**

> **Fixed** on 2026-08-17 in `src/lsp.rs`. EOF on stdin now ends the session
> normally instead of propagating out of `run_lsp_server`.

The read loop used `read_exact(&mut byte)?`. When the client closed stdin, that
returned `UnexpectedEof`, `?` propagated it, and `main` did:

```rust
if let Err(e) = iris::lsp::run_lsp_server() {
    eprintln!("LSP server error: {}", e);
    process::exit(1);
}
```

```text
LSP server error: failed to fill whole buffer
```

`"exit" => break` was handled, so an orderly `shutdown` + `exit` was fine. But VS
Code closes the pipe on **window reload** and on **extension-host restart**
without always sending `exit` first, and a language client treats a non-zero exit
as a *crash*. After a handful of those in one window it stops restarting the
server and reports that it failed to start — which is what the symptom looks
like from the user's side, with nothing wrong with the binary or its path.

Measured, driving the server with a real initialize / didOpen / documentSymbol /
shutdown sequence and then closing stdin:

| | exit | stderr |
|---|---|---|
| before | 1 | `LSP server error: failed to fill whole buffer` |
| after | 0 | clean, all responses still returned |

A truncated body mid-message — a client that was killed — is handled the same
way, and also exited 1 before.

Also fixed while there: the header parser called `stdin.lock()` once **per
byte**, so it acquired a lock for every character of every header. It now locks
once for the life of the server.

**If the symptom returns, check the binary the extension is pointing at.**
`iris.executablePath` should be a stable copy (`~/.iris/bin/iris.exe`), never
`target/debug/iris.exe` — every `cargo build` relinks that file, and on Windows a
running server holds it locked, so the two fight. Refresh the stable copy after a
build; there is a VS Code task for it.

---

## 59. `iris lsp` rejected `--stdio`, so the VS Code server never started — **FIXED**

> **Fixed** on 2026-08-17. `lsp` and `dap` now accept `--stdio` and ignore it;
> stdio is the only transport either speaks.

`vscode-languageclient` appends `--stdio` to the server's argv whenever
`TransportKind.stdio` is configured, which the IRIS extension does. `Command::Lsp`
was a unit variant, so clap rejected the flag:

```text
error: unexpected argument '--stdio' found
Usage: iris.exe lsp
```

The server exited 1 the instant it was spawned, the client's first write failed
with `write EPIPE`, and after five attempts VS Code gave up:

```text
The IRIS Language Server server crashed 5 times in the last 3 minutes.
The server will not be restarted.
```

**Nothing about the binary, the path or `iris.executablePath` was wrong**, which
is what made this hard to see. `iris lsp` run by hand works perfectly; only
`iris lsp --stdio` — the exact command VS Code runs — failed. The client's
`EPIPE` names the symptom rather than the cause, and the usage error goes to the
server's stderr, which only surfaces in the extension's **Output → IRIS Language
Server** channel. That channel is the first place to look for this class of
failure, not the last.

Two diagnoses were made and discarded before this one: the LSP server's EOF
handling (a real defect, fixed as #58, but not this) and machine resource
pressure (wrong).

`dap` had the same missing flag and, once past it, the same EOF defect as #58;
both are fixed. `tests/lsp_dap_cli_transport.rs` runs all four invocations.

---

## 60. Method-call forms of builtins drift from the builtin itself — **partly fixed**

Several builtins have two lowerings: `recv(ch)` and `ch.recv()`, `send(ch, v)`
and `ch.send(v)`, `list_pop(l)` and `l.pop()`. They are separate code paths, and
they have drifted.

**Fixed:** `ch.send(v)` did not record the channel's concrete element type the
way `send(ch, v)` does, so the matching `ch.recv()` fell back to the channel's
declared element type — `Infer` for an unannotated `channel()` — and left the
received value boxed. `assert(ch.recv() == 99)` then emitted

```text
%v153 = icmp eq ptr %v151, 99      ; invalid: ptr vs integer constant
```

while `assert(recv(ch) == 99)` was fine. Both forms now record and consult the
same map.

**Fixed:** `process.exit(1)` resolved through the module-qualified path, which
looked the name up in `fn_sigs` — where builtins also live — and emitted a
*generic* call, skipping the dispatch that turns `exit` into `ProcessExit`:

```text
%v38 = call i64 @exit(ptr 1)       ; against declare void @exit(i32)
```

That path now delegates to `lower_call`, so one implementation serves both.

**Still open:** `l.pop()` is typed `option<_>` but yields a raw `i64` at
runtime, so neither `l.pop() == 30` (a compile-time type error) nor
`l.pop().unwrap()` (`OptionUnwrap on non-option: I64(30)`) works. The builtin
`list_pop(l)` is consistent. `tests/test_methods.iris` asserts the pop through
the builtin and says why.

The pattern is worth stating plainly: **a builtin with a method-call sugar has
two implementations that can disagree, and every instance found so far has.**
The durable fix is for the sugar to desugar to the builtin rather than
re-implement it, as the extension-method path already does.

---

## 61. Native `json_stringify` prints an f64 bit pattern — **FIXED**

> **Fixed** on 2026-08-20. The reported symptom was one case of a general
> defect: **no record could be stringified natively.** A record with `str` and
> `i64` fields returned the literal `"null"`, not just a wrong number.
>
> A record is emitted natively as a pointer to a flat LLVM struct
> (`%Point = type { double, double }`). That is not an `IrisVal*`, but
> `box_to_ptr` classified it under "not a scalar — already a ptr, no boxing
> needed" and passed it straight to a runtime function that walks *tagged*
> values. The walker read the first field's bytes as the tag: for
> `Point { x: 1.5, … }` that is `4612811918334230528`, the IEEE-754 pattern of
> 1.5 — which is why the symptom looked like a float on an integer path rather
> than a pointer on a value path.
>
> Records are now built into the shape the runtime actually expects: an
> `IrisList` of boxed field values tagged `IRIS_TAG_STRUCT`, via a new
> `iris_box_struct` helper. Fields are boxed by their own IR type, so a nested
> record recurses.
>
> Verified across all four shapes, native output identical to interpreted:
> `{"0":1.5,"1":2.5}`, `{"0":"hi","1":7}`, and nested
> `{"0":{"0":1.5,"1":2.5},"1":{"0":3,"1":true}}`. `test_json_auto.iris` is off
> `KNOWN_DIVERGENT`.
>
> **Limit, stated:** JSON keys are numeric indices, not field names. Field names
> are not present at runtime, and this fix does not add them — the interpreter
> has always behaved the same way, so the two agree, but neither produces the
> JSON a caller would want to send anywhere. Naming that properly needs the
> record's field list carried into the runtime.

### Original report

```text
native: PASS: json_stringify(Point) = 4612811918334230528
interp: PASS: json_stringify(Point) = {"0":1.5,"1":2.5}
```

`4612811918334230528` is the IEEE-754 bit pattern of `1.5` read as an integer,
so a `double` is reaching an integer path somewhere in the native
`json_stringify` for a record. Found by the backend-agreement gate added for
#52; `tests/test_json_auto.iris` is listed in `KNOWN_DIVERGENT`.

---

## 62. `iris_codegen:` diagnostics were written to stdout — **FIXED**

> **Fixed** on 2026-08-17: all five emitters now use `eprintln!`.

The native build path printed progress and fallback notices —
`iris_codegen: using prebuilt runtime objects…`, `ld.lld link failed…` — to
**stdout**, mixed into the program's own output. Anything consuming that output
saw compiler chatter prepended to the program's results.

It hid a whole class of bug: the first backend-agreement sweep reported **all
122 corpus files as divergent**, because the native run's stdout began with a
codegen line the interpreter never emits. Real divergences were invisible behind
it. With the diagnostics on stderr the same sweep reported five, which was
actionable.

Diagnostics belong on stderr; stdout belongs to the program.

---

## 63. `for b in some_str` compiles but fails at runtime — **FIXED**

> **Fixed** on 2026-08-20. `for b in s` over a `str` iterates its bytes, which
> is what `tests/test_tier2.iris` expected: `sum_bytes("AB") == 131`.
>
> `lower_foreach` had one path and assumed it was a list — `ListLen` for the
> bound, `ListGet` for the element — so a string reached `list_len` and died at
> runtime. Both instructions have string counterparts already (`StrLen`,
> `StrIndex`, the latter documented as "byte value at position i"), so the fix
> is a branch on the iteratee's type, not a new primitive. No `bytes(s)` builtin
> is needed.
>
> The element type came from `match &iter_ty { IrType::List(inner) => …, _ =>
> IrType::Infer }` — the eighth instance in this codebase of a `_` arm quietly
> absorbing a case that needed handling, and it defaulted the byte to `i64` via
> the usual inference fallback. `str` is now handled and a **scalar iteratee is
> rejected at compile time** rather than emitting a `list_len` that cannot
> succeed:
>
> ```text
> error[E0103]: cannot iterate a value of type i64 -- `for` needs a list, a range or a str
> ```
>
> `Infer` deliberately stays on the list path: there the type is genuinely
> unknown, not known-bad.
>
> Verified on both backends: `sum_bytes("AB") == 131`, `sum_bytes("") == 0`, and
> a five-iteration count over `"hello"`. `test_tier2.iris` passes and is off
> `KNOWN_BROKEN`.

### Original report

```iris
def sum_bytes(s: str) -> i64 {
    var total = 0;
    for b in s { total = total + b }
    total
}
```

```text
error[E0400]: [runtime error] type error — list_len: not a list
```

Iterating a `str` directly is accepted by the lowerer, which then treats the
string as a list and calls `list_len` on it. Either the loop should iterate the
string's bytes (which is what `tests/test_tier2.iris` expects — it calls this
"char/byte iteration" and asserts `sum_bytes("AB") == 131`, i.e. 65 + 66), or
the compiler should reject it and require an explicit conversion.

There is no `bytes(s)` builtin either, so there is currently no working way to
iterate a string's bytes.

Found after fixing #56 unblocked the earlier assertions in `test_tier2.iris`,
which now reaches this line. That file stays in `KNOWN_BROKEN` for this reason
rather than the comparison one.
