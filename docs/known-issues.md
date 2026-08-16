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

## 2. `str` field in a record inside `result<T, E>` — **open**

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

## 3. Assigning to an enclosing `var` from `when` arms — **workaround exists**

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

## 4. 104 of 122 `.iris` tests assert nothing — **systemic**

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

## 5. Tail-call optimisation is interpreter-only — **documentation**

`tests/test_tco.iris` states it exercises TCO "in the interpreter". There is no
`musttail` in codegen, so deeply recursive functions can still overflow the stack
in native binaries. The capability should not be described without that
qualification.

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

## 10. Effect clauses and traits — **open, two defects**

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

## 13. `pub` is not enforced across module boundaries — **open, needs design**

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

## 16. A `handle` expression as a function's return value — **open, compiler bug**

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

## 21. A generic instantiated at a generic type — **open**

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

## Verified working

Confirmed correct by running and asserting output:

- Labelled `break` / `continue` on `loop`, `for`, `while`
  (`examples/01_basics/labeled_loops.iris`)
- Struct patterns, or-patterns, range patterns, guards
  (`examples/04_types_and_traits/struct_patterns.iris`)
- Extension methods, including chaining and on user records
  (`examples/02_functions/extensions_and_defaults.iris`)
- Default record fields, including partial override (same file)
