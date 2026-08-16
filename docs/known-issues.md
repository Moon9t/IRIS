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

## 34. `std.adaptive` reports risk numbers it never measures — **open, silent**

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
> **Limits, deliberately:** only *self* recursion. Mutual recursion needs a
> shared trampoline and is a different transformation, left alone rather than
> half-done. A call whose result is used before being returned —
> `n * fact(n - 1)` — is not a tail call and is untouched.

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
