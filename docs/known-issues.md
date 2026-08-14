# Known Issues

Defects found by writing new IRIS programs against v1.0.0-rc1 and verifying
their output. Each entry has a minimal reproduction.

---

## 1. Named arguments silently evaluate to 0 — **critical**

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

## 8. `==` on two enum values is unsupported — **open, runtime**

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

## 11. `test_autodiff_determinism_profiling` is flaky — **test defect, not a compiler defect**

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

## 15. A user function is silently discarded when a builtin shares its name — **critical**

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

## 17. Labelled `continue` fails in the interpreter — **open, backend divergence**

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

## 18. Trait objects only work through an annotated intermediate binding — **open**

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

## Verified working

Confirmed correct by running and asserting output:

- Labelled `break` / `continue` on `loop`, `for`, `while`
  (`examples/01_basics/labeled_loops.iris`)
- Struct patterns, or-patterns, range patterns, guards
  (`examples/04_types_and_traits/struct_patterns.iris`)
- Extension methods, including chaining and on user records
  (`examples/02_functions/extensions_and_defaults.iris`)
- Default record fields, including partial override (same file)
