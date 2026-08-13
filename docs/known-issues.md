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

## Verified working

Confirmed correct by running and asserting output:

- Labelled `break` / `continue` on `loop`, `for`, `while`
  (`examples/01_basics/labeled_loops.iris`)
- Struct patterns, or-patterns, range patterns, guards
  (`examples/04_types_and_traits/struct_patterns.iris`)
- Extension methods, including chaining and on user records
  (`examples/02_functions/extensions_and_defaults.iris`)
- Default record fields, including partial override (same file)
