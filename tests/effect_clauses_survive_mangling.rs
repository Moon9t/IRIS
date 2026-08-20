//! Effect clauses on impl methods must survive name mangling.
//!
//! Regression test for known-issues #39. `src/stdlib/container.iris` declares
//! `effect alloc` on every method that calls `list_len`, yet compiling anything
//! that brought it printed
//!
//! ```text
//! error[E0302]: function `list_len` requires effect `alloc`
//!               but caller `container__Sized__list__size` has effects `pure`
//! ```
//!
//! The clause was present and correct in the source. `EffectChecker` looked the
//! method up by *splitting the mangled name on `__`* — which stops working the
//! moment a module prefix is present, because a module prefix is itself joined
//! with `__`. `container__Sized__list__size` split into
//! `("container", "Sized", "list__size")`, matched no impl, and the declared
//! effects were never read: every trait method in a brought module was `pure`.
//!
//! Under `--strict-effects` those were build failures, which made this a
//! blocker for the single claim the effect system exists to support — that a
//! control path can be proven to allocate nothing. Any trait method failed that
//! proof regardless of what it did.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn write_temp(name: &str, src: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("iris_effect39_{}.iris", name));
    let mut f = std::fs::File::create(&p).expect("create temp source");
    f.write_all(src.as_bytes()).expect("write temp source");
    p
}

/// Compile with `--strict-effects`, returning combined output and success.
fn strict_compile(name: &str, src: &str) -> (bool, String) {
    let path = write_temp(name, src);
    let out = Command::new(env!("CARGO_BIN_EXE_iris"))
        .args(["--strict-effects", "--emit", "eval"])
        .arg(&path)
        .output()
        .expect("failed to launch the iris binary");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_file(&path);
    (out.status.success(), text)
}

/// The defect itself: a brought module's trait methods keep their clauses.
#[test]
fn a_brought_trait_method_keeps_its_effect_clause() {
    let (ok, text) = strict_compile(
        "container",
        r#"bring std.container
def main() -> i64 effect io, alloc, throw {
    val xs: list<i64> = list();
    list_push(xs, 1);
    list_push(xs, 2);
    assert(xs.size() == 2);
    println("ok");
    0
}
"#,
    );

    // The specific failure this issue is about.
    assert!(
        !text.contains("E0302"),
        "a trait method in a brought module was treated as `pure` again \
         (known-issues #39):\n{}",
        text
    );
    assert!(
        !text.contains("has effects `pure`"),
        "an impl method reported `pure` despite declaring an effect:\n{}",
        text
    );
    assert!(
        ok,
        "std.container must compile under --strict-effects:\n{}",
        text
    );
}

/// The clauses in `std.container` must describe what the bodies actually do.
///
/// Fixing #39 made the checker read impl-method clauses for the first time, and
/// it immediately found three wrong ones: the `option` impls declared `alloc`
/// while performing `throw` (they call `unwrap`, and never touch the heap).
/// They declared `alloc` only because the `list` impls above them do.
#[test]
fn container_clauses_match_container_bodies() {
    let (_, text) = strict_compile(
        "clauses",
        r#"bring std.container
def main() -> i64 effect io, alloc, throw {
    val o = some(3);
    assert(o.size() == 1);
    println("ok");
    0
}
"#,
    );
    assert!(
        !text.contains("E0303"),
        "a std.container method performs an effect its clause does not cover:\n{}",
        text
    );
}

/// Strict mode must still reject an *undeclared* effect. A regression test for
/// a fix to an effect checker is worth little if the checker has stopped
/// checking — this asserts the gate still closes.
#[test]
fn strict_mode_still_rejects_an_undeclared_effect() {
    let (ok, text) = strict_compile(
        "negative",
        r#"def main() -> i64 {
    println("this performs io with no effect clause");
    0
}
"#,
    );
    assert!(
        !ok,
        "strict mode accepted a function performing `io` with no effect clause; \
         the gate is no longer closing:\n{}",
        text
    );
}

// ---------------------------------------------------------------------------
// known-issues #64 — effects reached through a method call
// ---------------------------------------------------------------------------

/// The soundness hole. `xs.size()` allocates; `sneaky` declares nothing.
///
/// `--strict-effects` accepted this, which meant the guarantee the flag exists
/// to provide — "a function with no `effect` clause that compiles has been
/// proven to allocate nothing" — did not hold for any call graph containing
/// method-call syntax.
///
/// The cause was not ambiguity between impls, as first reported. The
/// `MethodCall` arm of the callee collector walked into the receiver and the
/// arguments and then **dropped the call itself**, so `xs.size()` contributed
/// nothing to the call graph at all.
#[test]
fn a_method_call_carries_its_effects() {
    let src_no_clause = r#"bring std.container
def sneaky(xs: list<i64>) -> i64 {
    xs.size()
}
def main() -> i64 effect io, alloc, throw {
    val xs: list<i64> = list();
    list_push(xs, 1);
    assert(sneaky(xs) == 1);
    println("ok");
    0
}
"#;
    let (ok, text) = strict_compile("hole", src_no_clause);
    assert!(
        !ok,
        "a function that allocates through a method call was certified \
         allocation-free (known-issues #64):\n{}",
        text
    );
    assert!(
        text.contains("alloc"),
        "the rejection should name the effect that leaked through the method \
         call:\n{}",
        text
    );

    // ... and declaring it must be enough to compile. A checker that rejects
    // everything is not a fix.
    let (ok, text) = strict_compile("hole_ok", &src_no_clause.replace(
        "def sneaky(xs: list<i64>) -> i64 {",
        "def sneaky(xs: list<i64>) -> i64 effect alloc {",
    ));
    assert!(
        ok,
        "declaring the effect should satisfy the checker:\n{}",
        text
    );
}

/// An unresolved method name takes the union over every impl defining it.
///
/// `std.container` has `Sized for list<T>` (allocates) and `Sized for
/// option<T>` (pure), both defining `size`. `total_size` and `is_singleton`
/// call `a.size()` and correctly declare `effect alloc`; before the fix the
/// checker could not see the allocation and reported their correct clauses as
/// "declares effect `alloc` that the body doesn't use".
///
/// Those clauses must not be deleted to silence the warning — the warning was
/// wrong, not the clause.
#[test]
fn an_unresolved_method_unions_the_impls_that_define_it() {
    let (ok, text) = strict_compile(
        "union",
        r#"bring std.container
def main() -> i64 effect io, alloc, throw {
    val xs: list<i64> = list();
    list_push(xs, 1);
    list_push(xs, 2);
    assert(xs.size() == 2);
    assert(is_singleton(xs) == false);
    println("ok");
    0
}
"#,
    );
    assert!(
        !text.contains("doesn't use"),
        "a correct effect clause was reported as unused, so the checker still \
         cannot see through a method call (#64):\n{}",
        text
    );
    assert!(ok, "std.container must stay strict-clean:\n{}", text);
}

/// Closures and `dyn Trait` dispatch carry their effects too.
///
/// Neither was checked before #64. A closure is caught because its body is
/// walked where it is written; `dyn` dispatch is caught because it is a method
/// call, and method calls are now edges in the call graph.
#[test]
fn closures_and_dyn_dispatch_carry_their_effects() {
    let (ok, text) = strict_compile(
        "closure",
        r#"def hidden() -> i64 {
    val f = |x: i64| { println("io from a closure"); x };
    f(1)
}
def main() -> i64 effect io { val r = hidden(); 0 }
"#,
    );
    assert!(!ok, "a closure's io escaped its caller's effect row:
{}", text);

    let (ok, text) = strict_compile(
        "dyn",
        r#"trait Speak { def say(self) -> i64 }
record Loud { n: i64 }
impl Speak for Loud { def say(self) -> i64 effect io { println("loud"); self.n } }
def hidden(s: dyn Speak) -> i64 { s.say() }
def main() -> i64 effect io {
    val l: Loud = Loud { n: 1 };
    val r = hidden(l);
    0
}
"#,
    );
    assert!(!ok, "dyn dispatch escaped its caller's effect row:
{}", text);
}

/// KNOWN-WRONG, pinned deliberately. See known-issues #65.
///
/// An effect reached through a *function-valued parameter* is still invisible:
/// `hidden` declares nothing, calls its own parameter, and performs whatever
/// that parameter performs. `f` is a parameter, not a function name, so the
/// callee collector has no name to record — unlike #64 this cannot be fixed by
/// recording one. It needs the callee's effects to be part of the parameter's
/// type (effect polymorphism), which is a language-surface change.
///
/// Asserted as-is so that closing the hole fails this test and forces the
/// assertion to be updated, rather than the gap surviving another release
/// unnoticed — the same device that made #34's fabricated metrics surface.
#[test]
fn a_function_valued_parameter_still_hides_its_effects() {
    let (ok, text) = strict_compile(
        "higher_order",
        r#"def noisy(x: i64) -> i64 effect io { println("noisy"); x }
def hidden(f: |i64| -> i64) -> i64 { f(1) }
def main() -> i64 effect io {
    val g = |x: i64| noisy(x);
    val r = hidden(g);
    0
}
"#,
    );
    assert!(
        ok,
        "#65 appears to be FIXED -- a function-valued parameter now carries its          effects. Good. Update this test to assert rejection, and mark #65 fixed          in docs/known-issues.md, CLAUDE.md and the iris-claims skill, all three          of which state this as the one remaining bound on the          allocation-freedom claim.
{}",
        text
    );
}
