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
