//! The `.iris` corpus: executed, and gated on asserting its results.
//!
//! Two problems, both named in `CLAUDE.md`, both addressed here.
//!
//! **Nothing globbed `tests/*.iris`.** The corpus was never executed by
//! `cargo test`, so a file could rot indefinitely with no run noticing.
//! Sweeping it by hand with `IRIS_FORCE_INTERP=1` is not equivalent: that
//! bypasses codegen, and the first native sweep found three crashes the
//! interpreter sweep could not see (`test_methods`, `test_move`,
//! `test_pattern_guards`), plus a file that passes natively while failing
//! interpreted. These tests drive the real CLI, so codegen is exercised.
//!
//! **Most files assert nothing.** 0 of 139 print results without
//! checking them, so they pass whenever the program compiles and exits 0,
//! regardless of whether the output is right (known-issues #4). Converting them
//! is mechanical but has to be done by *running* each file and reading its real
//! values, so it happens in batches. `NEEDS_ASSERTIONS` is the shrinking record
//! of what is left, and `the_needs_assertions_list_is_accurate` fails once a
//! listed file gains assertions, so the list cannot quietly drift.

use std::path::Path;
use std::process::Command;

/// Files the compiler is *supposed* to reject. Here, passing is the failure.
const MUST_FAIL: &[(&str, &str)] = &[
    ("test_borrow_error.iris",
     "the borrow checker must reject it"),
    ("test_exhaustiveness.iris",
     "a non-exhaustive match must be rejected"),
    ("test_exhaustiveness_simple.iris",
     "a non-exhaustive match must be rejected"),
    ("test_move_borrow_error.iris",
     "borrow of a moved value must be rejected"),
    ("test_move_error.iris",
     "use after move must be rejected"),
];

/// Files that do not currently run, each with why. A debt register, not a
/// permission slip: every entry names a cause, and the list should only shrink.
const KNOWN_BROKEN: &[(&str, &str)] = &[
    ("test_doc_comments.iris",
     "no zero-argument function, so there is nothing to evaluate"),
    ("test_features_11_14.iris",
     "parse error: uses syntax the compiler does not accept"),
    ("test_ffi_full.iris",
     "requires iris_ffitest.dll in the working directory"),
    ("test_generic_set.iris",
     "a type param only in the return type needs an annotation -- #14"),
    ("test_mod_min.iris",
     "parse error: uses syntax the compiler does not accept"),
    ("test_mod_simple.iris",
     "parse error: uses syntax the compiler does not accept"),
    ("test_nursery.iris",
     "print() arity"),
    ("test_par_map.iris",
     "parse error: uses syntax the compiler does not accept"),
    ("test_refine_fail.iris",
     "parse error: fails at parse, not at the refinement it is named for"),
    ("test_struct_update_simple.iris",
     "parse error: uses syntax the compiler does not accept"),
    ("test_tier2.iris",
     "for b in str fails at runtime -- #63"),
];

/// Files that do not yet assert their results. Shrinking; see #4.
const NEEDS_ASSERTIONS: &[&str] = &[

];

/// Files whose two backends disagree, each with why. The gate below asserts
/// that *everything else* agrees, so a new divergence fails rather than joining
/// this list silently.
const KNOWN_DIVERGENT: &[(&str, &str)] = &[
    ("test_adaptive.iris",
     "runs natively, fails interpreted -- std.adaptive, #34"),
    ("test_json_auto.iris",
     "native json_stringify prints an f64 bit pattern -- #61"),
    ("test_quick_wins.iris",
     "produces no output under the interpreter"),
];

fn corpus() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir("tests")
        .expect("tests/ must be readable")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".iris"))
        .collect();
    v.sort();
    v
}

fn asserts(name: &str) -> bool {
    std::fs::read_to_string(Path::new("tests").join(name))
        .unwrap_or_default()
        .contains("assert(")
}

/// Runs a corpus file through the real CLI. The exit code is `None` when the
/// process was killed by a signal, which is how a crash shows up.
fn run(name: &str) -> (Option<i32>, String) {
    run_with(name, false)
}

/// Runs a corpus file, optionally forcing the interpreter.
fn run_with(name: &str, force_interp: bool) -> (Option<i32>, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_iris"));
    cmd.args(["--emit", "eval"]).arg(Path::new("tests").join(name));
    if force_interp {
        cmd.env("IRIS_FORCE_INTERP", "1");
    }
    let out = cmd
        .output()
        .expect("failed to launch the iris binary");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code(), text)
}

/// Just the program's stdout, with the compiler's diagnostics excluded.
///
/// The divergence check must compare what the *program* printed. The native
/// path legitimately writes build notices to stderr that the interpreter never
/// emits, so comparing combined output reports every file as divergent.
fn stdout_of(name: &str, force_interp: bool) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_iris"));
    cmd.args(["--emit", "eval"]).arg(Path::new("tests").join(name));
    if force_interp {
        cmd.env("IRIS_FORCE_INTERP", "1");
    }
    let out = cmd.output().expect("failed to launch the iris binary");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

// -- The assertion gate (#4) ----------------------------------------------

/// Every corpus file must assert its results, or be listed as not yet doing so.
/// A *new* file that asserts nothing fails here instead of joining the backlog
/// unnoticed.
#[test]
fn every_iris_test_asserts_or_is_listed() {
    // A file that never runs cannot assert anything, so negative tests and
    // known breakages are exempt -- they are tracked by their own lists, each
    // with its own accuracy check.
    let never_runs: Vec<&str> = MUST_FAIL
        .iter()
        .map(|(f, _)| *f)
        .chain(KNOWN_BROKEN.iter().map(|(f, _)| *f))
        .collect();
    let unlisted: Vec<String> = corpus()
        .into_iter()
        .filter(|f| {
            !asserts(f)
                && !NEEDS_ASSERTIONS.contains(&f.as_str())
                && !never_runs.contains(&f.as_str())
        })
        .collect();
    assert!(
        unlisted.is_empty(),
        "these .iris tests assert nothing and are not on NEEDS_ASSERTIONS. A test \
         that only prints cannot fail. Add assert(...) to each:\n  {}",
        unlisted.join("\n  ")
    );
}

/// The backlog must stay honest: a file that has gained assertions comes off the
/// list, or the remaining count stops meaning anything.
#[test]
fn the_needs_assertions_list_is_accurate() {
    let stale: Vec<&str> = NEEDS_ASSERTIONS.iter().copied().filter(|f| asserts(f)).collect();
    assert!(
        stale.is_empty(),
        "these files now assert and must be removed from NEEDS_ASSERTIONS:\n  {}",
        stale.join("\n  ")
    );

    let gone: Vec<&str> = NEEDS_ASSERTIONS
        .iter()
        .copied()
        .filter(|f| !Path::new("tests").join(f).exists())
        .collect();
    assert!(
        gone.is_empty(),
        "these files no longer exist and must be removed from NEEDS_ASSERTIONS:\n  {}",
        gone.join("\n  ")
    );
}

// -- The execution gate ---------------------------------------------------

/// Every corpus file must run, unless it is a negative test or listed breakage.
/// This is what puts the corpus into CI at all.
#[test]
fn every_iris_test_runs() {
    let broken: Vec<&str> = KNOWN_BROKEN.iter().map(|(f, _)| *f).collect();
    let must_fail: Vec<&str> = MUST_FAIL.iter().map(|(f, _)| *f).collect();

    let mut failures = Vec::new();
    for f in corpus() {
        if broken.contains(&f.as_str()) || must_fail.contains(&f.as_str()) {
            continue;
        }
        match run(&f) {
            (Some(0), _) => {}
            (Some(c), out) => failures.push(format!(
                "{} exited {}: {}",
                f,
                c,
                out.lines().find(|l| l.contains("error")).unwrap_or("").trim()
            )),
            (None, _) => failures.push(format!("{} was killed by a signal (crash)", f)),
        }
    }
    assert!(
        failures.is_empty(),
        "corpus files that should run but did not:\n  {}",
        failures.join("\n  ")
    );
}

/// A negative test that starts passing is as much a defect as a positive test
/// that starts failing: it means the check it guards has been lost.
#[test]
fn negative_tests_still_fail() {
    let mut wrong = Vec::new();
    for (f, why) in MUST_FAIL {
        if !Path::new("tests").join(f).exists() {
            wrong.push(format!("{} is listed in MUST_FAIL but does not exist", f));
            continue;
        }
        if let (Some(0), _) = run(f) {
            wrong.push(format!("{} compiled and ran, but {}", f, why));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n  "));
}

/// The two backends must compute the same answer.
///
/// The gate used to run only the native path, so an interpreter-only failure
/// escaped CI entirely -- and the backends genuinely disagree in both
/// directions (known-issues #52). Asserting they *agree* is a stronger and
/// cheaper statement than asserting each passes separately: it catches a
/// regression in either one, and it caught `to_str` on an option printing a raw
/// address natively while the interpreter printed `some(6)`.
///
/// Only stdout is compared. Diagnostics go to stderr -- which is itself
/// something this check forced: `iris_codegen:` progress lines were being
/// written to stdout, so every single file "diverged" until they were moved.
#[test]
fn the_two_backends_agree() {
    let exempt: Vec<&str> = MUST_FAIL
        .iter()
        .map(|(f, _)| *f)
        .chain(KNOWN_BROKEN.iter().map(|(f, _)| *f))
        .chain(KNOWN_DIVERGENT.iter().map(|(f, _)| *f))
        .collect();

    let mut disagree = Vec::new();
    for f in corpus() {
        if exempt.contains(&f.as_str()) {
            continue;
        }
        let native = stdout_of(&f, false);
        let interp = stdout_of(&f, true);
        if native != interp {
            disagree.push(format!(
                "{}
      native: {}
      interp: {}",
                f,
                native.lines().last().unwrap_or("(no output)"),
                interp.lines().last().unwrap_or("(no output)")
            ));
        }
    }
    assert!(
        disagree.is_empty(),
        "these files produce different output on the two backends:
  {}",
        disagree.join("
  ")
    );
}

/// The divergence list must stay honest, like the others.
#[test]
fn the_known_divergent_list_is_accurate() {
    let mut wrong = Vec::new();
    for (f, _) in KNOWN_DIVERGENT {
        if !Path::new("tests").join(f).exists() {
            wrong.push(format!("{} is listed as divergent but does not exist", f));
            continue;
        }
        let native = stdout_of(f, false);
        let interp = stdout_of(f, true);
        if native == interp {
            wrong.push(format!("{} now agrees and must come off KNOWN_DIVERGENT", f));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("
  "));
}

/// The debt register must describe reality: a file that has been fixed comes off
/// KNOWN_BROKEN, or the list stops being a to-do list.
#[test]
fn the_known_broken_list_is_accurate() {
    let mut wrong = Vec::new();
    for (f, _) in KNOWN_BROKEN {
        if !Path::new("tests").join(f).exists() {
            wrong.push(format!("{} is listed as broken but does not exist", f));
            continue;
        }
        if let (Some(0), _) = run(f) {
            wrong.push(format!("{} now runs and must come off KNOWN_BROKEN", f));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n  "));
}
