//! The `lsp` and `dap` subcommands must accept the transport flag their
//! clients actually pass, and must treat a closed stdin as a clean shutdown.
//!
//! `vscode-languageclient` appends `--stdio` to the server's argv whenever
//! `TransportKind.stdio` is configured. `iris lsp` rejected the flag, so clap
//! printed a usage error and exited 1 the instant VS Code spawned it; the
//! client's first write then failed with `write EPIPE`, it retried, and after
//! five attempts reported "The IRIS Language Server server crashed 5 times in
//! the last 3 minutes. The server will not be restarted."
//!
//! Nothing about the binary or the configured path was wrong, which is what
//! made it hard to see from the editor. See known-issues #59 and #58.

use std::process::{Command, Stdio};

/// Runs the CLI with `args` and an immediately-closed stdin.
fn exit_code(args: &[&str]) -> i32 {
    let child = Command::new(env!("CARGO_BIN_EXE_iris"))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to launch the iris binary");
    child
        .wait_with_output()
        .expect("failed to wait for the iris binary")
        .status
        .code()
        .expect("process was killed by a signal")
}

#[test]
fn lsp_accepts_the_stdio_transport_flag() {
    assert_eq!(
        exit_code(&["lsp", "--stdio"]),
        0,
        "`iris lsp --stdio` must start; this is the exact command VS Code runs"
    );
}

#[test]
fn dap_accepts_the_stdio_transport_flag() {
    assert_eq!(exit_code(&["dap", "--stdio"]), 0);
}

#[test]
fn lsp_without_the_flag_still_works() {
    assert_eq!(exit_code(&["lsp"]), 0);
}

#[test]
fn dap_without_the_flag_still_works() {
    assert_eq!(exit_code(&["dap"]), 0);
}

/// A closed stdin is how an editor ends a session, not a failure. Both servers
/// used to propagate the EOF and exit 1, which a client reads as a crash.
#[test]
fn a_closed_stdin_is_a_clean_shutdown() {
    for args in [
        vec!["lsp"],
        vec!["lsp", "--stdio"],
        vec!["dap"],
        vec!["dap", "--stdio"],
    ] {
        assert_eq!(
            exit_code(&args),
            0,
            "`iris {}` must exit 0 when stdin closes",
            args.join(" ")
        );
    }
}
