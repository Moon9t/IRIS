//! CLI argument parsing using clap derive.
//!
//! Exports `parse_args` (backward-compatible with the old signature),
//! `version_text`, `help_text`, and the `CliArgs` / `ParseArgsResult` types
//! used by `main.rs`.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::EmitKind;

// ---------------------------------------------------------------------------
// EmitKind as a clap ValueEnum (so --emit values are auto-validated)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EmitKindCli {
    Ir,
    Llvm,
    #[clap(name = "llvm-complete")]
    LlvmComplete,
    Cuda,
    #[clap(name = "cuda-ptx")]
    CudaPtx,
    Simd,
    Jit,
    #[clap(name = "pgo-instrument")]
    PgoInstrument,
    #[clap(name = "pgo-optimize")]
    PgoOptimize,
    Graph,
    Onnx,
    #[clap(name = "onnx-binary")]
    OnnxBinary,
    Eval,
    Binary,
    #[clap(name = "tensorrt")]
    TensorRt,
}

impl From<EmitKindCli> for EmitKind {
    fn from(val: EmitKindCli) -> Self {
        match val {
            EmitKindCli::Ir => EmitKind::Ir,
            EmitKindCli::Llvm => EmitKind::Llvm,
            EmitKindCli::LlvmComplete => EmitKind::LlvmComplete,
            EmitKindCli::Cuda => EmitKind::Cuda,
            EmitKindCli::CudaPtx => EmitKind::CudaPtx,
            EmitKindCli::Simd => EmitKind::Simd,
            EmitKindCli::Jit => EmitKind::Jit,
            EmitKindCli::PgoInstrument => EmitKind::PgoInstrument,
            EmitKindCli::PgoOptimize => EmitKind::PgoOptimize,
            EmitKindCli::Graph => EmitKind::Graph,
            EmitKindCli::Onnx => EmitKind::Onnx,
            EmitKindCli::OnnxBinary => EmitKind::OnnxBinary,
            EmitKindCli::Eval => EmitKind::Eval,
            EmitKindCli::Binary => EmitKind::Binary,
            EmitKindCli::TensorRt => EmitKind::TensorRt,
        }
    }
}

// ---------------------------------------------------------------------------
// Clap CLI definition
// ---------------------------------------------------------------------------

/// IRIS — Intermediate Representation for Intelligent Systems compiler.
#[derive(Parser)]
#[command(name = "iris", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Output kind
    #[arg(long, value_enum, default_value_t = EmitKindCli::Ir)]
    pub emit: EmitKindCli,

    /// Write output to <file> instead of stdout
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Target preset/triple for LLVM and native builds
    #[arg(long = "target")]
    pub target: Option<String>,

    /// Dump IR to stderr after this pass
    #[arg(long = "dump-ir-after")]
    pub dump_ir_after: Option<String>,

    /// Legacy interpreter guardrail (max steps, default: 1 000 000)
    #[arg(long = "max-steps", default_value_t = 1_000_000)]
    pub max_steps: usize,

    /// Legacy interpreter guardrail (max call depth, default: 500)
    #[arg(long = "max-depth", default_value_t = 500)]
    pub max_depth: usize,

    /// Disable incremental compilation cache
    #[arg(long = "no-cache")]
    pub no_cache: bool,

    /// Run with sandboxed security (deny fs/network/ffi/process)
    #[arg(long = "sandbox")]
    pub sandbox: bool,

    /// Require every effectful function to declare an `effect` clause that
    /// covers what it does; a violation fails the build
    #[arg(long = "strict-effects")]
    pub strict_effects: bool,

    /// Input file
    pub file: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Build a native binary (same as --emit binary)
    Build {
        /// Input file
        file: Option<PathBuf>,
    },
    /// Build and run the binary
    Run {
        /// Input file
        file: Option<PathBuf>,
    },
    /// Start an interactive REPL session
    Repl,
    /// Start the LSP server (JSON-RPC on stdin/stdout)
    Lsp,
    /// Start the DAP debug adapter (JSON-RPC on stdin/stdout)
    Dap,
    /// Package manager commands (all remaining args passthrough)
    #[command(trailing_var_arg = true)]
    Pkg {
        /// Subcommand and arguments for the package manager
        args: Vec<String>,
    },
    /// Run performance benchmarks
    Bench {
        /// Input file
        file: Option<PathBuf>,
    },
    /// Run the profiler
    Profile {
        /// Input file
        file: Option<PathBuf>,
    },
    /// Discover and run test_ functions
    Test {
        /// Input file (optional — scans current directory)
        file: Option<PathBuf>,
        /// Filter tests by substring
        #[arg(long = "filter")]
        filter: Option<String>,
        /// Disable colored output
        #[arg(long = "no-color")]
        no_color: bool,
    },
    /// Show detailed explanation for an error code
    Explain {
        /// Error code (e.g. E0100)
        code: Option<String>,
    },
    /// Self-upgrade the IRIS compiler
    Upgrade {
        /// Check for available updates without installing
        #[arg(short = 'c', long = "check")]
        check: bool,
        /// Skip confirmation prompts
        #[arg(short = 'y', long = "yes")]
        yes: bool,
        /// Force reinstall even if up-to-date
        #[arg(short = 'f', long = "force")]
        force: bool,
    },
    /// Install a package or all dependencies from iris.toml
    Install {
        /// Git URL of the package to install (omit to install all from iris.toml)
        url: Option<String>,
    },
    /// Download and configure toolchain dependencies
    Setup,
    /// Generate HTML documentation from doc comments
    Docs {
        /// Input file
        file: Option<PathBuf>,
        /// Write output to <file> instead of stdout
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },
    /// Format IRIS source files
    Fmt {
        /// Input file (optional — formats all *.iris in current directory)
        file: Option<PathBuf>,
        /// Check if formatting is needed without modifying files (exit 1 if changes needed)
        #[arg(short = 'c', long = "check")]
        check: bool,
    },
}

// ---------------------------------------------------------------------------
// Public API — backward-compatible with old parse_args signature
// ---------------------------------------------------------------------------

/// Fully-parsed CLI arguments for a compilation request.
#[derive(Debug)]
pub struct CliArgs {
    pub path: PathBuf,
    pub emit: EmitKind,
    pub output: Option<PathBuf>,
    pub run_after_build: bool,
    pub target: Option<String>,
    pub dump_ir_after: Option<String>,
    pub max_steps: usize,
    pub max_depth: usize,
    pub no_cache: bool,
    pub sandbox: bool,
    pub strict_effects: bool,
}

/// Result of `parse_args` — backward-compatible with the old API.
#[derive(Debug)]
pub enum ParseArgsResult {
    Args(CliArgs),
    Help,
    Version,
    Repl,
    Lsp,
    Dap,
    Pkg {
        /// Raw arguments after `pkg` subcommand
        args: Vec<String>,
    },
    Bench,
    Profile,
    Test {
        /// Input file (optional — scans current directory)
        file: Option<PathBuf>,
        /// Filter tests by substring
        filter: Option<String>,
        /// Disable colored output
        no_color: bool,
    },
    Explain(Option<String>),
    Upgrade { check: bool, yes: bool, force: bool },
    Install { url: Option<String> },
    Setup,
    Fmt {
        file: Option<PathBuf>,
        check: bool,
    },
    Docs {
        file: Option<PathBuf>,
        output: Option<PathBuf>,
    },
}

/// Parses command-line arguments.
///
/// Uses `clap` internally, but returns the same `ParseArgsResult` enum
/// that `main.rs` already matches on.
pub fn parse_args(args: &[String]) -> Result<ParseArgsResult, String> {
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => {
            if e.kind() == clap::error::ErrorKind::DisplayHelp {
                return Ok(ParseArgsResult::Help);
            }
            if e.kind() == clap::error::ErrorKind::DisplayVersion {
                return Ok(ParseArgsResult::Version);
            }
            return Err(e.to_string());
        }
    };

    match cli.command {
        Some(Command::Build { file }) => {
            let path = file.or(cli.file).ok_or_else(|| "no input file specified".to_owned())?;
            Ok(ParseArgsResult::Args(CliArgs {
                emit: EmitKind::Binary,
                path,
                output: cli.output,
                run_after_build: false,
                target: cli.target,
                dump_ir_after: cli.dump_ir_after,
                max_steps: cli.max_steps,
                max_depth: cli.max_depth,
                no_cache: cli.no_cache,
                sandbox: cli.sandbox,
                strict_effects: cli.strict_effects,
            }))
        }
        Some(Command::Run { file }) => {
            let path = file.or(cli.file).ok_or_else(|| "no input file specified".to_owned())?;
            Ok(ParseArgsResult::Args(CliArgs {
                emit: EmitKind::Binary,
                path,
                output: cli.output,
                run_after_build: true,
                target: cli.target,
                dump_ir_after: cli.dump_ir_after,
                max_steps: cli.max_steps,
                max_depth: cli.max_depth,
                no_cache: cli.no_cache,
                sandbox: cli.sandbox,
                strict_effects: cli.strict_effects,
            }))
        }
        Some(Command::Repl) => Ok(ParseArgsResult::Repl),
        Some(Command::Lsp) => Ok(ParseArgsResult::Lsp),
        Some(Command::Dap) => Ok(ParseArgsResult::Dap),
        Some(Command::Pkg { args }) => Ok(ParseArgsResult::Pkg { args }),
        Some(Command::Bench { file }) => {
            if file.is_some() {
                // bench <file.iris> — treat as a compilation request
                let path = file.unwrap();
                Ok(ParseArgsResult::Args(CliArgs {
                    emit: EmitKind::Eval,
                    path,
                    output: cli.output,
                    run_after_build: false,
                    target: cli.target,
                    dump_ir_after: cli.dump_ir_after,
                    max_steps: cli.max_steps,
                    max_depth: cli.max_depth,
                    no_cache: cli.no_cache,
                    sandbox: cli.sandbox,
                    strict_effects: cli.strict_effects,
                }))
            } else {
                Ok(ParseArgsResult::Bench)
            }
        }
        Some(Command::Profile { file }) => {
            if file.is_some() {
                let path = file.unwrap();
                Ok(ParseArgsResult::Args(CliArgs {
                    emit: EmitKind::Eval,
                    path,
                    output: cli.output,
                    run_after_build: false,
                    target: cli.target,
                    dump_ir_after: cli.dump_ir_after,
                    max_steps: cli.max_steps,
                    max_depth: cli.max_depth,
                    no_cache: cli.no_cache,
                    sandbox: cli.sandbox,
                    strict_effects: cli.strict_effects,
                }))
            } else {
                Ok(ParseArgsResult::Profile)
            }
        }
        Some(Command::Test { file, filter, no_color }) => {
            Ok(ParseArgsResult::Test { file, filter, no_color })
        }
        Some(Command::Explain { code }) => Ok(ParseArgsResult::Explain(code)),
        Some(Command::Upgrade { check, yes, force }) => {
            Ok(ParseArgsResult::Upgrade { check, yes, force })
        }
        Some(Command::Setup) => Ok(ParseArgsResult::Setup),
        Some(Command::Install { url }) => Ok(ParseArgsResult::Install { url }),
        Some(Command::Fmt { file, check }) => Ok(ParseArgsResult::Fmt { file, check }),
        Some(Command::Docs { file, output }) => Ok(ParseArgsResult::Docs { file, output }),
        None => {
            // No subcommand — treat as direct compilation request
            let path = cli.file.ok_or_else(|| "no input file specified".to_owned())?;
            Ok(ParseArgsResult::Args(CliArgs {
                path,
                emit: cli.emit.into(),
                output: cli.output,
                run_after_build: false,
                target: cli.target,
                dump_ir_after: cli.dump_ir_after,
                max_steps: cli.max_steps,
                max_depth: cli.max_depth,
                no_cache: cli.no_cache,
                sandbox: cli.sandbox,
                strict_effects: cli.strict_effects,
            }))
        }
    }
}

/// Returns the version string for the CLI (GCC-style verbose output).
pub fn version_text() -> String {
    let version = env!("CARGO_PKG_VERSION");
    let build_date = option_env!("IRIS_BUILD_DATE").unwrap_or("unknown");
    let target = option_env!("IRIS_TARGET").unwrap_or("unknown");
    let host = option_env!("IRIS_HOST").unwrap_or("unknown");
    let profile = option_env!("IRIS_PROFILE").unwrap_or("unknown");
    let opt_level = option_env!("IRIS_OPT_LEVEL").unwrap_or("unknown");
    let git_hash = option_env!("IRIS_GIT_HASH").unwrap_or("unknown");
    let git_hash_short = option_env!("IRIS_GIT_HASH_SHORT").unwrap_or("unknown");
    let git_branch = option_env!("IRIS_GIT_BRANCH").unwrap_or("unknown");
    let git_dirty = option_env!("IRIS_GIT_DIRTY").unwrap_or("false");
    let rustc_ver = option_env!("IRIS_RUSTC_VERSION").unwrap_or("unknown");

    // Detect thread model.
    let thread_model = if cfg!(target_family = "windows") {
        "win32"
    } else {
        "posix"
    };

    let dirty_flag = if git_dirty == "true" {
        " (modified)"
    } else {
        ""
    };

    format!(
        "iris {version} ({git_hash_short} {build_date}){dirty}\n\
         IRIS — Intermediate Representation for Intelligent Systems\n\
         Copyright (C) 2024-2026 Moon & IRIS Project Contributors\n\
         License: GPL-2.0-or-later <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>\n\
         This is free software; you can redistribute it and/or modify it under\n\
         the terms of the GNU General Public License v2 (or later).\n\
         There is NO WARRANTY, to the extent permitted by law.\n\
         \n\
         Compiler:\n\
           Version:       {version}\n\
           Git commit:    {git_hash}\n\
           Git branch:    {git_branch}\n\
           Build date:    {build_date}\n\
         \n\
         Platform:\n\
           Target:        {target}\n\
           Host:          {host}\n\
           Thread model:  {thread_model}\n\
         \n\
         Build:\n\
           Profile:       {profile}\n\
           Opt level:     {opt_level}\n\
           Rust edition:  2021\n\
           Built with:    {rustc_ver}\n",
        version = version,
        git_hash_short = git_hash_short,
        git_hash = git_hash,
        git_branch = git_branch,
        build_date = build_date,
        dirty = dirty_flag,
        target = target,
        host = host,
        profile = profile,
        opt_level = opt_level,
        thread_model = thread_model,
        rustc_ver = rustc_ver,
    )
}

/// Returns the usage/help text for the CLI.
pub fn help_text() -> &'static str {
    "IRIS compiler\n\
     Usage: iris [subcommand] [options] <file.iris>\n\
     \n\
     Subcommands:\n\
       build                 Build native binary (same as --emit binary)\n\
       run                   Build and run the binary\n\
       test [file.iris]      Discover and run test_ functions (--filter <substr> --no-color)\n\
       install [url]         Install dependencies or a package from a Git URL\n\
       fmt [file.iris]       Format source files (--check to verify without modifying)\n\
       repl                  Start an interactive REPL session\n\
       lsp                   Start the LSP server (JSON-RPC on stdin/stdout)\n\
       dap                   Start the DAP debug adapter (JSON-RPC on stdin/stdout)\n\
       pkg <cmd>             Package manager (init, add, remove, install, list, build, run)\n\
       bench <file.iris>     Run performance benchmarks on a file\n\
       explain [code]        Show detailed explanation for an error code (e.g. E0100)\n\
        upgrade               Self-upgrade the IRIS compiler to the latest version\n\
        setup                 Download and configure toolchain dependencies\n\
        docs [file.iris]      Generate HTML documentation from doc comments\n\
     \n\
     Options:\n\
     --emit <kind>         Output kind: ir (default), llvm, llvm-complete, cuda, cuda-ptx, simd,\n\
                              jit, pgo-instrument, pgo-optimize, graph, onnx, onnx-binary,\n\
                              eval, binary, tensorrt\n\
       -o <file>             Write output to <file> instead of stdout\n\
       --target <triple>     Target preset/triple for llvm/binary outputs (e.g. linux-arm64)\n\
       --dump-ir-after <p>   Dump IR to stderr after pass <p> completes\n\
       --max-steps <n>       Legacy interpreter guardrail (ignored for native build/run/eval/jit)\n\
       --max-depth <n>       Legacy call-depth guardrail (ignored for native build/run/eval/jit)\n\
       --no-cache            Disable incremental compilation cache\n\
       --sandbox             Run with sandboxed security (deny fs/network/ffi/process)\n\
       --strict-effects      Require `effect` clauses that cover what each function\n\
                             does; an effect violation fails the build\n\
       --help, -h            Print this help and exit\n\
       --version, -V         Print version and exit\n"
}
