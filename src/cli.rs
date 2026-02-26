//! CLI argument parsing, exported from the library so integration tests can exercise it.

use std::path::PathBuf;

use crate::EmitKind;

/// Fully-parsed CLI arguments for a compilation request.
#[derive(Debug)]
pub struct CliArgs {
    pub path: PathBuf,
    pub emit: EmitKind,
    /// Write output to this file instead of stdout.
    pub output: Option<PathBuf>,
    /// Dump IR to stderr immediately after this pass completes.
    pub dump_ir_after: Option<String>,
    /// Maximum interpreter step count before aborting (default: 1 000 000).
    pub max_steps: usize,
    /// Maximum interpreter call depth before aborting (default: 500).
    pub max_depth: usize,
}

/// Result of `parse_args`.
#[derive(Debug)]
pub enum ParseArgsResult {
    /// Normal compilation/evaluation request.
    Args(CliArgs),
    /// `--help` was present; caller should print usage and exit 0.
    Help,
}

/// Parses command-line arguments (the full `std::env::args()` slice including `argv[0]`).
pub fn parse_args(args: &[String]) -> Result<ParseArgsResult, String> {
    let mut emit = EmitKind::Ir;
    let mut path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut dump_ir_after: Option<String> = None;
    let mut max_steps: usize = 1_000_000;
    let mut max_depth: usize = 500;
    let mut i = 1usize;

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => return Ok(ParseArgsResult::Help),
            "--emit" => {
                i += 1;
                let kind = args
                    .get(i)
                    .ok_or_else(|| "--emit requires an argument".to_owned())?;
                emit = match kind.as_str() {
                    "ir" => EmitKind::Ir,
                    "llvm" => EmitKind::Llvm,
                    "llvm-complete" => EmitKind::LlvmComplete,
                    "cuda" => EmitKind::Cuda,
                    "simd" => EmitKind::Simd,
                    "jit" => EmitKind::Jit,
                    "pgo-instrument" => EmitKind::PgoInstrument,
                    "pgo-optimize" => EmitKind::PgoOptimize,
                    "graph" => EmitKind::Graph,
                    "onnx" => EmitKind::Onnx,
                    "onnx-binary" => EmitKind::OnnxBinary,
                    "eval" => EmitKind::Eval,
                    other => {
                        return Err(format!(
                            "unknown emit kind: '{}' (valid: ir, llvm, llvm-complete, cuda, simd, jit, pgo-instrument, pgo-optimize, graph, onnx, onnx-binary, eval)",
                            other
                        ))
                    }
                };
            }
            "-o" => {
                i += 1;
                let file = args
                    .get(i)
                    .ok_or_else(|| "-o requires an argument".to_owned())?;
                output = Some(PathBuf::from(file));
            }
            "--dump-ir-after" => {
                i += 1;
                let name = args
                    .get(i)
                    .ok_or_else(|| "--dump-ir-after requires an argument".to_owned())?;
                dump_ir_after = Some(name.clone());
            }
            "--max-steps" => {
                i += 1;
                let n = args
                    .get(i)
                    .ok_or_else(|| "--max-steps requires an argument".to_owned())?;
                max_steps = n.parse::<usize>().map_err(|_| {
                    format!("--max-steps: '{}' is not a valid positive integer", n)
                })?;
            }
            "--max-depth" => {
                i += 1;
                let n = args
                    .get(i)
                    .ok_or_else(|| "--max-depth requires an argument".to_owned())?;
                max_depth = n.parse::<usize>().map_err(|_| {
                    format!("--max-depth: '{}' is not a valid positive integer", n)
                })?;
            }
            arg if !arg.starts_with('-') => {
                path = Some(PathBuf::from(arg));
            }
            other => return Err(format!("unknown argument: '{}'", other)),
        }
        i += 1;
    }

    let path = path.ok_or_else(|| "no input file specified".to_owned())?;
    Ok(ParseArgsResult::Args(CliArgs { path, emit, output, dump_ir_after, max_steps, max_depth }))
}

/// Returns the usage/help text for the CLI.
pub fn help_text() -> &'static str {
    "IRIS compiler\n\
     Usage: iris [options] <file.iris>\n\
     \n\
     Options:\n\
       --emit <kind>         Output kind: ir (default), llvm, llvm-complete, cuda, simd,\n\
                             jit, pgo-instrument, pgo-optimize, graph, onnx, onnx-binary, eval\n\
       -o <file>             Write output to <file> instead of stdout\n\
       --dump-ir-after <p>   Dump IR to stderr after pass <p> completes\n\
       --max-steps <n>       Max interpreter steps before abort (default: 1000000)\n\
       --max-depth <n>       Max call depth before abort (default: 500)\n\
       --help, -h            Print this help and exit\n"
}
