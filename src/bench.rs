//! Performance benchmarking framework for IRIS programs.
//!
//! `iris bench <file.iris>` compiles and executes a file multiple times,
//! reporting statistics: min, max, mean, median, and standard deviation.

use std::path::{Path, PathBuf};
use std::time::Instant;

/// Number of warm-up iterations before measurement.
const WARMUP_ITERS: usize = 3;
/// Default number of measured iterations.
const DEFAULT_ITERS: usize = 10;

/// Single benchmark result for one iteration.
#[derive(Debug, Clone)]
struct Sample {
    /// Parse time in microseconds.
    parse_us: u64,
    /// Lower + pass pipeline time in microseconds.
    compile_us: u64,
    /// Native execution time in microseconds.
    eval_us: u64,
    /// Total wall-clock time in microseconds.
    total_us: u64,
}

/// Aggregated statistics.
#[derive(Debug)]
struct Stats {
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    std_dev: f64,
}

fn compute_stats(values: &[f64]) -> Stats {
    let n = values.len() as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = sorted[0];
    let max = *sorted.last().expect("values is non-empty");
    let mean = sorted.iter().sum::<f64>() / n;
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    Stats {
        min,
        max,
        mean,
        median,
        std_dev,
    }
}

fn format_us(us: f64) -> String {
    if us >= 1_000_000.0 {
        format!("{:.3}s", us / 1_000_000.0)
    } else if us >= 1_000.0 {
        format!("{:.3}ms", us / 1_000.0)
    } else {
        format!("{:.1}µs", us)
    }
}

/// Run a benchmark for a single IRIS file.
fn bench_file(path: &Path, iterations: usize) -> Result<(), String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let module_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("bench");

    eprintln!("\x1b[1;36mBenchmarking\x1b[0m {}", path.display());
    eprintln!(
        "  {} warm-up iterations, {} measured iterations\n",
        WARMUP_ITERS, iterations
    );

    // Warm-up
    for _ in 0..WARMUP_ITERS {
        let _ = run_single(&source, module_name);
    }

    // Measured iterations
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        match run_single(&source, module_name) {
            Ok(s) => samples.push(s),
            Err(e) => return Err(format!("benchmark failed: {}", e)),
        }
    }

    // Report
    let parse_vals: Vec<f64> = samples.iter().map(|s| s.parse_us as f64).collect();
    let compile_vals: Vec<f64> = samples.iter().map(|s| s.compile_us as f64).collect();
    let eval_vals: Vec<f64> = samples.iter().map(|s| s.eval_us as f64).collect();
    let total_vals: Vec<f64> = samples.iter().map(|s| s.total_us as f64).collect();

    let parse = compute_stats(&parse_vals);
    let compile = compute_stats(&compile_vals);
    let eval = compute_stats(&eval_vals);
    let total = compute_stats(&total_vals);

    eprintln!(
        "  \x1b[1mPhase          Min          Mean         Median       Max          StdDev\x1b[0m"
    );
    eprintln!(
        "  {:<14} {:<12} {:<12} {:<12} {:<12} {}",
        "Parse",
        format_us(parse.min),
        format_us(parse.mean),
        format_us(parse.median),
        format_us(parse.max),
        format_us(parse.std_dev)
    );
    eprintln!(
        "  {:<14} {:<12} {:<12} {:<12} {:<12} {}",
        "Compile",
        format_us(compile.min),
        format_us(compile.mean),
        format_us(compile.median),
        format_us(compile.max),
        format_us(compile.std_dev)
    );
    eprintln!(
        "  {:<14} {:<12} {:<12} {:<12} {:<12} {}",
        "Eval",
        format_us(eval.min),
        format_us(eval.mean),
        format_us(eval.median),
        format_us(eval.max),
        format_us(eval.std_dev)
    );
    eprintln!(
        "  \x1b[1m{:<14} {:<12} {:<12} {:<12} {:<12} {}\x1b[0m",
        "Total",
        format_us(total.min),
        format_us(total.mean),
        format_us(total.median),
        format_us(total.max),
        format_us(total.std_dev)
    );

    eprintln!(
        "\n  throughput: \x1b[1;32m{:.0}\x1b[0m iterations/sec",
        1_000_000.0 / total.mean
    );

    Ok(())
}

/// Execute one full pipeline iteration (lex → parse → lower → passes → native run).
fn run_single(source: &str, module_name: &str) -> Result<Sample, String> {
    use crate::parser::lexer::Lexer;
    use crate::parser::parse::Parser;

    let t0 = Instant::now();

    // Parse
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("{}", e))?;
    let mut ast = Parser::new(&tokens)
        .parse_module()
        .map_err(|e| format!("{}", e))?;
    let t_parse = t0.elapsed();

    // Compile (lower + passes)
    let t1 = Instant::now();
    let ir = crate::compile_ast_to_module(&mut ast, module_name, None).map_err(|e| format!("{}", e))?;
    let t_compile = t1.elapsed();

    // Execute natively through the LLVM pipeline.
    let t2 = Instant::now();
    crate::eval_ir_module(&ir).map_err(|e| format!("{}", e))?;
    let t_eval = t2.elapsed();

    let t_total = t0.elapsed();

    Ok(Sample {
        parse_us: t_parse.as_micros() as u64,
        compile_us: t_compile.as_micros() as u64,
        eval_us: t_eval.as_micros() as u64,
        total_us: t_total.as_micros() as u64,
    })
}

// ---------------------------------------------------------------------------
// CLI dispatcher
// ---------------------------------------------------------------------------

/// Parse `iris bench [options] <file.iris>` and run.
pub fn run_bench_command(args: &[String]) -> Result<(), String> {
    let mut file: Option<PathBuf> = None;
    let mut iterations = DEFAULT_ITERS;
    let mut i = 2; // skip "iris bench"
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" | "-n" => {
                i += 1;
                iterations = args
                    .get(i)
                    .ok_or("--iterations requires a number")?
                    .parse::<usize>()
                    .map_err(|_| "--iterations: not a valid number")?;
            }
            "--help" | "-h" => {
                eprintln!("{}", bench_help_text());
                return Ok(());
            }
            arg if !arg.starts_with('-') => {
                file = Some(PathBuf::from(arg));
            }
            other => return Err(format!("unknown bench option: '{}'", other)),
        }
        i += 1;
    }

    let file = file.ok_or("usage: iris bench <file.iris>")?;
    if !file.exists() {
        return Err(format!("file not found: {}", file.display()));
    }

    bench_file(&file, iterations)
}

fn bench_help_text() -> &'static str {
    "IRIS Benchmark Runner\n\
     \n\
     Usage: iris bench [options] <file.iris>\n\
     \n\
     Options:\n\
       -n, --iterations <N>  Number of measured iterations (default: 10)\n\
       --help, -h             Show this help\n\
     \n\
     The benchmark measures parse, compile, and native execution times separately,\n\
     reporting min/max/mean/median/stddev for each phase.\n"
}
