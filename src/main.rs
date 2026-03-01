use std::path::PathBuf;
use std::process;

use iris::cli::{parse_args, ParseArgsResult};

/// 64 MB stack — Windows default is only 1 MB, which overflows on deeply
/// nested IRIS expressions during recursive IR lowering.
const STACK_SIZE: usize = 64 * 1024 * 1024;

fn main() {
    let builder = std::thread::Builder::new().stack_size(STACK_SIZE);
    let handler = builder
        .spawn(run)
        .expect("failed to spawn main thread with enlarged stack");
    if let Err(e) = handler.join() {
        eprintln!("error: {:?}", e);
        process::exit(1);
    }
}

fn run() {
    let args: Vec<String> = std::env::args().collect();

    match parse_args(&args) {
        Ok(ParseArgsResult::Help) => {
            print!("{}", iris::cli::help_text());
            process::exit(0);
        }
        Ok(ParseArgsResult::Version) => {
            print!("{}", iris::cli::version_text());
            process::exit(0);
        }
        Ok(ParseArgsResult::Repl) => {
            run_repl();
        }
        Ok(ParseArgsResult::Lsp) => {
            if let Err(e) = iris::lsp::run_lsp_server() {
                eprintln!("LSP server error: {}", e);
                process::exit(1);
            }
        }
        Ok(ParseArgsResult::Dap) => {
            if let Err(e) = iris::dap::run_dap_server() {
                eprintln!("DAP server error: {}", e);
                process::exit(1);
            }
        }
        Ok(ParseArgsResult::Args(cli)) => {
            if cli.emit == iris::EmitKind::Binary {
                let module = match iris::compile_file_to_module_with_opts(&cli.path, cli.dump_ir_after.as_deref()) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        process::exit(1);
                    }
                };
                let output_path = cli.output.unwrap_or_else(|| {
                    PathBuf::from(format!("iris_out{}", std::env::consts::EXE_SUFFIX))
                });
                match iris::codegen::build_binary(&module, &output_path) {
                    Ok(path) => {
                        eprintln!("wrote binary: {}", path.display());
                        if cli.run_after_build {
                            // Canonicalize so Command finds the binary in the
                            // current directory on Windows (relative paths
                            // without ".\" are not searched).
                            let run_path = std::fs::canonicalize(&path)
                                .unwrap_or_else(|_| path.clone());
                            let status = std::process::Command::new(&run_path)
                                .status()
                                .unwrap_or_else(|e| {
                                    eprintln!("error: could not run binary: {}", e);
                                    process::exit(1);
                                });
                            process::exit(status.code().unwrap_or(1));
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {}", e);
                        process::exit(1);
                    }
                }
                return;
            }

            match iris::compile_file_with_full_opts(&cli.path, cli.emit, cli.max_steps, cli.max_depth, cli.dump_ir_after.as_deref()) {
                Ok(output) => {
                    if let Some(out_path) = cli.output {
                        if let Err(e) = std::fs::write(&out_path, &output) {
                            eprintln!("error: cannot write '{}': {}", out_path.display(), e);
                            process::exit(1);
                        }
                    } else {
                        print!("{}", output);
                    }
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
        }
        Err(msg) => {
            eprintln!("error: {}", msg);
            eprintln!("{}", iris::cli::help_text());
            process::exit(1);
        }
    }
}

fn run_repl() {
    use std::io::{BufRead, Write};
    let mut repl = iris::ReplState::new();
    let version = env!("CARGO_PKG_VERSION");
    eprintln!("IRIS {} REPL", version);
    eprintln!("  :help for commands · :quit to exit · Ctrl+D to exit");
    eprintln!();
    let stdin = std::io::stdin();
    let mut accumulator = String::new();
    let mut brace_depth: i32 = 0;

    loop {
        // Show continuation prompt when inside a multi-line block.
        if brace_depth > 0 {
            eprint!("... ");
        } else {
            eprint!(">> ");
        }
        let _ = std::io::stderr().flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => {
                // EOF (Ctrl+D) — flush any pending accumulator then exit.
                if !accumulator.trim().is_empty() {
                    run_repl_input(&mut repl, accumulator.trim());
                }
                break;
            }
            Ok(_) => {}
        }

        // Track brace depth for multiline input.
        for ch in line.chars() {
            if ch == '{' { brace_depth += 1; }
            if ch == '}' { brace_depth -= 1; }
        }
        accumulator.push_str(&line);

        // Only evaluate when braces are balanced.
        if brace_depth <= 0 {
            brace_depth = 0;
            let input = accumulator.trim().to_owned();
            accumulator.clear();
            if !input.is_empty() {
                run_repl_input(&mut repl, &input);
            }
        }
    }
}

fn run_repl_input(repl: &mut iris::ReplState, input: &str) {
    match repl.eval(input) {
        Ok(s) if !s.is_empty() => println!("{}", s),
        Ok(_) => {}
        Err(e) => eprintln!("error: {}", e),
    }
}
