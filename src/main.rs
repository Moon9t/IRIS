use std::path::PathBuf;
use std::process;

use iris::cli::{parse_args, ParseArgsResult};
use iris::diagnostics::render_error;
use iris::parser::lexer::Lexer;
use iris::parser::parse::Parser;

fn main() {
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
            let source = std::fs::read_to_string(&cli.path).unwrap_or_else(|e| {
                eprintln!("error: cannot read '{}': {}", cli.path.display(), e);
                process::exit(1);
            });

            let module_name = cli.path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("module");

            if cli.emit == iris::EmitKind::Binary {
                let tokens = match Lexer::new(&source).tokenize() {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("{}", render_error(&source, &e.into()));
                        process::exit(1);
                    }
                };
                let ast = match Parser::new(&tokens).parse_module() {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("{}", render_error(&source, &e.into()));
                        process::exit(1);
                    }
                };
                let module = match iris::compile_ast_to_module(&ast, module_name, cli.dump_ir_after.as_deref()) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("{}", render_error(&source, &e));
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
                            let status = std::process::Command::new(&path)
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

            match iris::compile_with_full_opts(&source, module_name, cli.emit, cli.max_steps, cli.max_depth, cli.dump_ir_after.as_deref()) {
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
                    eprintln!("{}", render_error(&source, &e));
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
    eprintln!("IRIS REPL  (type :quit to exit, :reset to clear state, :help for help)");
    let stdin = std::io::stdin();
    loop {
        eprint!(">> ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break, // EOF
            Ok(_) => {}
        }
        let trimmed = line.trim();
        match trimmed {
            ":quit" | ":q" | ":exit" => break,
            ":reset" => { repl.reset(); eprintln!("state cleared"); }
            ":help" => eprintln!(
                "Commands:\n  :quit  — exit\n  :reset — clear accumulated state\n  :help  — this message\n\
                 Syntax:\n  val x = expr       — bind variable\n  def f(...) -> T {{ }} — define function\n  expr               — evaluate expression"
            ),
            "" => {}
            input => match repl.eval(input) {
                Ok(s) if !s.is_empty() => println!("{}", s),
                Ok(_) => {}
                Err(e) => eprintln!("error: {}", e),
            },
        }
    }
}
