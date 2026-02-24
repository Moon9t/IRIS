use std::process;

use iris::cli::{parse_args, ParseArgsResult};
use iris::diagnostics::render_error;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match parse_args(&args) {
        Ok(ParseArgsResult::Help) => {
            print!("{}", iris::cli::help_text());
            process::exit(0);
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

            match iris::compile(&source, module_name, cli.emit) {
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
