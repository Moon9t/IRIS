//! Integration tests for native binary build and run.
//!
//! Requires clang in PATH for full build; if clang is missing, build_binary returns
//! an error and the test still passes (we only assert the pipeline runs).

use std::path::PathBuf;

use iris::codegen::build_binary;
use iris::parser::lexer::Lexer;
use iris::parser::parse::Parser;

#[test]
fn test_binary_build_hello() {
    let src = r#"
def main() -> i64 {
    print("Hello, IRIS!");
    0
}
"#;
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let ast = Parser::new(&tokens).parse_module().expect("parse");
    let module = iris::compile_ast_to_module(&ast, "hello", None).expect("compile to module");

    assert!(
        module.functions().iter().any(|f| f.name == "main" && f.params.is_empty()),
        "module should have main() as entry"
    );

    let out = std::env::temp_dir().join(format!("iris_test_hello_{}", std::process::id()));
    let out_path = if std::env::consts::EXE_SUFFIX.is_empty() {
        out
    } else {
        out.with_extension(std::env::consts::EXE_SUFFIX.trim_start_matches('.'))
    };

    match build_binary(&module, &out_path) {
        Ok(path) => {
            let status = std::process::Command::new(&path)
                .status()
                .expect("run binary");
            let _ = std::fs::remove_file(&path);
            assert!(status.success(), "binary should exit 0");
        }
        Err(e) => {
            let msg = format!("{}", e);
            if !msg.contains("clang") && !msg.contains("binary") {
                panic!("unexpected build error: {}", e);
            }
        }
    }
}
