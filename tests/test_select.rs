//! Integration tests for the channel `select()` builtin.

use iris::{compile, EmitKind};

#[test]
fn test_select_first_ready() {
    let src = r#"
def f() -> i64 {
    val ch1 = channel()
    val ch2 = channel()
    send(ch2, 42);
    // select(ch1, ch2) should return 1 because ch2 is ready and ch1 is empty
    select(ch1, ch2)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "1",
        "select(ch1, ch2) should return 1, got: {}",
        out.trim()
    );
}

#[test]
fn test_select_none_ready() {
    let src = r#"
def f() -> i64 {
    val ch1 = channel()
    val ch2 = channel()
    // select(ch1, ch2) should return -1 because both are empty
    select(ch1, ch2)
}
"#;
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    assert_eq!(
        out.trim(),
        "-1",
        "select(ch1, ch2) should return -1, got: {}",
        out.trim()
    );
}

#[test]
fn test_select_native_binary() {
    let src = r#"
def main() -> i64 {
    val ch1 = channel()
    val ch2 = channel()
    send(ch2, 99);
    val r = select(ch1, ch2)
    if r == 1 { 0 } else { 1 }
}
"#;
    let tokens = iris::parser::lexer::Lexer::new(src).tokenize().expect("lex");
    let mut ast = iris::parser::parse::Parser::new(&tokens).parse_module().expect("parse");
    let module = iris::compile_ast_to_module(&mut ast, "select_native", None).expect("compile to module");
    
    let out = std::env::temp_dir().join(format!("iris_test_select_{}", std::process::id()));
    let out_path = if std::env::consts::EXE_SUFFIX.is_empty() {
        out
    } else {
        out.with_extension(std::env::consts::EXE_SUFFIX.trim_start_matches('.'))
    };

    match iris::codegen::build_binary(&module, &out_path) {
        Ok(path) => {
            let status = std::process::Command::new(&path)
                .status()
                .expect("run binary");
            let _ = std::fs::remove_file(&path);
            assert!(status.success(), "select binary should exit with code 0 (meaning select returned 1)");
        }
        Err(e) => {
            let msg = format!("{}", e);
            if !msg.contains("clang") && !msg.contains("binary") {
                panic!("unexpected build error: {}", e);
            }
        }
    }
}
