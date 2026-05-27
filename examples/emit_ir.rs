use iris::{compile_multi, EmitKind};

fn main() {
    let src = r#"
bring std.http_server

def main() -> str {
    var r = router_new()
    r = router_add(r, "GET",  "/health",       "health_handler")
    r = router_add(r, "POST", "/api/predict",  "predict_handler")
    r = router_add(r, "GET",  "/api/models",   "list_models_handler")
    router_match(r, "POST", "/api/predict")
}
"#;

    match compile_multi(&[("main", src)], "main", EmitKind::Llvm) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("compile error: {}", e),
    }
}
