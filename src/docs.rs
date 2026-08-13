use crate::parser::ast::{AstFunction, AstModule};
use crate::parser::lexer::Lexer;
use crate::parser::parse::Parser;

/// A documented item extracted from the AST.
pub enum DocItem {
    Function {
        name: String,
        is_pub: bool,
        doc: Option<String>,
        signature: String,
    },
    Struct {
        name: String,
        is_pub: bool,
        doc: Option<String>,
        fields: Vec<(String, String)>,
    },
    Enum {
        name: String,
        is_pub: bool,
        doc: Option<String>,
        variants: Vec<(String, Option<String>)>,
    },
    Trait {
        name: String,
        doc: Option<String>,
        methods: Vec<(String, String)>,
    },
    Const {
        name: String,
        is_pub: bool,
        doc: Option<String>,
        ty: Option<String>,
    },
    TypeAlias {
        name: String,
        is_pub: bool,
        doc: Option<String>,
        ty: String,
    },
    Effect {
        name: String,
        doc: Option<String>,
        operations: Vec<(String, String)>,
    },
}

fn format_type(ty: &crate::parser::ast::AstType) -> String {
    use crate::parser::ast::AstType;
    match ty {
        AstType::Scalar(k, _) => format!("{:?}", k).to_lowercase(),
        AstType::Named(n, _) => n.clone(),
        AstType::Tuple(elems, _) => {
            let inner: Vec<_> = elems.iter().map(format_type).collect();
            format!("({})", inner.join(", "))
        }
        AstType::Array { elem, len, .. } => format!("[{}; {}]", format_type(elem), len),
        AstType::Option(inner, _) => format!("option<{}>", format_type(inner)),
        AstType::Result(ok, err, _) => format!("result<{}, {}>", format_type(ok), format_type(err)),
        AstType::List(inner, _) => format!("list<{}>", format_type(inner)),
        AstType::Map(k, v, _) => format!("map<{}, {}>", format_type(k), format_type(v)),
        AstType::Chan(inner, _) => format!("chan<{}>", format_type(inner)),
        AstType::Fn { params, ret, .. } => {
            let p: Vec<_> = params.iter().map(format_type).collect();
            format!("({}) -> {}", p.join(", "), format_type(ret))
        }
        AstType::Generic { name, args, .. } => {
            let a: Vec<_> = args.iter().map(format_type).collect();
            format!("{}<{}>", name, a.join(", "))
        }
        AstType::Ref(inner, _) => format!("&{}", format_type(inner)),
        AstType::RefMut(inner, _) => format!("&mut {}", format_type(inner)),
        AstType::DynTrait { trait_name, .. } => format!("dyn {}", trait_name),
        _ => "...".to_string(),
    }
}

fn format_params(params: &[crate::parser::ast::AstParam]) -> String {
    params
        .iter()
        .map(|p| {
            if p.default.is_some() {
                format!("{}: {} = ...", p.name.name, format_type(&p.ty))
            } else {
                format!("{}: {}", p.name.name, format_type(&p.ty))
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_fn_signature(f: &AstFunction) -> String {
    let async_prefix = if f.is_async { "async " } else { "" };
    let pub_prefix = if f.is_pub { "pub " } else { "" };
    let type_params = if f.type_params.is_empty() {
        String::new()
    } else {
        // Simplified — just show names
        let names: Vec<_> = f
            .type_params
            .iter()
            .map(|p| match p {
                crate::parser::ast::AstGenericParam::Type(n, _, _) => n.clone(),
                crate::parser::ast::AstGenericParam::Const { name, .. } => {
                    format!("const {}", name)
                }
                crate::parser::ast::AstGenericParam::Hkt(n, _, _, _) => n.clone(),
            })
            .collect();
        format!("[{}]", names.join(", "))
    };
    let effects = if f.effects.is_empty() {
        String::new()
    } else {
        format!(" effect {}", f.effects.join(", "))
    };
    format!(
        "{}{}{}def {}{}({}) -> {}{}",
        pub_prefix,
        async_prefix,
        if f.is_pub || f.is_async { "" } else { "" },
        f.name.name,
        type_params,
        format_params(&f.params),
        format_type(&f.return_ty),
        effects,
    )
}

fn extract_items(module: &AstModule) -> Vec<DocItem> {
    let mut items = Vec::new();

    for f in &module.functions {
        items.push(DocItem::Function {
            name: f.name.name.clone(),
            is_pub: f.is_pub,
            doc: f.doc_comment.clone(),
            signature: format_fn_signature(f),
        });
    }
    for s in &module.structs {
        let fields = s
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), format_type(&f.ty)))
            .collect();
        items.push(DocItem::Struct {
            name: s.name.name.clone(),
            is_pub: s.is_pub,
            doc: s.doc_comment.clone(),
            fields,
        });
    }
    for e in &module.enums {
        let variants = e
            .variants
            .iter()
            .map(|v| {
                let payload = if v.fields.is_empty() {
                    None
                } else {
                    let tys: Vec<_> = v.fields.iter().map(format_type).collect();
                    Some(tys.join(", "))
                };
                (v.name.name.clone(), payload)
            })
            .collect();
        items.push(DocItem::Enum {
            name: e.name.name.clone(),
            is_pub: e.is_pub,
            doc: e.doc_comment.clone(),
            variants,
        });
    }
    for t in &module.traits {
        let methods = t
            .methods
            .iter()
            .map(|m| {
                let sig = format!(
                    "def {}({}) -> {}",
                    m.name.name,
                    format_params(&m.params),
                    format_type(&m.return_ty)
                );
                (m.name.name.clone(), sig)
            })
            .collect();
        items.push(DocItem::Trait {
            name: t.name.name.clone(),
            doc: t.doc_comment.clone(),
            methods,
        });
    }
    for c in &module.consts {
        items.push(DocItem::Const {
            name: c.name.name.clone(),
            is_pub: c.is_pub,
            doc: c.doc_comment.clone(),
            ty: c.ty.as_ref().map(format_type),
        });
    }
    for ta in &module.type_aliases {
        items.push(DocItem::TypeAlias {
            name: ta.name.clone(),
            is_pub: ta.is_pub,
            doc: ta.doc_comment.clone(),
            ty: format_type(&ta.ty),
        });
    }
    for eff in &module.effects {
        let operations = eff
            .operations
            .iter()
            .map(|op| {
                let sig = format!(
                    "def {}({}) -> {}",
                    op.name.name,
                    format_params(&op.params),
                    format_type(&op.ret_ty)
                );
                (op.name.name.clone(), sig)
            })
            .collect();
        items.push(DocItem::Effect {
            name: eff.name.name.clone(),
            doc: eff.doc_comment.clone(),
            operations,
        });
    }

    items
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn render_item(item: &DocItem) -> String {
    match item {
        DocItem::Function {
            name,
            is_pub,
            doc,
            signature,
        } => {
            let pub_badge = if *is_pub {
                r#"<span class="badge pub">pub</span> "#
            } else {
                ""
            };
            let doc_html = doc
                .as_deref()
                .map(|d| format!("<p class=\"doc\">{}</p>", html_escape(d)))
                .unwrap_or_default();
            format!(
                "<div class=\"item\" id=\"fn-{0}\">\n  <h3>{1}function <code class=\"sig\">{2}</code></h3>\n  {3}\n</div>",
                html_escape(name),
                pub_badge,
                html_escape(signature),
                doc_html,
            )
        }
        DocItem::Struct {
            name,
            is_pub,
            doc,
            fields,
        } => {
            let pub_badge = if *is_pub {
                r#"<span class="badge pub">pub</span> "#
            } else {
                ""
            };
            let doc_html = doc
                .as_deref()
                .map(|d| format!("<p class=\"doc\">{}</p>", html_escape(d)))
                .unwrap_or_default();
            let field_rows: Vec<_> = fields
                .iter()
                .map(|(fn_, ft)| {
                    format!(
                        r#"<tr><td><code>{}</code></td><td><code>{}</code></td></tr>"#,
                        html_escape(fn_),
                        html_escape(ft)
                    )
                })
                .collect();
            let fields_table = if field_rows.is_empty() {
                String::new()
            } else {
                format!(
                    r#"<table class="fields">{}</table>"#,
                    field_rows.join("\n    ")
                )
            };
            format!(
                "<div class=\"item\" id=\"record-{0}\">\n  <h3>{1}record <code class=\"sig\">{0}</code></h3>\n  {2}\n  {3}\n</div>",
                html_escape(name),
                pub_badge,
                doc_html,
                fields_table,
            )
        }
        DocItem::Enum {
            name,
            is_pub,
            doc,
            variants,
        } => {
            let pub_badge = if *is_pub {
                r#"<span class="badge pub">pub</span> "#
            } else {
                ""
            };
            let doc_html = doc
                .as_deref()
                .map(|d| format!("<p class=\"doc\">{}</p>", html_escape(d)))
                .unwrap_or_default();
            let variant_rows: Vec<_> = variants
                .iter()
                .map(|(vn, payload)| {
                    let payload_str = match payload {
                        Some(p) => format!("(<code>{}</code>)", html_escape(p)),
                        None => String::new(),
                    };
                    format!(
                        r#"<tr><td><code>{}</code></td><td>{}</td></tr>"#,
                        html_escape(vn),
                        payload_str
                    )
                })
                .collect();
            let variants_table = if variant_rows.is_empty() {
                String::new()
            } else {
                format!(
                    r#"<table class="fields">{}</table>"#,
                    variant_rows.join("\n    ")
                )
            };
            format!(
                "<div class=\"item\" id=\"choice-{0}\">\n  <h3>{1}choice <code class=\"sig\">{0}</code></h3>\n  {2}\n  {3}\n</div>",
                html_escape(name),
                pub_badge,
                doc_html,
                variants_table,
            )
        }
        DocItem::Trait { name, doc, methods } => {
            let doc_html = doc
                .as_deref()
                .map(|d| format!("<p class=\"doc\">{}</p>", html_escape(d)))
                .unwrap_or_default();
            let method_rows: Vec<_> = methods
                .iter()
                .map(|(mn, msig)| {
                    format!(
                        r#"<tr><td><code>{}</code></td><td><code>{}</code></td></tr>"#,
                        html_escape(mn),
                        html_escape(msig)
                    )
                })
                .collect();
            let methods_table = if method_rows.is_empty() {
                String::new()
            } else {
                format!(
                    r#"<table class="fields">{}</table>"#,
                    method_rows.join("\n    ")
                )
            };
            format!(
                "<div class=\"item\" id=\"trait-{0}\">\n  <h3>trait <code class=\"sig\">{0}</code></h3>\n  {1}\n  {2}\n</div>",
                html_escape(name),
                doc_html,
                methods_table,
            )
        }
        DocItem::Const {
            name,
            is_pub,
            doc,
            ty,
        } => {
            let pub_badge = if *is_pub {
                r#"<span class="badge pub">pub</span> "#
            } else {
                ""
            };
            let doc_html = doc
                .as_deref()
                .map(|d| format!("<p class=\"doc\">{}</p>", html_escape(d)))
                .unwrap_or_default();
            let ty_str = ty
                .as_ref()
                .map(|t| format!(": {}", html_escape(t)))
                .unwrap_or_default();
            format!(
                "<div class=\"item\" id=\"const-{0}\">\n  <h3>{1}const <code class=\"sig\">{0}{2}</code></h3>\n  {3}\n</div>",
                html_escape(name),
                pub_badge,
                ty_str,
                doc_html,
            )
        }
        DocItem::TypeAlias {
            name,
            is_pub,
            doc,
            ty,
        } => {
            let pub_badge = if *is_pub {
                r#"<span class="badge pub">pub</span> "#
            } else {
                ""
            };
            let doc_html = doc
                .as_deref()
                .map(|d| format!("<p class=\"doc\">{}</p>", html_escape(d)))
                .unwrap_or_default();
            format!(
                "<div class=\"item\" id=\"type-{0}\">\n  <h3>{1}type <code class=\"sig\">{0} = {2}</code></h3>\n  {3}\n</div>",
                html_escape(name),
                pub_badge,
                html_escape(ty),
                doc_html,
            )
        }
        DocItem::Effect { name, doc, operations } => {
            let doc_html = doc
                .as_deref()
                .map(|d| format!("<p class=\"doc\">{}</p>", html_escape(d)))
                .unwrap_or_default();
            let op_rows: Vec<_> = operations
                .iter()
                .map(|(on, osig)| {
                    format!(
                        "<tr><td><code>{}</code></td><td><code>{}</code></td></tr>",
                        html_escape(on),
                        html_escape(osig)
                    )
                })
                .collect();
            let ops_table = if op_rows.is_empty() {
                String::new()
            } else {
                format!(
                    r#"<table class="fields">{}</table>"#,
                    op_rows.join("\n    ")
                )
            };
            format!(
                "<div class=\"item\" id=\"effect-{0}\">\n  <h3>effect <code class=\"sig\">{0}</code></h3>\n  {1}\n  {2}\n</div>",
                html_escape(name),
                doc_html,
                ops_table,
            )
        }
    }
}

fn render_toc(items: &[DocItem]) -> String {
    let mut entries = Vec::new();
    for item in items {
        match item {
            DocItem::Function { name, .. } => {
                let escaped = html_escape(name);
                entries.push(format!(
                    "<li><a href=\"#fn-{0}\">fn {0}</a></li>", escaped
                ));
            }
            DocItem::Struct { name, .. } => {
                let escaped = html_escape(name);
                entries.push(format!(
                    "<li><a href=\"#record-{0}\">record {0}</a></li>", escaped
                ));
            }
            DocItem::Enum { name, .. } => {
                let escaped = html_escape(name);
                entries.push(format!(
                    "<li><a href=\"#choice-{0}\">choice {0}</a></li>", escaped
                ));
            }
            DocItem::Trait { name, .. } => {
                let escaped = html_escape(name);
                entries.push(format!(
                    "<li><a href=\"#trait-{0}\">trait {0}</a></li>", escaped
                ));
            }
            DocItem::Const { name, .. } => {
                let escaped = html_escape(name);
                entries.push(format!(
                    "<li><a href=\"#const-{0}\">const {0}</a></li>", escaped
                ));
            }
            DocItem::TypeAlias { name, .. } => {
                let escaped = html_escape(name);
                entries.push(format!(
                    "<li><a href=\"#type-{0}\">type {0}</a></li>", escaped
                ));
            }
            DocItem::Effect { name, .. } => {
                let escaped = html_escape(name);
                entries.push(format!(
                    "<li><a href=\"#effect-{0}\">effect {0}</a></li>", escaped
                ));
            }
        }
    }
    format!("<ul>\n{}\n</ul>", entries.join("\n"))
}

/// Generate HTML documentation from an IRIS source string.
pub fn generate_docs(source: &str, filename: &str) -> Result<String, String> {
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| format!("lexer error: {}", e))?;
    let mut parser = Parser::new(&tokens);
    let module = parser
        .parse_module()
        .map_err(|e| format!("parse error: {}", e))?;

    let items = extract_items(&module);
    let toc = render_toc(&items);
    let item_html: Vec<_> = items.iter().map(render_item).collect();

    let title = html_escape(filename);

    Ok(format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — IRIS Documentation</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1a1a2e; background: #f8f9fa; display: flex; min-height: 100vh; }}
  nav {{ width: 260px; background: #16213e; color: #e0e0e0; padding: 24px 16px; position: fixed; top: 0; left: 0; bottom: 0; overflow-y: auto; }}
  nav h2 {{ font-size: 16px; color: #a8d8ea; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }}
  nav ul {{ list-style: none; }}
  nav li {{ margin: 4px 0; }}
  nav a {{ color: #c4c4c4; text-decoration: none; font-size: 14px; display: block; padding: 4px 8px; border-radius: 4px; }}
  nav a:hover {{ background: #1a1a40; color: #fff; }}
  main {{ margin-left: 260px; padding: 32px 48px; max-width: 900px; flex: 1; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; color: #16213e; }}
  .subtitle {{ color: #6c757d; margin-bottom: 32px; font-size: 14px; }}
  .item {{ margin-bottom: 36px; padding-bottom: 24px; border-bottom: 1px solid #dee2e6; }}
  .item:last-child {{ border-bottom: none; }}
  h3 {{ font-size: 18px; margin-bottom: 8px; }}
  .sig {{ background: #eef2ff; padding: 2px 8px; border-radius: 4px; font-size: 15px; }}
  .doc {{ color: #495057; margin: 8px 0 12px 0; line-height: 1.6; }}
  .badge {{ font-size: 11px; padding: 2px 6px; border-radius: 3px; font-weight: 600; vertical-align: middle; }}
  .badge.pub {{ background: #d4edda; color: #155724; }}
  table.fields {{ border-collapse: collapse; margin: 8px 0 0 0; font-size: 14px; }}
  table.fields td {{ padding: 6px 16px 6px 0; vertical-align: top; }}
  table.fields td:first-child {{ font-weight: 600; white-space: nowrap; }}
  code {{ font-family: "SF Mono", "Fira Code", "Cascadia Code", Consolas, monospace; font-size: 13px; }}
</style>
</head>
<body>
<nav>
  <h2>Contents</h2>
  {toc}
</nav>
<main>
  <h1>{title}</h1>
  <p class="subtitle">Auto-generated documentation</p>
  {items}
</main>
</body>
</html>"#,
        title = title,
        toc = toc,
        items = item_html.join("\n"),
    ))
}
