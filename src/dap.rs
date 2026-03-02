//! Debug Adapter Protocol (DAP) server for IRIS.
//!
//! [`run_dap_server`] implements the DAP JSON-RPC protocol over stdin/stdout,
//! allowing any DAP-compatible editor (VSCode, Neovim with nvim-dap, etc.) to
//! debug IRIS programs interactively.
//!
//! The server uses [`crate::debugger::DebugSession`] for trace-based debugging
//! with source-level breakpoints.

use std::io::{Read, Write};

use crate::debugger::DebugSession;

/// Runs the DAP server, reading JSON-RPC messages from stdin and writing
/// responses/events to stdout. Blocks until the client sends `disconnect`.
///
/// Use `iris dap` to start this server; configure your editor to use
/// `iris dap` as the debug adapter command with adapter type `"iris"`.
pub fn run_dap_server() -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut session = DebugSession::new();
    let mut seq = 1i64;
    let mut source_path = String::new();
    let mut source_content = String::new();

    loop {
        // Read Content-Length header.
        let mut content_length: usize = 0;
        loop {
            let mut byte = [0u8];
            let mut chars = String::new();
            loop {
                stdin.lock().read_exact(&mut byte)?;
                if byte[0] == b'\r' { continue; }
                if byte[0] == b'\n' { break; }
                chars.push(byte[0] as char);
            }
            if chars.is_empty() { break; }
            if chars.to_lowercase().starts_with("content-length:") {
                let val = chars["content-length:".len()..].trim();
                content_length = val.parse().unwrap_or(0);
            }
        }
        if content_length == 0 { continue; }

        let mut body = vec![0u8; content_length];
        stdin.lock().read_exact(&mut body)?;
        let body_str = String::from_utf8_lossy(&body);

        let msg: serde_json::Value = match serde_json::from_str(&body_str) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let request_seq = msg["seq"].as_i64().unwrap_or(0);
        let command = msg["command"].as_str().unwrap_or("");
        let arguments = msg.get("arguments").cloned().unwrap_or(serde_json::Value::Null);

        let send = |body: serde_json::Value| -> std::io::Result<()> {
            let text = serde_json::to_string(&body).unwrap_or_default();
            write!(stdout.lock(), "Content-Length: {}\r\n\r\n{}", text.len(), text)?;
            stdout.lock().flush()
        };

        match command {
            "initialize" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": {
                        "supportsConfigurationDoneRequest": true,
                        "supportsEvaluateForHovers": true,
                        "supportsStepBack": true,
                        "supportsExceptionInfoRequest": true,
                        "supportsSetVariable": false,
                        "supportsValueFormattingOptions": false,
                        "supportsTerminateRequest": true,
                        "supportsRestartRequest": false,
                        "supportsCompletionsRequest": false,
                        "supportsModulesRequest": false,
                        "supportsLoadedSourcesRequest": true,
                        "supportsLogPoints": false,
                        "supportsBreakpointLocationsRequest": false,
                        "supportsStepInTargetsRequest": false,
                    }
                }))?;
                seq += 1;
                // Send initialized event.
                send(serde_json::json!({
                    "seq": seq, "type": "event", "event": "initialized"
                }))?;
                seq += 1;
            }
            "launch" => {
                source_path = arguments["program"].as_str().unwrap_or("").to_owned();
                if !source_path.is_empty() {
                    if let Ok(src) = std::fs::read_to_string(&source_path) {
                        source_content = src.clone();
                        session.set_source(&src);
                    }
                }
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command, "body": {}
                }))?;
                seq += 1;
                // Send output event for program start
                send(serde_json::json!({
                    "seq": seq, "type": "event", "event": "output",
                    "body": {
                        "category": "console",
                        "output": format!("Debugging: {}\n", source_path)
                    }
                }))?;
                seq += 1;
            }
            "setBreakpoints" => {
                // Reset the session but keep source.
                let old_source = source_content.clone();
                session = DebugSession::new();
                if !old_source.is_empty() {
                    session.set_source(&old_source);
                }
                let bps = arguments["breakpoints"].as_array()
                    .cloned()
                    .unwrap_or_default();
                let mut verified = Vec::new();
                for bp in &bps {
                    let line = bp["line"].as_u64().unwrap_or(0) as u32;
                    session.set_breakpoint(line);
                    verified.push(serde_json::json!({ "verified": true, "line": line }));
                }
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "breakpoints": verified }
                }))?;
                seq += 1;
            }
            "setExceptionBreakpoints" => {
                // Accept but don't do anything special (all panics are caught by trace)
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": {}
                }))?;
                seq += 1;
            }
            "configurationDone" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command, "body": {}
                }))?;
                seq += 1;
                // Start execution.
                match session.start() {
                    Ok(()) => {
                        // Send stopped event at first breakpoint.
                        if let Some(frame) = session.continue_to_breakpoint() {
                            send(serde_json::json!({
                                "seq": seq, "type": "event", "event": "stopped",
                                "body": {
                                    "reason": "breakpoint",
                                    "threadId": 1,
                                    "allThreadsStopped": true,
                                    "line": frame.line,
                                    "description": format!("Breakpoint hit at line {}", frame.line),
                                    "text": format!("Paused in {}", frame.func_name),
                                }
                            }))?;
                            seq += 1;
                        } else {
                            // No breakpoints hit — program finished.
                            send(serde_json::json!({
                                "seq": seq, "type": "event", "event": "output",
                                "body": {
                                    "category": "console",
                                    "output": "Program completed without hitting any breakpoints.\n"
                                }
                            }))?;
                            seq += 1;
                            send(serde_json::json!({
                                "seq": seq, "type": "event", "event": "terminated"
                            }))?;
                            seq += 1;
                        }
                    }
                    Err(e) => {
                        // Use the rich error renderer for debug output
                        let error_msg = if !source_content.is_empty() {
                            crate::diagnostics::render_error(&source_content, &e)
                        } else {
                            format!("error: {}\n", e)
                        };
                        send(serde_json::json!({
                            "seq": seq, "type": "event", "event": "output",
                            "body": { "category": "stderr", "output": error_msg }
                        }))?;
                        seq += 1;
                        send(serde_json::json!({
                            "seq": seq, "type": "event", "event": "terminated"
                        }))?;
                        seq += 1;
                    }
                }
            }
            "continue" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command, "body": { "allThreadsContinued": true }
                }))?;
                seq += 1;
                if let Some(frame) = session.continue_to_breakpoint() {
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "stopped",
                        "body": {
                            "reason": "breakpoint",
                            "threadId": 1,
                            "line": frame.line,
                            "description": format!("Breakpoint hit at line {}", frame.line),
                        }
                    }))?;
                    seq += 1;
                } else {
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "output",
                        "body": { "category": "console", "output": "Program finished.\n" }
                    }))?;
                    seq += 1;
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "terminated"
                    }))?;
                    seq += 1;
                }
            }
            "next" | "stepIn" | "stepOut" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command, "body": {}
                }))?;
                seq += 1;
                if session.step() {
                    let line = session.current_frame().map(|f| f.line).unwrap_or(0);
                    let func = session.current_frame().map(|f| f.func_name.clone()).unwrap_or_default();
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "stopped",
                        "body": {
                            "reason": "step",
                            "threadId": 1,
                            "line": line,
                            "description": format!("Stepped to line {} in {}", line, func),
                        }
                    }))?;
                    seq += 1;
                } else {
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "output",
                        "body": { "category": "console", "output": "Program finished.\n" }
                    }))?;
                    seq += 1;
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "terminated"
                    }))?;
                    seq += 1;
                }
            }
            "stepBack" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command, "body": {}
                }))?;
                seq += 1;
                if session.step_back() {
                    let line = session.current_frame().map(|f| f.line).unwrap_or(0);
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "stopped",
                        "body": {
                            "reason": "step",
                            "threadId": 1,
                            "line": line,
                            "description": "Stepped back",
                        }
                    }))?;
                    seq += 1;
                } else {
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "stopped",
                        "body": { "reason": "step", "threadId": 1, "description": "Already at program start" }
                    }))?;
                    seq += 1;
                }
            }
            "stackTrace" => {
                let frames: Vec<serde_json::Value> = session.all_visible_frames().into_iter()
                    .enumerate()
                    .map(|(idx, f)| {
                        let mut frame = serde_json::json!({
                            "id": idx,
                            "name": f.func_name,
                            "line": f.line,
                            "column": f.column,
                        });
                        if !source_path.is_empty() {
                            frame["source"] = serde_json::json!({
                                "name": std::path::Path::new(&source_path)
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| source_path.clone()),
                                "path": &source_path,
                            });
                        }
                        frame
                    })
                    .collect();
                let total = frames.len();
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "stackFrames": frames, "totalFrames": total }
                }))?;
                seq += 1;
            }
            "scopes" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "scopes": [
                        { "name": "Locals", "variablesReference": 1, "expensive": false },
                    ] }
                }))?;
                seq += 1;
            }
            "variables" => {
                let vars: Vec<serde_json::Value> = session.current_frame()
                    .map(|f| f.variables.iter().map(|(name, val)| {
                        // Try to determine the type from the value string
                        let ty = if val.parse::<i64>().is_ok() { "i64" }
                            else if val.parse::<f64>().is_ok() { "f64" }
                            else if val == "true" || val == "false" { "bool" }
                            else { "str" };
                        serde_json::json!({
                            "name": name,
                            "value": val,
                            "type": ty,
                            "variablesReference": 0,
                        })
                    }).collect())
                    .unwrap_or_default();
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "variables": vars }
                }))?;
                seq += 1;
            }
            "threads" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "threads": [{ "id": 1, "name": "main" }] }
                }))?;
                seq += 1;
            }
            "evaluate" => {
                let expr = arguments["expression"].as_str().unwrap_or("").to_owned();
                let context = arguments["context"].as_str().unwrap_or("repl");
                // Build an eval source from the current debug frame variables + expression.
                let ctx_vars: Vec<(String, String)> = session.current_frame()
                    .map(|f| f.variables.clone())
                    .unwrap_or_default();
                let result = evaluate_in_context(&ctx_vars, &expr);

                // For hover context, return a cleaner result
                let display_result = if context == "hover" && result.starts_with("(cannot evaluate") {
                    // Don't show error on hover
                    String::new()
                } else {
                    result
                };

                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": !display_result.is_empty(),
                    "command": command,
                    "body": { "result": display_result, "variablesReference": 0 }
                }))?;
                seq += 1;
            }
            "loadedSources" => {
                let mut sources = Vec::new();
                if !source_path.is_empty() {
                    sources.push(serde_json::json!({
                        "name": std::path::Path::new(&source_path)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| source_path.clone()),
                        "path": &source_path,
                    }));
                }
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "sources": sources }
                }))?;
                seq += 1;
            }
            "exceptionInfo" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": {
                        "exceptionId": "panic",
                        "description": "IRIS runtime panic",
                        "breakMode": "always"
                    }
                }))?;
                seq += 1;
            }
            "disconnect" | "terminate" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command, "body": {}
                }))?;
                break;
            }
            _ => {
                // Unknown command — respond with null body to avoid client stalling.
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command, "body": {}
                }))?;
                seq += 1;
            }
        }
    }
    Ok(())
}

/// Evaluates an expression in the context of the current debug frame's variables.
/// Constructs a synthetic IRIS source with the variable bindings, then compiles and runs it.
fn evaluate_in_context(vars: &[(String, String)], expr: &str) -> String {
    if expr.trim().is_empty() {
        return String::new();
    }

    // Build val bindings from variable snapshot.
    let mut bindings = String::new();
    for (name, val_str) in vars {
        // Skip names that aren't valid identifiers.
        if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
            // Heuristic: if the value looks like a number, use it directly; else wrap as str.
            let v = if val_str.parse::<i64>().is_ok() {
                val_str.clone()
            } else if val_str.parse::<f64>().is_ok() {
                val_str.clone()
            } else if val_str == "true" || val_str == "false" {
                val_str.clone()
            } else {
                format!("\"{}\"", val_str.replace('"', "\\\""))
            };
            bindings.push_str(&format!("    val {} = {};\n", name, v));
        }
    }

    // Try each return type.
    for ret_ty in &["i64", "f64", "bool", "str"] {
        let src = format!(
            "def __dbg_eval__() -> {} {{\n{}\n    {}\n}}\n",
            ret_ty, bindings, expr
        );
        if let Ok(result) = crate::compile(&src, "__dbg__", crate::EmitKind::Eval) {
            return result.trim_end_matches('\n').to_owned();
        }
    }

    format!("(cannot evaluate: {})", expr)
}
