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

        let mut send = |body: serde_json::Value| -> std::io::Result<()> {
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
                        "supportsStepInTargetsRequest": false,
                        "supportsSetVariable": false,
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
                let source_path = arguments["program"].as_str().unwrap_or("");
                if !source_path.is_empty() {
                    if let Ok(src) = std::fs::read_to_string(source_path) {
                        session.set_source(&src);
                    }
                }
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command, "body": {}
                }))?;
                seq += 1;
            }
            "setBreakpoints" => {
                session = DebugSession::new(); // reset between launch/setBreakpoints
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
                                }
                            }))?;
                            seq += 1;
                        } else {
                            // No breakpoints hit — program finished.
                            send(serde_json::json!({
                                "seq": seq, "type": "event", "event": "terminated"
                            }))?;
                            seq += 1;
                        }
                    }
                    Err(e) => {
                        send(serde_json::json!({
                            "seq": seq, "type": "event", "event": "output",
                            "body": { "category": "stderr", "output": format!("error: {}\n", e) }
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
                        "body": { "reason": "breakpoint", "threadId": 1, "line": frame.line }
                    }))?;
                    seq += 1;
                } else {
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
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "stopped",
                        "body": { "reason": "step", "threadId": 1, "line": line }
                    }))?;
                    seq += 1;
                } else {
                    send(serde_json::json!({
                        "seq": seq, "type": "event", "event": "terminated"
                    }))?;
                    seq += 1;
                }
            }
            "stackTrace" => {
                let frames: Vec<serde_json::Value> = session.current_frame().into_iter()
                    .map(|f| serde_json::json!({
                        "id": 0,
                        "name": f.func_name,
                        "line": f.line,
                        "column": f.column,
                        "source": {}
                    }))
                    .collect();
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "stackFrames": frames, "totalFrames": frames.len() }
                }))?;
                seq += 1;
            }
            "scopes" => {
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "scopes": [{ "name": "Locals", "variablesReference": 1, "expensive": false }] }
                }))?;
                seq += 1;
            }
            "variables" => {
                let vars: Vec<serde_json::Value> = session.current_frame()
                    .map(|f| f.variables.iter().map(|(name, val)| serde_json::json!({
                        "name": name,
                        "value": val,
                        "variablesReference": 0,
                    })).collect())
                    .unwrap_or_default();
                send(serde_json::json!({
                    "seq": seq, "type": "response", "request_seq": request_seq,
                    "success": true, "command": command,
                    "body": { "variables": vars }
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
