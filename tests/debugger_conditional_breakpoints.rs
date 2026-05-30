//! Integration tests for Debugger Conditional Breakpoints (v0.6.1).

use iris::debugger::{BreakpointInfo, DebugSession};

#[test]
fn test_debugger_conditional_breakpoints() {
    let src = r#"
def main() -> i64 {
    var i = 0;
    var sum = 0;
    while i < 10 {
        sum = sum + i;
        i = i + 1;
    }
    sum
}
"#;
    let mut session = DebugSession::new();
    session.set_source(src);

    // Set a conditional breakpoint on line 6 ("sum = sum + i;") when i == 5
    let info = BreakpointInfo {
        condition: Some("i == 5".to_owned()),
        hit_condition: None,
        log_message: None,
        hit_count: 0,
    };
    session.set_breakpoint(6, Some(info));
    session.start().expect("failed to start debugger");

    let frame = session.continue_to_breakpoint();
    assert!(frame.is_some(), "expected conditional breakpoint to be hit");

    let frame_ref = frame.unwrap();
    assert_eq!(frame_ref.line, 6);

    // Verify that the variable i in the snapshot has value "5"
    let i_var = frame_ref.variables.iter().find(|(name, _)| name == "i");
    assert!(i_var.is_some(), "variable 'i' should be in scope");
    assert_eq!(
        i_var.unwrap().1,
        "5",
        "conditional breakpoint hit on wrong iteration"
    );
}
