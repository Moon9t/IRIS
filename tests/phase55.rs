//! Phase 55: ForEach loops — `for x in list_expr { body }`
//!
//! Tests for: `for x in xs { ... }` desugared to while-loop over list.

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// Basic foreach: sum of elements
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_sum() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    var sum = 0
    for x in xs {
        sum = sum + x
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ---------------------------------------------------------------------------
// Foreach over empty list: body is never executed
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_empty() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    var count = 0
    for x in xs {
        count = count + 1
    }
    count
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "0");
}

// ---------------------------------------------------------------------------
// Foreach: count elements (length reconstruction)
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_count() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    push(xs, 40);
    var count = 0
    for x in xs {
        count = count + 1
    }
    count
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "4");
}

// ---------------------------------------------------------------------------
// Foreach: multiply all elements (product)
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_product() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 2);
    push(xs, 3);
    push(xs, 5);
    var prod = 1
    for x in xs {
        prod = prod * x
    }
    prod
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "30");
}

// ---------------------------------------------------------------------------
// Foreach: single element list
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_single() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 42);
    var total = 0
    for x in xs {
        total = total + x
    }
    total
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "42");
}

// ---------------------------------------------------------------------------
// Foreach: IR text contains foreach-related IR (list_len, list_get)
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_ir_text() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    var sum = 0
    for x in xs {
        sum = sum + x
    }
    sum
}
"#;
    let ir = compile(src, "test", EmitKind::Ir).unwrap();
    assert!(ir.contains("list_len"), "expected list_len in IR:\n{}", ir);
    assert!(ir.contains("list_get"), "expected list_get in IR:\n{}", ir);
}

// ---------------------------------------------------------------------------
// Foreach: nested foreach loops
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_nested() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    val ys = list()
    push(ys, 10);
    push(ys, 20);
    var total = 0
    for x in xs {
        for y in ys {
            total = total + x * y
        }
    }
    total
}
"#;
    // (1*10 + 1*20) + (2*10 + 2*20) = 30 + 60 = 90
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "90");
}

// ---------------------------------------------------------------------------
// Foreach: sum with multiplication
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_sum_mul() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 2);
    push(xs, 4);
    push(xs, 6);
    push(xs, 8);
    var sum = 0
    for x in xs {
        sum = sum + x
    }
    sum
}
"#;
    // 2 + 4 + 6 + 8 = 20
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "20");
}

// ---------------------------------------------------------------------------
// Foreach: LLVM IR contains list_len and list_get calls
// ---------------------------------------------------------------------------

#[test]
fn test_foreach_llvm() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 5);
    var s = 0
    for x in xs {
        s = s + x
    }
    s
}
"#;
    let ll = compile(src, "test", EmitKind::Llvm).unwrap();
    assert!(ll.contains("iris_list_len"), "expected iris_list_len in LLVM:\n{}", ll);
    assert!(ll.contains("iris_list_get"), "expected iris_list_get in LLVM:\n{}", ll);
}
