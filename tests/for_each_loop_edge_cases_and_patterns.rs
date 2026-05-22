//! Phase 119 integration tests: for-each loop edge cases and patterns.

use iris::{compile, EmitKind};

// ── 1. Foreach with string list ─────────────────────────────────────────────
#[test]
fn test_foreach_string_list() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, "hello");
    push(xs, "world");
    push(xs, "foo");
    var total_len = 0
    for s in xs {
        total_len = total_len + len(s)
    }
    total_len
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 5 + 5 + 3 = 13
    assert_eq!(result.trim(), "13");
}

// ── 2. Foreach building new list ────────────────────────────────────────────
#[test]
fn test_foreach_build_list() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    val ys = list()
    for x in xs {
        push(ys, x * 10);
    }
    list_get(ys, 0) + list_get(ys, 1) + list_get(ys, 2)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ── 3. Foreach with conditional accumulation ────────────────────────────────
#[test]
fn test_foreach_conditional() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    push(xs, 4);
    push(xs, 5);
    push(xs, 6);
    var even_sum = 0
    for x in xs {
        even_sum = even_sum + x * (1 - x % 2)
    }
    even_sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 2 + 4 + 6 = 12
    assert_eq!(result.trim(), "12");
}

// ── 4. Foreach counting matches ─────────────────────────────────────────────
#[test]
fn test_foreach_count_matches() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    push(xs, 40);
    push(xs, 50);
    var count = 0
    for x in xs {
        count = if x > 25 { count + 1 } else { count }
    }
    count
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 5. Foreach with max finding ─────────────────────────────────────────────
#[test]
fn test_foreach_find_max() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 3);
    push(xs, 7);
    push(xs, 2);
    push(xs, 9);
    push(xs, 1);
    var mx = list_get(xs, 0)
    for x in xs {
        mx = if x > mx { x } else { mx }
    }
    mx
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "9");
}

// ── 6. Foreach with min finding ─────────────────────────────────────────────
#[test]
fn test_foreach_find_min() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 3);
    push(xs, 7);
    push(xs, 2);
    push(xs, 9);
    push(xs, 1);
    var mn = list_get(xs, 0)
    for x in xs {
        mn = if x < mn { x } else { mn }
    }
    mn
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 7. Nested foreach ───────────────────────────────────────────────────────
#[test]
fn test_nested_foreach() {
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
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // (1*10 + 1*20) + (2*10 + 2*20) = 30 + 60 = 90
    assert_eq!(result.trim(), "90");
}

// ── 8. Foreach with running average ─────────────────────────────────────────
#[test]
fn test_foreach_sum_and_count() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 10);
    push(xs, 20);
    push(xs, 30);
    push(xs, 40);
    var sum = 0
    var count = 0
    for x in xs {
        sum = sum + x
        count = count + 1
    }
    sum / count
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "25");
}
