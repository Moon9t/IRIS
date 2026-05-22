//! Phase 96 integration tests: stdlib collections — set, queue, heap.

use iris::{compile_multi, EmitKind};

// ── 1. set_add deduplicates; set_len counts unique items ────────────────────
#[test]
fn test_set_add_dedup() {
    let src = r#"
bring std.set
def f() -> i64 {
    val s = set_new()
    val s2 = set_add(s, "a")
    val s3 = set_add(s2, "b")
    val s4 = set_add(s3, "a")
    set_len(s4)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 2. set_union merges two sets, no duplicates ──────────────────────────────
#[test]
fn test_set_union() {
    let src = r#"
bring std.set
def f() -> i64 {
    val s1 = set_new()
    val s1a = set_add(s1, "x")
    val s2 = set_new()
    val s2a = set_add(s2, "y")
    val s2b = set_add(s2a, "x")
    val s3 = set_union(s1a, s2b)
    set_len(s3)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 3. set_difference removes elements present in rhs ───────────────────────
#[test]
fn test_set_difference() {
    let src = r#"
bring std.set
def f() -> i64 {
    val s1 = set_new()
    val s1a = set_add(s1, "a")
    val s1b = set_add(s1a, "b")
    val s2 = set_new()
    val s2a = set_add(s2, "a")
    val diff = set_difference(s1b, s2a)
    set_len(diff)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 4. enqueue / dequeue preserves FIFO order ───────────────────────────────
#[test]
fn test_queue_fifo() {
    let src = r#"
bring std.queue
def f() -> i64 {
    val q = queue_new()
    val q2 = enqueue(q, 10)
    val q3 = enqueue(q2, 20)
    dequeue_val(q3)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 5. queue_is_empty after draining ────────────────────────────────────────
#[test]
fn test_queue_is_empty() {
    let src = r#"
bring std.queue
def bool_to_i64(b: bool) -> i64 { when b { true => 1, false => 0 } }
def f() -> i64 {
    val q = queue_new()
    val q2 = enqueue(q, 5)
    val q3 = dequeue_queue(q2)
    bool_to_i64(queue_is_empty(q3))
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "1");
}

// ── 6. heap_pop_val returns the minimum element ─────────────────────────────
#[test]
fn test_heap_min() {
    let src = r#"
bring std.heap
def f() -> i64 {
    val h = heap_new()
    val h2 = heap_push(h, 5)
    val h3 = heap_push(h2, 3)
    val h4 = heap_push(h3, 7)
    heap_pop_val(h4)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 7. heap pops two minimums in sorted order ───────────────────────────────
#[test]
fn test_heap_sorted_pops() {
    let src = r#"
bring std.heap
def f() -> i64 {
    val h0 = heap_new()
    val h1 = heap_push(h0, 5)
    val h2 = heap_push(h1, 1)
    val h3 = heap_push(h2, 3)
    val h4 = heap_push(h3, 2)
    val h5 = heap_push(h4, 4)
    val v1 = heap_pop_val(h5)
    val h6 = heap_pop_heap(h5)
    val v2 = heap_pop_val(h6)
    v1 + v2 * 10
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "21");
}

// ── 8. set_intersection gives common elements ───────────────────────────────
#[test]
fn test_set_intersection() {
    let src = r#"
bring std.set
def f() -> i64 {
    val s1 = set_new()
    val s1a = set_add(s1, "a")
    val s1b = set_add(s1a, "b")
    val s1c = set_add(s1b, "c")
    val s2 = set_new()
    val s2a = set_add(s2, "b")
    val s2b = set_add(s2a, "c")
    val s2c = set_add(s2b, "d")
    val inter = set_intersection(s1c, s2c)
    set_len(inter)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}
