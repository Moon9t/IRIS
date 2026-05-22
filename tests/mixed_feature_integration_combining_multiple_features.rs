//! Phase 120 integration tests: mixed feature integration — combining multiple features.

use iris::{compile, EmitKind};

// ── 1. Record + closure + map ───────────────────────────────────────────────
#[test]
fn test_struct_closure_combo() {
    let src = r#"
record Pair { a: i64, b: i64 }
def f() -> i64 {
    val p = Pair { a: 3, b: 7 }
    val sum_fields = |x: i64| x + p.a + p.b
    sum_fields(0)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "10");
}

// ── 2. List + map + for-each ────────────────────────────────────────────────
#[test]
fn test_list_map_foreach() {
    let src = r#"
def f() -> i64 {
    val m = map()
    val keys = list()
    push(keys, "a");
    push(keys, "b");
    push(keys, "c");
    map_set(m, "a", 1);
    map_set(m, "b", 2);
    map_set(m, "c", 3);
    var total = 0
    for k in keys {
        val opt = map_get(m, k)
        total = if is_some(opt) { total + unwrap(opt) } else { total }
    }
    total
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ── 3. Recursion + generic ──────────────────────────────────────────────────
#[test]
fn test_recursion_generic() {
    let src = r#"
def max_val[T](a: T, b: T) -> T {
    if a > b { a } else { b }
}
def fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
def f() -> i64 {
    max_val(fib(6), fib(7))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // fib(6)=8, fib(7)=13 => max = 13
    assert_eq!(result.trim(), "13");
}

// ── 4. F-string + record ────────────────────────────────────────────────────
#[test]
fn test_fstring_struct() {
    let src = r#"
record Person { name: str, age: i64 }
def f() -> str {
    val p = Person { name: "Alice", age: 30 }
    val n = p.name
    f"Name: {n}"
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "Name: Alice");
}

// ── 5. Closure + for-each + list ────────────────────────────────────────────
#[test]
fn test_closure_foreach_list() {
    let src = r#"
def f() -> i64 {
    val xs = list()
    push(xs, 1);
    push(xs, 2);
    push(xs, 3);
    val multiplier = 10
    val ys = xs.map(|x: i64| x * multiplier)
    var sum = 0
    for y in ys {
        sum = sum + y
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ── 6. Choice + option combination ──────────────────────────────────────────
#[test]
fn test_choice_option() {
    let src = r#"
choice Action { Go(i64), Stop }
def process(a: Action) -> i64 {
    when a {
        Action.Go(speed) => speed,
        Action.Stop => 0,
    }
}
def f() -> i64 {
    process(Action.Go(60))
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "60");
}

// ── 7. Method + condition + loop ────────────────────────────────────────────
#[test]
fn test_method_condition_loop() {
    let src = r#"
record Counter { count: i64 }
impl Counter {
    def get(self: Counter) -> i64 { self.count }
}
def f() -> i64 {
    var sum = 0
    for i in 1..5 {
        val c = Counter { count: i * 10 }
        sum = if c.get() > 20 { sum + c.get() } else { sum }
    }
    sum
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // i=3 => 30, i=4 => 40; sum = 70
    assert_eq!(result.trim(), "70");
}

// ── 8. String operations + list ─────────────────────────────────────────────
#[test]
fn test_string_list_combo() {
    let src = r#"
def f() -> i64 {
    val words = list()
    push(words, "hello");
    push(words, "world");
    push(words, "iris");
    var total_len = 0
    for w in words {
        val upper = to_upper(w)
        total_len = total_len + len(upper)
    }
    total_len
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    // 5 + 5 + 4 = 14
    assert_eq!(result.trim(), "14");
}
