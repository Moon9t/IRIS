use iris::{compile, EmitKind};

fn eval_i64(src: &str) -> i64 {
    let out = compile(src, "test", EmitKind::Eval).expect("should eval");
    out.trim().parse().expect("expected i64 output")
}

fn assert_eval_i64(src: &str, expected: i64) {
    let value = eval_i64(src);
    assert_eq!(value, expected, "source:\n{}", src);
}

// Arithmetic

#[test]
fn test_add_1_2() {
    assert_eval_i64(r#"def f() -> i64 { 1 + 2 }"#, 3);
}

#[test]
fn test_add_5_7() {
    assert_eval_i64(r#"def f() -> i64 { 5 + 7 }"#, 12);
}

#[test]
fn test_add_chain() {
    assert_eval_i64(r#"def f() -> i64 { 1 + 2 + 3 + 4 }"#, 10);
}

#[test]
fn test_sub_9_4() {
    assert_eval_i64(r#"def f() -> i64 { 9 - 4 }"#, 5);
}

#[test]
fn test_sub_3_8() {
    assert_eval_i64(r#"def f() -> i64 { 3 - 8 }"#, -5);
}

#[test]
fn test_sub_chain() {
    assert_eval_i64(r#"def f() -> i64 { 10 - 3 - 2 }"#, 5);
}

#[test]
fn test_mul_3_4() {
    assert_eval_i64(r#"def f() -> i64 { 3 * 4 }"#, 12);
}

#[test]
fn test_mul_7_0() {
    assert_eval_i64(r#"def f() -> i64 { 7 * 0 }"#, 0);
}

#[test]
fn test_mul_chain() {
    assert_eval_i64(r#"def f() -> i64 { 2 * 3 * 4 }"#, 24);
}

#[test]
fn test_div_9_3() {
    assert_eval_i64(r#"def f() -> i64 { 9 / 3 }"#, 3);
}

#[test]
fn test_div_20_4() {
    assert_eval_i64(r#"def f() -> i64 { 20 / 4 }"#, 5);
}

#[test]
fn test_div_floor() {
    assert_eval_i64(r#"def f() -> i64 { 7 / 2 }"#, 3);
}

#[test]
fn test_mod_10_3() {
    assert_eval_i64(r#"def f() -> i64 { 10 % 3 }"#, 1);
}

#[test]
fn test_mod_14_5() {
    assert_eval_i64(r#"def f() -> i64 { 14 % 5 }"#, 4);
}

#[test]
fn test_precedence_1() {
    assert_eval_i64(r#"def f() -> i64 { 1 + 2 * 3 }"#, 7);
}

#[test]
fn test_precedence_2() {
    assert_eval_i64(r#"def f() -> i64 { (1 + 2) * 3 }"#, 9);
}

#[test]
fn test_negation_1() {
    assert_eval_i64(r#"def f() -> i64 { 0 - 5 }"#, -5);
}

#[test]
fn test_negation_2() {
    assert_eval_i64(r#"def f() -> i64 { -7 }"#, -7);
}

#[test]
fn test_mix_ops_1() {
    assert_eval_i64(r#"def f() -> i64 { 5 * 2 + 3 }"#, 13);
}

#[test]
fn test_mix_ops_2() {
    assert_eval_i64(r#"def f() -> i64 { 5 + 2 * 3 - 1 }"#, 10);
}

// Comparisons and booleans

#[test]
fn test_cmp_lt_true() {
    assert_eval_i64(r#"def f() -> i64 { if 1 < 2 { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_cmp_lt_false() {
    assert_eval_i64(r#"def f() -> i64 { if 2 < 1 { 1 } else { 0 } }"#, 0);
}

#[test]
fn test_cmp_le_true() {
    assert_eval_i64(r#"def f() -> i64 { if 2 <= 2 { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_cmp_le_false() {
    assert_eval_i64(r#"def f() -> i64 { if 3 <= 2 { 1 } else { 0 } }"#, 0);
}

#[test]
fn test_cmp_gt_true() {
    assert_eval_i64(r#"def f() -> i64 { if 3 > 2 { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_cmp_gt_false() {
    assert_eval_i64(r#"def f() -> i64 { if 2 > 3 { 1 } else { 0 } }"#, 0);
}

#[test]
fn test_cmp_ge_true() {
    assert_eval_i64(r#"def f() -> i64 { if 3 >= 3 { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_cmp_ge_false() {
    assert_eval_i64(r#"def f() -> i64 { if 2 >= 3 { 1 } else { 0 } }"#, 0);
}

#[test]
fn test_cmp_eq_true() {
    assert_eval_i64(r#"def f() -> i64 { if 4 == 4 { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_cmp_eq_false() {
    assert_eval_i64(r#"def f() -> i64 { if 4 == 5 { 1 } else { 0 } }"#, 0);
}

#[test]
fn test_cmp_ne_true() {
    assert_eval_i64(r#"def f() -> i64 { if 4 != 5 { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_cmp_ne_false() {
    assert_eval_i64(r#"def f() -> i64 { if 4 != 4 { 1 } else { 0 } }"#, 0);
}

#[test]
fn test_bool_and_true() {
    assert_eval_i64(r#"def f() -> i64 { if true && true { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_bool_and_false() {
    assert_eval_i64(r#"def f() -> i64 { if true && false { 1 } else { 0 } }"#, 0);
}

#[test]
fn test_bool_or_true() {
    assert_eval_i64(r#"def f() -> i64 { if false || true { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_bool_or_false() {
    assert_eval_i64(
        r#"def f() -> i64 { if false || false { 1 } else { 0 } }"#,
        0,
    );
}

#[test]
fn test_bool_not_true() {
    assert_eval_i64(r#"def f() -> i64 { if !false { 1 } else { 0 } }"#, 1);
}

#[test]
fn test_bool_not_false() {
    assert_eval_i64(r#"def f() -> i64 { if !true { 1 } else { 0 } }"#, 0);
}

#[test]
fn test_bool_combined_1() {
    assert_eval_i64(
        r#"def f() -> i64 { if (1 < 2) && (2 < 3) { 1 } else { 0 } }"#,
        1,
    );
}

#[test]
fn test_bool_combined_2() {
    assert_eval_i64(
        r#"def f() -> i64 { if (1 < 2) && (3 < 2) { 1 } else { 0 } }"#,
        0,
    );
}

// Control flow

#[test]
fn test_if_else_value() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val x = if 1 < 2 { 3 } else { 4 }
    x
}
"#,
        3,
    );
}

#[test]
fn test_if_else_nested() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val x = if 1 > 2 { 0 } else { if 2 > 1 { 5 } else { 6 } }
    x
}
"#,
        5,
    );
}

#[test]
fn test_while_sum() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var i = 0
    var sum = 0
    while i < 5 {
        sum = sum + i
        i = i + 1
    }
    sum
}
"#,
        10,
    );
}

#[test]
fn test_while_zero() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var i = 0
    while false {
        i = i + 1
    }
    i
}
"#,
        0,
    );
}

#[test]
fn test_while_break() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var i = 0
    while i < 10 {
        if i == 3 { break }
        i = i + 1
    }
    i
}
"#,
        3,
    );
}

#[test]
fn test_while_continue() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var i = 0
    var sum = 0
    while i < 5 {
        i = i + 1
        if i == 3 { continue }
        sum = sum + i
    }
    sum
}
"#,
        12,
    );
}

#[test]
fn test_for_sum() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var sum = 0
    for i in 0..5 {
        sum = sum + i
    }
    sum
}
"#,
        10,
    );
}

#[test]
fn test_for_zero() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var sum = 1
    for i in 0..0 {
        sum = 0
    }
    sum
}
"#,
        1,
    );
}

#[test]
fn test_loop_break() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var x = 0
    loop {
        x = 7
        break
    }
    x
}
"#,
        7,
    );
}

#[test]
fn test_return_early() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    if true { return 9 }
    1
}
"#,
        9,
    );
}

#[test]
fn test_loop_nested() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var i = 0
    while i < 2 {
        var j = 0
        while j < 2 {
            j = j + 1
        }
        i = i + 1
    }
    i
}
"#,
        2,
    );
}

#[test]
fn test_if_in_loop() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var i = 0
    var sum = 0
    while i < 5 {
        if i % 2 == 0 { sum = sum + i }
        i = i + 1
    }
    sum
}
"#,
        6,
    );
}

#[test]
fn test_for_with_if() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var sum = 0
    for i in 0..5 {
        if i < 3 { sum = sum + 1 }
    }
    sum
}
"#,
        3,
    );
}

#[test]
fn test_if_else_branch() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val x = if 1 > 2 { 1 } else { 2 }
    x
}
"#,
        2,
    );
}

#[test]
fn test_loop_increment() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var x = 0
    loop {
        x = x + 1
        if x == 3 { break }
    }
    x
}
"#,
        3,
    );
}

// Data structures

#[test]
fn test_array_index() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val arr = [1, 2, 3]
    arr[1]
}
"#,
        2,
    );
}

#[test]
fn test_array_store() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    var arr = [1, 2, 3]
    arr[1] = 9
    arr[1]
}
"#,
        9,
    );
}

#[test]
fn test_array_sum_for() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val arr = [1, 2, 3]
    var sum = 0
    for i in 0..3 {
        sum = sum + arr[i]
    }
    sum
}
"#,
        6,
    );
}

#[test]
fn test_array_sum_while() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val arr = [1, 2, 3]
    var i = 0
    var sum = 0
    while i < 3 {
        sum = sum + arr[i]
        i = i + 1
    }
    sum
}
"#,
        6,
    );
}

#[test]
fn test_tuple_index() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val t = (1, 2, 3)
    t.2
}
"#,
        3,
    );
}

#[test]
fn test_tuple_destructure() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val (a, b) = (3, 4)
    a + b
}
"#,
        7,
    );
}

#[test]
fn test_tuple_nested() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val t = ((1, 2), 3)
    val a = t.0
    a.1
}
"#,
        2,
    );
}

#[test]
fn test_record_field() {
    assert_eval_i64(
        r#"
record Point { x: i64, y: i64 }
def f() -> i64 {
    val p = Point { x: 2, y: 3 }
    p.y
}
"#,
        3,
    );
}

#[test]
fn test_record_return() {
    assert_eval_i64(
        r#"
record Point { x: i64, y: i64 }
def make(x: i64) -> Point { Point { x: x, y: x + 1 } }
def f() -> i64 {
    val p = make(4)
    p.y
}
"#,
        5,
    );
}

#[test]
fn test_choice_when() {
    assert_eval_i64(
        r#"
choice Color { Red, Green, Blue }
def f() -> i64 {
    val c = Color.Green
    when c {
        Color.Red => 1,
        Color.Green => 2,
        Color.Blue => 3,
    }
}
"#,
        2,
    );
}

#[test]
fn test_choice_with_data() {
    assert_eval_i64(
        r#"
choice Shape { Circle(i64), Square(i64) }
def f() -> i64 {
    val s = Shape.Circle(3)
    when s {
        Shape.Circle(r) => r,
        Shape.Square(x) => x,
    }
}
"#,
        3,
    );
}

#[test]
fn test_option_unwrap() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    unwrap(some(9))
}
"#,
        9,
    );
}

#[test]
fn test_option_when() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val o = some(4)
    when o {
        some(x) => x,
        none => 0,
    }
}
"#,
        4,
    );
}

#[test]
fn test_result_when() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val r = ok(7)
    when r {
        ok(x) => x,
        err(e) => 0,
    }
}
"#,
        7,
    );
}

#[test]
fn test_result_unwrap() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    unwrap(ok(11))
}
"#,
        11,
    );
}

// Functions, generics, traits, closures, strings

#[test]
fn test_function_call() {
    assert_eval_i64(
        r#"
def add(a: i64, b: i64) -> i64 { a + b }
def f() -> i64 { add(2, 3) }
"#,
        5,
    );
}

#[test]
fn test_default_param() {
    assert_eval_i64(
        r#"
def inc(x: i64 = 1) -> i64 { x + 1 }
def f() -> i64 { inc() }
"#,
        2,
    );
}

#[test]
fn test_generic_identity() {
    assert_eval_i64(
        r#"
def id[T](x: T) -> T { x }
def f() -> i64 { id(3) }
"#,
        3,
    );
}

#[test]
fn test_generic_first() {
    assert_eval_i64(
        r#"
def first[T, U](a: T, b: U) -> T { a }
def f() -> i64 { first(9, "x") }
"#,
        9,
    );
}

#[test]
fn test_trait_double() {
    assert_eval_i64(
        r#"
trait Double { def double(x: i64) -> i64 }
impl Double for i64 { def double(x: i64) -> i64 { x * 2 } }
def f() -> i64 { double(21) }
"#,
        42,
    );
}

#[test]
fn test_closure_basic() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val add = |x: i64| x + 1
    add(4)
}
"#,
        5,
    );
}

#[test]
fn test_closure_capture() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val n = 5
    val add = |x: i64| x + n
    add(7)
}
"#,
        12,
    );
}

#[test]
fn test_closure_two_params() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val mul = |a: i64, b: i64| a * b
    mul(3, 4)
}
"#,
        12,
    );
}

#[test]
fn test_string_len() {
    assert_eval_i64(
        r#"
def f() -> i64 { len("hello") }
"#,
        5,
    );
}

#[test]
fn test_string_concat_len() {
    assert_eval_i64(
        r#"
def f() -> i64 {
    val s = concat("a", "bc")
    len(s)
}
"#,
        3,
    );
}
