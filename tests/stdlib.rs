//! Comprehensive standard library integration tests.
//!
//! Each test brings one or more stdlib modules via `bring std.<name>`,
//! exercises the public API with concrete inputs, and verifies the returned
//! value through the interpreter (`EmitKind::Eval`).
//!
//! # Known name-collision limitations
//! IRIS builtins (`min(a,b)`, `max(a,b)`, `contains(s,sub)`) take priority
//! over user-defined functions of the same name when the lowerer resolves
//! calls.  Tests that would hit these conflicts use alternative names or
//! builtins directly.

use iris::{compile_multi, EmitKind};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compile + eval `src` (single "main" module) and return the trimmed output.
fn eval(src: &str) -> String {
    compile_multi(&[("main", src)], "main", EmitKind::Eval)
        .unwrap_or_else(|e| panic!("compile/eval failed:\n{}\nsrc:\n{}", e, src))
        .trim()
        .to_owned()
}

/// Like `eval` but does NOT trim — needed for padding/whitespace tests.
fn eval_raw(src: &str) -> String {
    compile_multi(&[("main", src)], "main", EmitKind::Eval)
        .unwrap_or_else(|e| panic!("compile/eval failed:\n{}\nsrc:\n{}", e, src))
}

// ---------------------------------------------------------------------------
// 1. math
// ---------------------------------------------------------------------------

#[test]
fn math_gcd() {
    assert_eq!(
        eval(
            r#"
        bring std.math
        def main() -> i64 { gcd(48, 18) }
    "#
        ),
        "6"
    );
}

#[test]
fn math_lcm() {
    assert_eq!(
        eval(
            r#"
        bring std.math
        def main() -> i64 { lcm(4, 6) }
    "#
        ),
        "12"
    );
}

#[test]
fn math_is_even_odd() {
    assert_eq!(
        eval(
            r#"
        bring std.math
        def main() -> i64 {
            val e = if is_even(10) { 1 } else { 0 }
            val o = if is_odd(7) { 1 } else { 0 }
            e + o
        }
    "#
        ),
        "2"
    );
}

#[test]
fn math_clamp() {
    assert_eq!(
        eval(
            r#"
        bring std.math
        def main() -> i64 {
            val lo  = clamp_i64(0 - 5, 0, 10)
            val hi  = clamp_i64(20, 0, 10)
            val mid = clamp_i64(5, 0, 10)
            lo + hi + mid
        }
    "#
        ),
        "15"
    );
}

#[test]
fn math_abs() {
    assert_eq!(
        eval(
            r#"
        bring std.math
        def main() -> i64 { abs_i64(0 - 42) }
    "#
        ),
        "42"
    );
}

#[test]
fn math_min_max() {
    assert_eq!(
        eval(
            r#"
        bring std.math
        def main() -> i64 { min_i64(3, 7) + max_i64(3, 7) }
    "#
        ),
        "10"
    );
}

// ---------------------------------------------------------------------------
// 2. iter
// ---------------------------------------------------------------------------

#[test]
fn iter_sum() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 {
            val xs: list<i64> = list()
            val _ = list_push(xs, 1)
            val _ = list_push(xs, 2)
            val _ = list_push(xs, 3)
            val _ = list_push(xs, 4)
            val _ = list_push(xs, 5)
            sum(xs)
        }
    "#
        ),
        "15"
    );
}

#[test]
fn iter_product() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 {
            val xs: list<i64> = list()
            val _ = list_push(xs, 1)
            val _ = list_push(xs, 2)
            val _ = list_push(xs, 3)
            val _ = list_push(xs, 4)
            product(xs)
        }
    "#
        ),
        "24"
    );
}

#[test]
fn iter_range() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 { sum(range(1, 6)) }
    "#
        ),
        "15"
    );
}

/// iter.min / iter.max conflict with the 2-arg builtin min/max.
/// Verify semantics by accessing first/last element of a range directly.
#[test]
fn iter_range_first_last() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 {
            val xs = range(3, 8)
            list_get(xs, 0) + list_get(xs, list_len(xs) - 1)
        }
    "#
        ),
        "10"
    ); // 3 + 7
}

#[test]
fn iter_take_drop() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 {
            val xs = range(0, 10)
            val t = take(xs, 3)
            val d = drop(xs, 7)
            sum(t) + sum(d)
        }
    "#
        ),
        "27"
    ); // take=[0,1,2]=3; drop=[7,8,9]=24
}

#[test]
fn iter_reverse() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 {
            val rev = reverse(range(1, 4))
            list_get(rev, 0)
        }
    "#
        ),
        "3"
    );
}

/// Use builtin list_contains to avoid name conflict with string `contains`.
#[test]
fn iter_list_contains() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 {
            val xs = range(0, 5)
            val has = if list_contains(xs, 3) { 1 } else { 0 }
            val not = if list_contains(xs, 9) { 0 } else { 1 }
            has + not
        }
    "#
        ),
        "2"
    );
}

#[test]
fn iter_count() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 { count(range(0, 5), 3) }
    "#
        ),
        "1"
    );
}

#[test]
fn iter_index_of() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        def main() -> i64 { index_of(range(10, 15), 12) }
    "#
        ),
        "2"
    );
}

// ---------------------------------------------------------------------------
// 3. string
// ---------------------------------------------------------------------------

#[test]
fn string_is_empty() {
    assert_eq!(
        eval(
            r#"
        bring std.string
        def main() -> i64 {
            val e  = if is_empty("") { 1 } else { 0 }
            val ne = if is_empty("hi") { 0 } else { 1 }
            e + ne
        }
    "#
        ),
        "2"
    );
}

#[test]
fn string_pad_left_length() {
    // Verify via len() — leading spaces would be stripped by trim().
    assert_eq!(
        eval(
            r#"
        bring std.string
        def main() -> i64 { len(pad_left("42", 5, "0")) }
    "#
        ),
        "5"
    );
}

#[test]
fn string_pad_left_value() {
    let out = eval_raw(
        r#"
        bring std.string
        def main() -> str { pad_left("42", 5, "0") }
    "#,
    );
    assert_eq!(out.trim(), "00042");
}

#[test]
fn string_pad_right() {
    assert_eq!(
        eval(
            r#"
        bring std.string
        def main() -> str { pad_right("hi", 5, "-") }
    "#
        ),
        "hi---"
    );
}

#[test]
fn string_str_join() {
    assert_eq!(
        eval(
            r#"
        bring std.string
        def main() -> str {
            val parts: list<str> = list()
            val _ = list_push(parts, "a")
            val _ = list_push(parts, "b")
            val _ = list_push(parts, "c")
            str_join(parts, "-")
        }
    "#
        ),
        "a-b-c"
    );
}

#[test]
fn string_repeat() {
    assert_eq!(
        eval(
            r#"
        bring std.string
        def main() -> str { str_repeat("ab", 3) }
    "#
        ),
        "ababab"
    );
}

// ---------------------------------------------------------------------------
// 4. fmt
// ---------------------------------------------------------------------------

#[test]
fn fmt_pad_int_length() {
    assert_eq!(
        eval(
            r#"
        bring std.fmt
        def main() -> i64 { len(pad_int(7, 4)) }
    "#
        ),
        "4"
    );
}

#[test]
fn fmt_zero_pad_int() {
    assert_eq!(
        eval(
            r#"
        bring std.fmt
        def main() -> str { zero_pad_int(42, 6) }
    "#
        ),
        "000042"
    );
}

#[test]
fn fmt_left_right_align_lengths() {
    assert_eq!(
        eval(
            r#"
        bring std.fmt
        def main() -> i64 {
            len(left_align("hi", 5)) + len(right_align("hi", 5))
        }
    "#
        ),
        "10"
    );
}

#[test]
fn fmt_sprintf_d() {
    assert_eq!(
        eval(
            r#"
        bring std.fmt
        def main() -> str {
            val args: list<str> = list()
            val _ = list_push(args, to_str(42))
            sprintf("val=%d", args)
        }
    "#
        ),
        "val=42"
    );
}

#[test]
fn fmt_sprintf_zero_pad() {
    assert_eq!(
        eval(
            r#"
        bring std.fmt
        def main() -> str {
            val args: list<str> = list()
            val _ = list_push(args, to_str(7))
            sprintf("%05d", args)
        }
    "#
        ),
        "00007"
    );
}

#[test]
fn fmt_sprintf_string_arg() {
    assert_eq!(
        eval(
            r#"
        bring std.fmt
        def main() -> str {
            val args: list<str> = list()
            val _ = list_push(args, "world")
            sprintf("hello %s!", args)
        }
    "#
        ),
        "hello world!"
    );
}

// ---------------------------------------------------------------------------
// 5. set
// ---------------------------------------------------------------------------

#[test]
fn set_add_contains_len() {
    assert_eq!(
        eval(
            r#"
        bring std.set
        def main() -> i64 {
            var s = set_new()
            s = set_add(s, "apple")
            s = set_add(s, "banana")
            s = set_add(s, "apple")
            set_len(s) + if set_contains(s, "banana") { 1 } else { 0 }
        }
    "#
        ),
        "3"
    );
}

#[test]
fn set_remove() {
    assert_eq!(
        eval(
            r#"
        bring std.set
        def main() -> i64 {
            var s = set_new()
            s = set_add(s, "x")
            s = set_add(s, "y")
            s = set_remove(s, "x")
            set_len(s)
        }
    "#
        ),
        "1"
    );
}

#[test]
fn set_union_intersection() {
    assert_eq!(
        eval(
            r#"
        bring std.set
        def main() -> i64 {
            var a = set_new()
            a = set_add(a, "a")
            a = set_add(a, "b")
            var b = set_new()
            b = set_add(b, "b")
            b = set_add(b, "c")
            set_len(set_union(a, b)) + set_len(set_intersection(a, b))
        }
    "#
        ),
        "4"
    );
}

#[test]
fn set_difference() {
    assert_eq!(
        eval(
            r#"
        bring std.set
        def main() -> i64 {
            var a = set_new()
            a = set_add(a, "x")
            a = set_add(a, "y")
            a = set_add(a, "z")
            var b = set_new()
            b = set_add(b, "y")
            set_len(set_difference(a, b))
        }
    "#
        ),
        "2"
    );
}

// ---------------------------------------------------------------------------
// 6. queue
// ---------------------------------------------------------------------------

#[test]
fn queue_enqueue_dequeue() {
    assert_eq!(
        eval(
            r#"
        bring std.queue
        def main() -> i64 {
            var q = queue_new()
            q = enqueue(q, 10)
            q = enqueue(q, 20)
            q = enqueue(q, 30)
            dequeue_val(q) + queue_len(q)
        }
    "#
        ),
        "13"
    );
}

#[test]
fn queue_peek_is_empty() {
    assert_eq!(
        eval(
            r#"
        bring std.queue
        def main() -> i64 {
            var q = queue_new()
            val e1 = if queue_is_empty(q) { 1 } else { 0 }
            q = enqueue(q, 99)
            val pk = queue_peek(q)
            val e2 = if queue_is_empty(q) { 1 } else { 0 }
            e1 + pk + e2
        }
    "#
        ),
        "100"
    );
}

// ---------------------------------------------------------------------------
// 7. heap (min-heap)
// ---------------------------------------------------------------------------

#[test]
fn heap_push_peek_len() {
    assert_eq!(
        eval(
            r#"
        bring std.heap
        def main() -> i64 {
            var h = heap_new()
            h = heap_push(h, 5)
            h = heap_push(h, 1)
            h = heap_push(h, 3)
            heap_peek(h) + heap_len(h)
        }
    "#
        ),
        "4"
    );
}

#[test]
fn heap_sorted_pops() {
    assert_eq!(
        eval(
            r#"
        bring std.heap
        def main() -> i64 {
            var h = heap_new()
            h = heap_push(h, 8)
            h = heap_push(h, 2)
            h = heap_push(h, 5)
            val a = heap_pop_val(h)
            h = heap_pop_heap(h)
            val b = heap_pop_val(h)
            a + b
        }
    "#
        ),
        "7"
    );
}

// ---------------------------------------------------------------------------
// 8. time
// ---------------------------------------------------------------------------

#[test]
fn time_format_ms() {
    assert_eq!(
        eval(
            r#"
        bring std.time
        def main() -> str { format_duration(450) }
    "#
        ),
        "450ms"
    );
}

#[test]
fn time_format_seconds() {
    let out = eval(
        r#"
        bring std.time
        def main() -> str { format_duration(2500) }
    "#,
    );
    assert!(out.contains('s'), "should contain 's': {}", out);
}

#[test]
fn time_now_ms_positive() {
    assert_eq!(
        eval(
            r#"
        bring std.time
        def main() -> i64 { if now_ms() > 0 { 1 } else { 0 } }
    "#
        ),
        "1"
    );
}

#[test]
fn time_elapsed_non_negative() {
    assert_eq!(
        eval(
            r#"
        bring std.time
        def main() -> i64 {
            val start = now_ms()
            if elapsed_ms(start) >= 0 { 1 } else { 0 }
        }
    "#
        ),
        "1"
    );
}

// ---------------------------------------------------------------------------
// 9. testing helpers
// ---------------------------------------------------------------------------

#[test]
fn testing_assert_eq_pass() {
    assert_eq!(
        eval(
            r#"
        bring std.testing
        def main() -> bool { assert_eq(6, 6, "ok") }
    "#
        ),
        "true"
    );
}

#[test]
fn testing_assert_eq_fail() {
    let out = eval(
        r#"
        bring std.testing
        def main() -> bool { assert_eq(5, 6, "fail") }
    "#,
    );
    assert!(out.contains("false"));
    assert!(out.contains("FAIL"));
}

#[test]
fn testing_assert_str_eq() {
    assert_eq!(
        eval(
            r#"
        bring std.testing
        def main() -> bool { assert_str_eq("hello", "hello", "match") }
    "#
        ),
        "true"
    );
}

#[test]
fn testing_assert_true_false() {
    assert_eq!(
        eval(
            r#"
        bring std.testing
        def main() -> bool {
            val t = assert_true(1 < 2, "1 < 2")
            val f = assert_false(2 < 1, "2 not < 1")
            if t { if f { true } else { false } } else { false }
        }
    "#
        ),
        "true"
    );
}

#[test]
fn testing_assert_list_eq() {
    assert_eq!(
        eval(
            r#"
        bring std.testing
        def main() -> bool {
            val a: list<i64> = list()
            val _ = list_push(a, 1)
            val _ = list_push(a, 2)
            val _ = list_push(a, 3)
            val b: list<i64> = list()
            val _ = list_push(b, 1)
            val _ = list_push(b, 2)
            val _ = list_push(b, 3)
            assert_list_eq(a, b, "equal")
        }
    "#
        ),
        "true"
    );
}

// ---------------------------------------------------------------------------
// 10. ml
// ---------------------------------------------------------------------------

#[test]
fn ml_sigmoid_bounds() {
    assert_eq!(
        eval(
            r#"
        bring std.ml
        def main() -> i64 {
            val z  = sigmoid(0.0)
            val hi = sigmoid(100.0)
            val lo = sigmoid(0.0 - 100.0)
            val ok_z  = if z  > 0.4 { if z  < 0.6  { 1 } else { 0 } } else { 0 }
            val ok_hi = if hi > 0.99 { 1 } else { 0 }
            val ok_lo = if lo < 0.01 { 1 } else { 0 }
            ok_z + ok_hi + ok_lo
        }
    "#
        ),
        "3"
    );
}

#[test]
fn ml_sigmoid_deriv_from_output() {
    let out = eval(
        r#"
        bring std.ml
        def main() -> f64 {
            sigmoid_deriv_from_output(0.5)
        }
    "#,
    );
    let v: f64 = out.parse().expect("f64");
    assert!(
        (v - 0.25).abs() < 1e-9,
        "sigmoid_deriv_from_output = {} expected 0.25",
        v
    );
}

#[test]
fn ml_dot_product() {
    let out = eval(
        r#"
        bring std.ml
        def main() -> f64 {
            val a: list<f64> = list()
            val _ = list_push(a, 1.0)
            val _ = list_push(a, 2.0)
            val _ = list_push(a, 3.0)
            val b: list<f64> = list()
            val _ = list_push(b, 4.0)
            val _ = list_push(b, 5.0)
            val _ = list_push(b, 6.0)
            dot(a, b)
        }
    "#,
    );
    let v: f64 = out.parse().expect("f64");
    assert!((v - 32.0).abs() < 1e-9, "dot = {} expected 32", v);
}

#[test]
fn ml_vec_mean() {
    assert_eq!(
        eval(
            r#"
        bring std.ml
        def main() -> i64 {
            val xs: list<f64> = list()
            val _ = list_push(xs, 2.0)
            val _ = list_push(xs, 4.0)
            val _ = list_push(xs, 6.0)
            val m = vec_mean(xs)
            if m > 3.9 { if m < 4.1 { 1 } else { 0 } } else { 0 }
        }
    "#
        ),
        "1"
    );
}

#[test]
fn ml_standardize_zero_mean() {
    assert_eq!(
        eval(
            r#"
        bring std.ml
        def main() -> i64 {
            val xs: list<f64> = list()
            val _ = list_push(xs, 1.0)
            val _ = list_push(xs, 2.0)
            val _ = list_push(xs, 3.0)
            val m = vec_mean(standardize(xs))
            val tol = 0.01
            val neg_tol = 0.0 - tol
            if m > neg_tol { if m < tol { 1 } else { 0 } } else { 0 }
        }
    "#
        ),
        "1"
    );
}

// ---------------------------------------------------------------------------
// 11. nn — neural network smoke tests
// ---------------------------------------------------------------------------

#[test]
fn nn_mlp_weight_count() {
    assert_eq!(
        eval(
            r#"
        bring std.nn
        def main() -> i64 {
            val sizes: list<i64> = list()
            val _ = list_push(sizes, 2)
            val _ = list_push(sizes, 4)
            val _ = list_push(sizes, 1)
            val net = mlp_create(sizes)
            val nw = list_len(net.0)
            val nb = list_len(net.1)
            if nw == 12 { if nb == 5 { 1 } else { 0 } } else { 0 }
        }
    "#
        ),
        "1"
    );
}

#[test]
fn nn_mlp_forward_range() {
    assert_eq!(
        eval(
            r#"
        bring std.nn
        def main() -> i64 {
            val sizes: list<i64> = list()
            val _ = list_push(sizes, 2)
            val _ = list_push(sizes, 3)
            val _ = list_push(sizes, 1)
            val net = mlp_create(sizes)
            val input: list<f64> = list()
            val _ = list_push(input, 0.5)
            val _ = list_push(input, 0.5)
            val acts = mlp_forward(net.0, net.1, net.2, input, 0)
            val y = list_get(list_get(acts, list_len(acts) - 1), 0)
            if y > 0.0 { if y < 1.0 { 1 } else { 0 } } else { 0 }
        }
    "#
        ),
        "1"
    );
}

#[test]
fn nn_mlp_backward_single_neuron_gradient() {
    let out = eval(
        r#"
        bring std.nn
        def main() -> f64 {
            val weights: list<f64> = list()
            val _ = list_push(weights, 0.0)
            val biases: list<f64> = list()
            val _ = list_push(biases, 0.0)
            val shapes: list<i64> = list()
            val _ = list_push(shapes, encode_shape(1, 1))
            val x: list<f64> = list()
            val _ = list_push(x, 1.0)
            val y: list<f64> = list()
            val _ = list_push(y, 1.0)
            val acts = mlp_forward(weights, biases, shapes, x, 0)
            val grads = mlp_backward(weights, shapes, acts, y, 0)
            list_get(grads.0, 0)
        }
    "#,
    );
    let v: f64 = out.parse().expect("f64");
    assert!(
        (v + 0.125).abs() < 1e-9,
        "weight gradient = {} expected -0.125",
        v
    );
}

#[test]
fn nn_mlp_train_step_adam_reduces_loss() {
    assert_eq!(
        eval(
            r#"
        bring std.nn
        def main() -> i64 {
            val weights: list<f64> = list()
            val _ = list_push(weights, 0.0)
            val biases: list<f64> = list()
            val _ = list_push(biases, 0.0)
            val shapes: list<i64> = list()
            val _ = list_push(shapes, encode_shape(1, 1))
            val x: list<f64> = list()
            val _ = list_push(x, 1.0)
            val y: list<f64> = list()
            val _ = list_push(y, 1.0)

            val x_flat: list<f64> = list()
            val _ = list_push(x_flat, 1.0)
            val y_flat: list<f64> = list()
            val _ = list_push(y_flat, 1.0)

            val before = mlp_eval_loss(weights, biases, shapes, x_flat, y_flat, 1, 1, 1, 0)
            val state = mlp_adam_state(weights, biases)
            val updated = mlp_train_step_adam(weights, biases, shapes, x, y, state.0, state.1, state.2, state.3, 0.1, 0.9, 0.999, 0.00000001, 1, 0)
            val after = mlp_eval_loss(updated.0, updated.1, shapes, x_flat, y_flat, 1, 1, 1, 0)
            if after < before { 1 } else { 0 }
        }
    "#
        ),
        "1"
    );
}

#[test]
fn reverse_mode_square_plus_linear_grad() {
    let out = eval(
        r#"
        def main() -> f64 {
            val x = tape(3.0)
            val y = x * x + 2.0 * x
            val _ = backward(y)
            grad(x)
        }
    "#,
    );
    let v: f64 = out.parse().expect("f64");
    assert!((v - 8.0).abs() < 1e-9, "grad(x) = {} expected 8.0", v);
}

#[test]
fn reverse_mode_leaf_grad() {
    let out = eval(
        r#"
        def main() -> f64 {
            val x = tape(3.0)
            val _ = backward(x)
            grad(x)
        }
    "#,
    );
    let v: f64 = out.parse().expect("f64");
    assert!((v - 1.0).abs() < 1e-9, "grad(x) = {} expected 1.0", v);
}

#[test]
fn reverse_mode_exp_grad() {
    let out = eval(
        r#"
        def main() -> f64 {
            val x = tape(0.0)
            val y = exp(x)
            val _ = backward(y)
            grad(x)
        }
    "#,
    );
    let v: f64 = out.parse().expect("f64");
    assert!((v - 1.0).abs() < 1e-9, "grad(x) = {} expected 1.0", v);
}

#[test]
fn tensor_matmul2_values() {
    assert_eq!(
        eval(
            r#"
        bring std.tensorx
        def main() -> i64 {
            val a_shape: list<i64> = list()
            val _ = list_push(a_shape, 2)
            val _ = list_push(a_shape, 2)
            val a_data: list<f64> = list()
            val _ = list_push(a_data, 1.0)
            val _ = list_push(a_data, 2.0)
            val _ = list_push(a_data, 3.0)
            val _ = list_push(a_data, 4.0)

            val b_shape: list<i64> = list()
            val _ = list_push(b_shape, 2)
            val _ = list_push(b_shape, 2)
            val b_data: list<f64> = list()
            val _ = list_push(b_data, 5.0)
            val _ = list_push(b_data, 6.0)
            val _ = list_push(b_data, 7.0)
            val _ = list_push(b_data, 8.0)

            val out = tensor_matmul2(tensor_from_data(a_data, a_shape), tensor_from_data(b_data, b_shape), 2, 2, 2)
            val data = tensor_data(out)
            if list_len(data) == 4
                && list_get(data, 0) == 19.0
                && list_get(data, 1) == 22.0
                && list_get(data, 2) == 43.0
                && list_get(data, 3) == 50.0 {
                1
            } else {
                0
            }
        }
    "#
        ),
        "1"
    );
}

#[test]
fn tensor_batch_matmul_values() {
    assert_eq!(
        eval(
            r#"
        bring std.tensorx
        def main() -> i64 {
            val a_shape: list<i64> = list()
            val _ = list_push(a_shape, 2)
            val _ = list_push(a_shape, 1)
            val _ = list_push(a_shape, 2)
            val a_data: list<f64> = list()
            val _ = list_push(a_data, 1.0)
            val _ = list_push(a_data, 2.0)
            val _ = list_push(a_data, 3.0)
            val _ = list_push(a_data, 4.0)

            val b_shape: list<i64> = list()
            val _ = list_push(b_shape, 2)
            val _ = list_push(b_shape, 2)
            val _ = list_push(b_shape, 1)
            val b_data: list<f64> = list()
            val _ = list_push(b_data, 5.0)
            val _ = list_push(b_data, 6.0)
            val _ = list_push(b_data, 7.0)
            val _ = list_push(b_data, 8.0)

            val out = tensor_batch_matmul(tensor_from_data(a_data, a_shape), tensor_from_data(b_data, b_shape), 2, 1, 2, 1)
            val data = tensor_data(out)
            if list_len(data) == 2
                && list_get(data, 0) == 17.0
                && list_get(data, 1) == 53.0 {
                1
            } else {
                0
            }
        }
    "#
        ),
        "1"
    );
}

// ---------------------------------------------------------------------------
// 12. Cross-module: iter + testing
// ---------------------------------------------------------------------------

#[test]
fn cross_iter_testing() {
    assert_eq!(
        eval(
            r#"
        bring std.iter
        bring std.testing
        def main() -> bool {
            val xs = range(1, 6)
            val expected: list<i64> = list()
            val _ = list_push(expected, 1)
            val _ = list_push(expected, 2)
            val _ = list_push(expected, 3)
            val _ = list_push(expected, 4)
            val _ = list_push(expected, 5)
            assert_list_eq(xs, expected, "range(1,6)")
        }
    "#
        ),
        "true"
    );
}

// ---------------------------------------------------------------------------
// 13. path
// ---------------------------------------------------------------------------

#[test]
fn path_basename() {
    assert_eq!(
        eval(
            r#"
        bring std.path
        def main() -> str { basename("/home/user/file.iris") }
    "#
        ),
        "file.iris"
    );
}

#[test]
fn path_dirname() {
    assert_eq!(
        eval(
            r#"
        bring std.path
        def main() -> str { dirname("/home/user/file.iris") }
    "#
        ),
        "/home/user"
    );
}

#[test]
fn path_extension() {
    assert_eq!(
        eval(
            r#"
        bring std.path
        def main() -> str { extension("notes.txt") }
    "#
        ),
        "txt"
    );
}

#[test]
fn path_join_path() {
    let out = eval(
        r#"
        bring std.path
        def main() -> str { join_path("/usr", "local") }
    "#,
    );
    assert!(out.contains("local"), "join_path: {}", out);
}

// ---------------------------------------------------------------------------
// 14. log
// ---------------------------------------------------------------------------

#[test]
fn log_info_returns_true() {
    let out = eval(
        r#"
        bring std.log
        def main() -> bool { info("hello log") }
    "#,
    );
    assert!(out.contains("true"));
    assert!(out.contains("hello log"));
}

#[test]
fn log_at_level_filter() {
    assert_eq!(
        eval(
            r#"
        bring std.log
        def main() -> bool { log_at(0, 2, "debug suppressed") }
    "#
        ),
        "false"
    );
}

// ---------------------------------------------------------------------------
// 15. deque
// ---------------------------------------------------------------------------

#[test]
fn deque_push_peek() {
    assert_eq!(
        eval(
            r#"
        bring std.deque
        def main() -> i64 {
            val dq  = deque_create()
            val dq2 = push_back(dq, 10)
            val dq3 = push_back(dq2, 20)
            val dq4 = push_front(dq3, 5)
            peek_front(dq4) + peek_back(dq4)
        }
    "#
        ),
        "25"
    );
}

// ---------------------------------------------------------------------------
// 16. bitset
// ---------------------------------------------------------------------------

#[test]
fn bitset_set_get() {
    assert_eq!(
        eval(
            r#"
        bring std.bitset
        def main() -> i64 {
            val bs  = bitset_new(64)
            val bs2 = bitset_set(bs, 3)
            val bs3 = bitset_set(bs2, 7)
            val has3 = if bitset_get(bs3, 3) { 1 } else { 0 }
            val has7 = if bitset_get(bs3, 7) { 1 } else { 0 }
            val no5  = if bitset_get(bs3, 5) { 0 } else { 1 }
            has3 + has7 + no5
        }
    "#
        ),
        "3"
    );
}

// ---------------------------------------------------------------------------
// 17. ais & rl (Autonomous Intelligent Systems & Reinforcement Learning)
// ---------------------------------------------------------------------------

#[test]
fn ais_decision_strategies() {
    assert_eq!(
        eval(
            r#"
        bring std.ais
        def main() -> i64 {
            val logits : list<f64> = list()
            val _ = list_push(logits, 1.0)
            val _ = list_push(logits, 5.0)
            val _ = list_push(logits, 2.0)
            val best = argmax(logits)
            
            // epsilon_greedy with 0.0 epsilon should be greedy (argmax)
            val eg = epsilon_greedy(logits, 0.0)
            
            // softmax sample should return valid index
            val sm = softmax_sample(logits)
            val ok_sm = if sm >= 0 { if sm < 3 { 1 } else { 0 } } else { 0 }

            best + eg + ok_sm
        }
    "#
        ),
        "3"
    );
}

#[test]
fn ais_reward_processing() {
    assert_eq!(
        eval(
            r#"
        bring std.ais
        def main() -> i64 {
            val rewards : list<f64> = list()
            val _ = list_push(rewards, 0.0)
            val _ = list_push(rewards, 0.0)
            val _ = list_push(rewards, 10.0)
            val returns = discount_rewards(rewards, 0.9)
            val v0 = list_get(returns, 0)
            val v2 = list_get(returns, 2)
            
            // v0 = 10 * 0.9 * 0.9 = 8.1
            // v2 = 10.0
            val ok = if v0 > 8.09 { if v0 < 8.11 { if v2 == 10.0 { 1 } else { 0 } } else { 0 } } else { 0 }
            ok
        }
    "#
        ),
        "1"
    );
}

#[test]
fn rl_q_learning_and_sarsa() {
    assert_eq!(
        eval(
            r#"
        bring std.rl
        def main() -> i64 {
            val q = q_table_new(3, 2)
            val _ = q_set(q, 0, 1, 2, 5.0)
            val v0 = q_get(q, 0, 1, 2)
            
            // Q-learning update: state=0, action=0, reward=1.0, next_state=0
            // Q(0,0) starts at 0.0. Q(0,1) is 5.0.
            // td_target = 1.0 + 0.9 * 5.0 = 5.5
            // new_value = 0.0 + 0.1 * (5.5 - 0.0) = 0.55
            val updated = q_learning_update(q, 0, 0, 1.0, 0, 2, 0.1, 0.9)
            
            // Best action at state 0 should be 1 (val 5.0)
            val best = q_best_action(q, 0, 2)

            val ok = if v0 == 5.0 { if updated > 0.54 { if updated < 0.56 { if best == 1 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
            ok
        }
    "#
        ),
        "1"
    );
}

#[test]
fn rl_replay_buffer() {
    assert_eq!(
        eval(
            r#"
        bring std.rl
        def main() -> i64 {
            var buf = replay_buffer_new(2)
            val s1 : list<f64> = list()
            val _ = list_push(s1, 1.0)
            val s2 : list<f64> = list()
            val _ = list_push(s2, 2.0)
            val e1 = Experience { state: s1, action: 1, reward: 0.5, next_state: s2, done: false }
            
            buf = replay_buffer_push(buf, e1)
            val sz1 = buf.size
            buf = replay_buffer_push(buf, e1)
            val sz2 = buf.size
            
            val idxs = replay_buffer_sample_indices(buf, 5)
            val ok_sz = if sz1 == 1 { if sz2 == 2 { 1 } else { 0 } } else { 0 }
            val ok_idxs = if list_len(idxs) == 5 { 1 } else { 0 }
            
            ok_sz + ok_idxs
        }
    "#
        ),
        "2"
    );
}

#[test]
fn ais_model_persistence() {
    assert_eq!(
        eval(
            r#"
        bring std.ais
        def main() -> i64 {
            val path = "temp_model_test_weights.txt"
            val weights : list<f64> = list()
            val _ = list_push(weights, 1.5)
            val _ = list_push(weights, 0.0 - 2.5)
            val _ = list_push(weights, 3.14)
            
            val ok_save = model_save(weights, path)
            val loaded = model_load(path)
            
            val len_ok = if list_len(loaded) == 3 { 1 } else { 0 }
            val val_ok = if list_get(loaded, 0) == 1.5 { if list_get(loaded, 2) == 3.14 { 1 } else { 0 } } else { 0 }
            
            // Clean up temp file
            val _ = remove_file(path)
            
            if ok_save { len_ok + val_ok } else { 0 }
        }
    "#
        ),
        "2"
    );
}
