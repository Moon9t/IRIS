//! Phase 67 integration tests: recursive ADT (self-referential choice types).

use iris::{compile, EmitKind};

// ---------------------------------------------------------------------------
// 1. Recursive Peano natural numbers: Zero | Succ(Nat)
// ---------------------------------------------------------------------------
#[test]
fn test_recursive_adt_peano_construct() {
    let src = r#"
choice Nat { Zero, Succ(Nat) }
def two(dummy: i64) -> Nat { Nat.Succ(Nat.Succ(Nat.Zero)) }
def nat_to_i64(n: Nat) -> i64 {
    when n {
        Nat.Zero => 0,
        Nat.Succ(m) => 1 + nat_to_i64(m),
    }
}
def f() -> i64 { nat_to_i64(two(0)) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ---------------------------------------------------------------------------
// 2. Addition using Peano naturals
// ---------------------------------------------------------------------------
#[test]
fn test_recursive_adt_peano_add() {
    let src = r#"
choice Nat { Zero, Succ(Nat) }
def nat_add(a: Nat, b: Nat) -> Nat {
    when a {
        Nat.Zero => b,
        Nat.Succ(pred) => Nat.Succ(nat_add(pred, b)),
    }
}
def nat_to_i64(n: Nat) -> i64 {
    when n {
        Nat.Zero => 0,
        Nat.Succ(m) => 1 + nat_to_i64(m),
    }
}
def from_i64(n: i64) -> Nat {
    when n {
        0 => Nat.Zero,
        _ => Nat.Succ(from_i64(n - 1)),
    }
}
def f() -> i64 { nat_to_i64(nat_add(from_i64(3), from_i64(4))) }
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "7");
}

// ---------------------------------------------------------------------------
// 3. Linked list: Nil | Cons(i64, List)
// ---------------------------------------------------------------------------
#[test]
fn test_recursive_adt_list_sum() {
    let src = r#"
choice List { Nil, Cons(i64, List) }
def sum(xs: List) -> i64 {
    when xs {
        List.Nil => 0,
        List.Cons(h, t) => h + sum(t),
    }
}
def f() -> i64 {
    val xs = List.Cons(1, List.Cons(2, List.Cons(3, List.Nil)))
    sum(xs)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ---------------------------------------------------------------------------
// 4. List length
// ---------------------------------------------------------------------------
#[test]
fn test_recursive_adt_list_len() {
    let src = r#"
choice List { Nil, Cons(i64, List) }
def length(xs: List) -> i64 {
    when xs {
        List.Nil => 0,
        List.Cons(h, t) => 1 + length(t),
    }
}
def f() -> i64 {
    val xs = List.Cons(10, List.Cons(20, List.Cons(30, List.Nil)))
    length(xs)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ---------------------------------------------------------------------------
// 5. Binary tree: Leaf | Node(Tree, i64, Tree)
// ---------------------------------------------------------------------------
#[test]
fn test_recursive_adt_tree_sum() {
    let src = r#"
choice Tree { Leaf, Node(Tree, i64, Tree) }
def tree_sum(t: Tree) -> i64 {
    when t {
        Tree.Leaf => 0,
        Tree.Node(l, v, r) => tree_sum(l) + v + tree_sum(r),
    }
}
def f() -> i64 {
    val t = Tree.Node(
        Tree.Node(Tree.Leaf, 1, Tree.Leaf),
        2,
        Tree.Node(Tree.Leaf, 3, Tree.Leaf)
    )
    tree_sum(t)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "6");
}

// ---------------------------------------------------------------------------
// 6. Maybe (option-like) recursive: Just | Nothing
// ---------------------------------------------------------------------------
#[test]
fn test_recursive_adt_maybe() {
    let src = r#"
choice Maybe { Nothing, Just(i64) }
def from_maybe(m: Maybe, default: i64) -> i64 {
    when m {
        Maybe.Nothing => default,
        Maybe.Just(v) => v,
    }
}
def f() -> i64 {
    from_maybe(Maybe.Just(42), 0) + from_maybe(Maybe.Nothing, 99)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "141");
}

// ---------------------------------------------------------------------------
// 7. List reverse using accumulator
// ---------------------------------------------------------------------------
#[test]
fn test_recursive_adt_list_reverse() {
    let src = r#"
choice List { Nil, Cons(i64, List) }
def rev_acc(xs: List, acc: List) -> List {
    when xs {
        List.Nil => acc,
        List.Cons(h, t) => rev_acc(t, List.Cons(h, acc)),
    }
}
def head(xs: List) -> i64 {
    when xs {
        List.Nil => -1,
        List.Cons(h, t) => h,
    }
}
def f() -> i64 {
    val xs = List.Cons(1, List.Cons(2, List.Cons(3, List.Nil)))
    val rev = rev_acc(xs, List.Nil)
    head(rev)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ---------------------------------------------------------------------------
// 8. Fibonacci using recursive pair structure
// ---------------------------------------------------------------------------
#[test]
fn test_recursive_adt_tree_depth() {
    let src = r#"
choice Tree { Leaf, Node(Tree, Tree) }
def depth(t: Tree) -> i64 {
    when t {
        Tree.Leaf => 0,
        Tree.Node(l, r) => {
            val dl = depth(l)
            val dr = depth(r)
            1 + when dl > dr { true => dl, false => dr }
        },
    }
}
def f() -> i64 {
    val t = Tree.Node(
        Tree.Node(Tree.Node(Tree.Leaf, Tree.Leaf), Tree.Leaf),
        Tree.Leaf
    )
    depth(t)
}
"#;
    let result = compile(src, "test", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}
