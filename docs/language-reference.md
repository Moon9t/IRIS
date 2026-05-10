# IRIS Language Reference

IRIS is a statically-typed, expression-oriented DSL designed for ML pipelines, data processing, and systems programming. It compiles to native code via LLVM IR.

---

## Table of Contents

1. [Basics](#basics)
2. [Types](#types)
3. [Bindings](#bindings)
4. [Functions](#functions)
5. [Control Flow](#control-flow)
6. [Collections](#collections)
7. [Strings](#strings)
8. [Records and Enums](#records-and-enums)
9. [Options and Results](#options-and-results)
10. [Closures](#closures)
11. [Traits and Impls](#traits-and-impls)
12. [Concurrency](#concurrency)
13. [Modules (bring)](#modules-bring)
14. [Builtins](#builtins)

---

## Basics

IRIS source files use the `.iris` extension. The entry point is a zero-argument function named `main` that returns `i64`:

```iris
def main() -> i64 {
    print("Hello, world!");
    0
}
```

Functions return the value of their last expression — no explicit `return` required (though `return expr` is supported for early exit).

Statements are separated by semicolons `;` when they appear inside blocks and are not the final expression. The tail expression of a block is its return value.

---

## Types

### Scalar types

| Type   | Description                        |
|--------|------------------------------------|
| `i64`  | 64-bit signed integer (default)    |
| `i32`  | 32-bit signed integer              |
| `f64`  | 64-bit float (double)              |
| `f32`  | 32-bit float                       |
| `bool` | Boolean (`true` / `false`)         |
| `str`  | UTF-8 string (heap-allocated)      |

### Compound types

| Type              | Description                              |
|-------------------|------------------------------------------|
| `list<T>`         | Growable array                           |
| `map<str, T>`     | Hash map with string keys                |
| `option<T>`       | Nullable wrapper (`some(v)` / `none`)    |
| `result<T, E>`    | Success/error wrapper (`ok(v)` / `err(e)`) |
| `(T1, T2, ...)`   | Tuple                                    |
| `[T; N]`          | Fixed-size array                         |
| `tensor<f32, [M, N]>` | Multi-dimensional tensor            |

### Type aliases

```iris
type Matrix = list<list<f64>>
```

---

## Bindings

```iris
val x = 42          // inferred immutable binding
var y = 0           // inferred mutable binding; can be reassigned
val z: f64 = 3.14   // with explicit type annotation
```

In tooling and docs, unannotated `val` and `var` declarations are both treated
as inferred bindings. Use an explicit annotation when the type should be part of
the source contract.

Reassigning a `var`:
```iris
var counter = 0
counter = counter + 1
```

---

## Functions

```iris
pub def add(a: i64, b: i64) -> i64 {
    a + b
}
```

- `pub` makes a function visible when the file is brought into another file.
- Parameters have the form `name: type`.
- Return type follows `->`.
- The last expression is the return value; `return expr` exits early.

### Async functions

```iris
async def fetch(url: str) -> str {
    await http_get(url)
}
```

---

## Control Flow

### if / else

```iris
val label = if score > 0.5 { "positive" } else { "negative" }
```

`if` is an expression — both branches must have the same type.

### while

```iris
var i = 0
while i < 10 {
    print(i);
    i = i + 1
}
```

### for (range loop)

```iris
for i in 0..10 {
    print(i);
}
```

### loop / break / continue

```iris
loop {
    val x = read_i64();
    if x == 0 { break }
}
```

### when (pattern matching)

Used with `choice` enums:

```iris
choice Direction { North, South, East, West }

val msg = when dir {
    Direction.North => "going north",
    Direction.South => "going south",
    _               => "other direction"
}
```

Also works with `option<T>`:
```iris
val result = when my_option {
    some(v) => v * 2,
    none    => 0
}
```

And `result<T, E>`:
```iris
val value = when my_result {
    ok(v)  => v,
    err(e) => { print(e); 0 }
}
```

### return (early exit)

```iris
def find_first(xs: list<i64>, target: i64) -> i64 {
    var i = 0
    while i < list_len(xs) {
        if list_get(xs, i) == target { return i }
        i = i + 1
    }
    -1
}
```

---

## Collections

### Lists

```iris
val nums = list()           // empty list<i64>
push(nums, 10);
push(nums, 20);
val n  = list_len(nums)     // 2
val v  = list_get(nums, 0)  // 10
list_set(nums, 0, 99);
val top = list_pop(nums)    // 99
```

### Maps

```iris
val m = map()
map_set(m, "key", 42);
val v = map_get(m, "key")       // some(42)
val found = map_contains(m, "key")  // true
map_remove(m, "key");
val n = map_len(m)
```

### Arrays (fixed-size)

```iris
val arr: [i64; 5] = [1, 2, 3, 4, 5]
val x = arr[2]       // load
arr[2] = 99          // store
```

### Tuples

```iris
val pair = (10, "hello")
val first  = pair.0     // 10
val second = pair.1     // "hello"
```

---

## Strings

```iris
val s = "hello"
val n = len(s)                      // 5
val t = concat(s, " world")         // "hello world"
val up = to_upper(s)               // "HELLO"
val low = to_lower(s)              // "hello"
val tr = trim("  hi  ")            // "hi"
val rep = repeat("ab", 3)          // "ababab"
val b = contains(s, "ell")         // true
val b2 = starts_with(s, "he")      // true
val b3 = ends_with(s, "lo")        // true
val num = to_str(42)               // "42"

// Slicing
val sub = slice(s, 1, 3)           // "el"

// Splitting / Joining
val parts = split("a,b,c", ",")    // list["a","b","c"]
val joined = join(parts, "-")      // "a-b-c"

// Finding (returns option<i64>)
val pos = find(s, "ll")            // some(2)
```

### String formatting

```iris
val msg = format("x={}, y={}", x, y)
```

---

## Records and Enums

### Records (structs)

```iris
record Point { x: f64, y: f64 }

def make_point(x: f64, y: f64) -> Point {
    Point { x: x, y: y }
}

val p = make_point(1.0, 2.0)
val dist = sqrt(p.x * p.x + p.y * p.y)
```

### Enums (choice)

```iris
choice Color { Red, Green, Blue }

val c = Color.Green

val name = when c {
    Color.Red   => "red",
    Color.Green => "green",
    Color.Blue  => "blue"
}
```

---

## Options and Results

### option<T>

```iris
val a: option<i64> = some(42)
val b: option<i64> = none

val x = is_some(a)     // true
val v = unwrap(a)      // 42 (panics if none)

val safe = when a {
    some(n) => n,
    none    => 0
}
```

### result<T, E>

```iris
val r: result<i64, str> = ok(42)
val e: result<i64, str> = err("failed")

val success = is_ok(r)    // true
val value   = unwrap(r)   // 42

// ? propagation (inside a result-returning function)
def parse_and_double(s: str) -> result<i64, str> {
    val n = parse_i64(s)?
    ok(n * 2)
}
```

---

## Closures

```iris
val double = |x: i64| x * 2
val result = double(5)     // 10

// Multi-expression closure
val clamp = |x: i64, lo: i64, hi: i64| {
    if x < lo { lo } else { if x > hi { hi } else { x } }
}
```

Closures capture variables from their enclosing scope.

---

## Traits and Impls

```iris
trait Shape {
    def area(self) -> f64
}

record Circle { radius: f64 }

impl Shape for Circle {
    def area(self) -> f64 {
        3.14159 * self.radius * self.radius
    }
}

val c = Circle { radius: 5.0 }
val a = c.area()
```

---

## Concurrency

### Channels

```iris
val ch = channel()
spawn {
    send(ch, 42);
}
val v = recv(ch)    // blocks until value arrives
```

### Parallel for

```iris
par for i in 0..1000 {
    heavy_work(i);
}
```

### Atomics

```iris
val counter = atomic(0)
atomic_store(counter, 100)
val v = atomic_load(counter)
atomic_add(counter, 1)
```

---

## Modules (bring)

### Stdlib modules

```iris
bring std.math
bring std.string
bring std.fs
bring std.time
bring std.testing
```

Available stdlib modules: `math`, `string`, `fmt`, `set`, `queue`, `heap`, `deque`, `bitset`, `iter`, `time`, `path`, `fs`, `json`, `csv`, `http`, `kv`, `table`, `dataset`, `dataframe`, `os`, `ffi`, `crypto`, `async`, `testing`, `log`, `ml`, `nn`.

### File modules

```iris
bring "path/to/other.iris"
```

---

## Builtins

### I/O

| Function         | Signature                  | Description                          |
|------------------|----------------------------|--------------------------------------|
| `print(v)`       | `(any) -> unit`            | Print value with newline             |
| `read_line()`    | `() -> str`                | Read a line from stdin               |
| `read_i64()`     | `() -> i64`                | Read an integer from stdin           |
| `read_f64()`     | `() -> f64`                | Read a float from stdin              |

### Math

| Function          | Signature                          | Description              |
|-------------------|------------------------------------|--------------------------|
| `abs(x)`          | `(f64) -> f64`                     | Absolute value           |
| `sqrt(x)`         | `(f64) -> f64`                     | Square root              |
| `pow(base, exp)`  | `(f64, f64) -> f64`                | Power                    |
| `sin/cos/tan(x)`  | `(f64) -> f64`                     | Trig functions           |
| `exp(x)`          | `(f64) -> f64`                     | e^x                      |
| `log(x)`          | `(f64) -> f64`                     | Natural log              |
| `log2(x)`         | `(f64) -> f64`                     | Log base 2               |
| `floor/ceil(x)`   | `(f64) -> f64`                     | Rounding                 |
| `round(x)`        | `(f64) -> f64`                     | Round to nearest         |
| `sign(x)`         | `(f64) -> f64`                     | Sign (-1, 0, 1)          |
| `min(a, b)`       | `(f64, f64) -> f64`                | Minimum                  |
| `max(a, b)`       | `(f64, f64) -> f64`                | Maximum                  |
| `clamp(x, lo, hi)`| `(f64, f64, f64) -> f64`          | Clamp to range           |

### Strings

| Function                      | Signature                        | Description              |
|-------------------------------|----------------------------------|--------------------------|
| `len(s)`                      | `(str) -> i64`                   | String length            |
| `concat(a, b)`                | `(str, str) -> str`              | Concatenate              |
| `to_str(v)`                   | `(any) -> str`                   | Convert to string        |
| `to_upper(s)` / `to_lower(s)` | `(str) -> str`                   | Case conversion          |
| `trim(s)`                     | `(str) -> str`                   | Strip whitespace         |
| `repeat(s, n)`                | `(str, i64) -> str`              | Repeat string            |
| `contains(s, sub)`            | `(str, str) -> bool`             | Substring check          |
| `starts_with(s, p)`           | `(str, str) -> bool`             | Prefix check             |
| `ends_with(s, p)`             | `(str, str) -> bool`             | Suffix check             |
| `slice(s, lo, hi)`            | `(str, i64, i64) -> str`         | Substring               |
| `find(s, sub)`                | `(str, str) -> option<i64>`      | Find index              |
| `split(s, delim)`             | `(str, str) -> list<str>`        | Split string            |
| `join(lst, delim)`            | `(list<str>, str) -> str`        | Join strings            |
| `format(fmt, ...)`            | `(str, ...) -> str`              | Format string (`{}`)    |

### Lists

| Function                | Signature                        | Description              |
|-------------------------|----------------------------------|--------------------------|
| `list()`                | `() -> list<T>`                  | Create empty list        |
| `push(lst, v)`          | `(list<T>, T) -> unit`           | Append element           |
| `list_pop(lst)`         | `(list<T>) -> T`                 | Remove and return last   |
| `list_len(lst)`         | `(list<T>) -> i64`               | Length                   |
| `list_get(lst, i)`      | `(list<T>, i64) -> T`            | Get by index             |
| `list_set(lst, i, v)`   | `(list<T>, i64, T) -> unit`      | Set by index             |
| `list_sort(lst)`        | `(list<i64>) -> list<i64>`       | Sort (ascending)         |
| `list_concat(a, b)`     | `(list<T>, list<T>) -> list<T>`  | Concatenate lists        |
| `list_slice(lst, lo, hi)`| `(list<T>, i64, i64) -> list<T>` | Sublist               |
| `list_contains(lst, v)` | `(list<T>, T) -> bool`           | Membership check         |

### Maps

| Function                | Signature                        | Description              |
|-------------------------|----------------------------------|--------------------------|
| `map()`                 | `() -> map<str, T>`              | Create empty map         |
| `map_set(m, k, v)`      | `(map, str, T) -> unit`          | Insert/update entry      |
| `map_get(m, k)`         | `(map, str) -> option<T>`        | Get by key               |
| `map_contains(m, k)`    | `(map, str) -> bool`             | Key exists               |
| `map_remove(m, k)`      | `(map, str) -> unit`             | Remove entry             |
| `map_len(m)`            | `(map) -> i64`                   | Number of entries        |

### Options and Results

| Function       | Description                                        |
|----------------|----------------------------------------------------|
| `some(v)`      | Wrap in `some`                                     |
| `none`         | The `none` value                                   |
| `is_some(opt)` | Returns `true` if the option holds a value         |
| `unwrap(opt)`  | Extracts the value, panics if `none`               |
| `ok(v)`        | Wrap in `ok`                                       |
| `err(e)`       | Wrap in `err`                                      |
| `is_ok(res)`   | Returns `true` if result is `ok`                   |

### Control

| Function       | Description                                        |
|----------------|----------------------------------------------------|
| `panic(msg)`   | Abort with an error message                        |
| `assert(cond)` | Panic if condition is false                        |

---

## CLI Reference

```
iris run file.iris              # compile + run through the LLVM/native pipeline
iris build file.iris -o out     # compile to native binary
iris test [file.iris]           # run test_ functions
iris --emit ir file.iris        # print IR
iris --emit llvm file.iris      # print LLVM IR
iris lsp                        # start LSP server (for editors)
iris pkg init                   # create iris.toml manifest
iris pkg add <name>             # add dependency
iris pkg install                # install dependencies
```
