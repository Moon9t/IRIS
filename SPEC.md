# IRIS Language Specification

**Version 0.5.0**
**Last updated: 2026-05-26**

> This document defines the syntax, semantics, and type system of the IRIS
> programming language. It serves as the authoritative reference for compiler
> implementors, tool authors, and language users.

---

## Table of Contents

1. [Notation](#1-notation)
2. [Lexical Structure](#2-lexical-structure)
3. [Types](#3-types)
4. [Declarations](#4-declarations)
5. [Expressions](#5-expressions)
6. [Statements](#6-statements)
7. [Pattern Matching](#7-pattern-matching)
8. [Functions](#8-functions)
9. [Generics and Traits](#9-generics-and-traits)
10. [Modules and Imports](#10-modules-and-imports)
11. [Concurrency](#11-concurrency)
12. [Error Handling](#12-error-handling)
13. [Automatic Differentiation](#13-automatic-differentiation)
14. [Tensors and ML](#14-tensors-and-ml)
15. [Foreign Function Interface](#15-foreign-function-interface)
16. [Standard Library](#16-standard-library)
17. [Built-in Functions](#17-built-in-functions)
18. [Compiler Pipeline](#18-compiler-pipeline)
19. [Runtime Semantics](#19-runtime-semantics)
20. [Appendix A: Complete Grammar (BNF)](#appendix-a-complete-grammar-bnf)
21. [Appendix B: Reserved Words](#appendix-b-reserved-words)
22. [Appendix C: Operator Precedence](#appendix-c-operator-precedence)

---

## 1. Notation

This specification uses a modified BNF notation:

| Notation | Meaning |
|----------|---------|
| `keyword` | Terminal: literal text |
| `UPPER` | Non-terminal: defined elsewhere |
| `[ X ]` | Optional: zero or one occurrence of X |
| `{ X }` | Repetition: zero or more occurrences of X |
| `X \| Y` | Alternative: either X or Y |
| `( X )` | Grouping |
| `"text"` | Literal string |

---

## 2. Lexical Structure

### 2.1 Source Encoding

IRIS source files are UTF-8 encoded text. The file extension is `.iris`.

### 2.2 Whitespace and Comments

Whitespace (spaces, tabs, newlines, carriage returns) separates tokens and is
otherwise insignificant.

Line comments begin with `//` and extend to the end of the line:

```iris
// This is a comment
def main() -> i64 { 0 }  // inline comment
```

Block comments are not supported in version 0.2.0.

### 2.3 Keywords

The following identifiers are reserved as keywords:

```
def    val    var    return   if     else    while   loop
break  continue  record  bring  when   choice  for    in
spawn  par    async  await   const  type    trait   impl
pub    extern  model  layer  input  output  to
```

### 2.4 Type Keywords

```
f32  f64  i32  i64  bool  tensor  str
```

### 2.5 Literals

#### Integer Literals

A sequence of ASCII digits, parsed as `i64`:

```
0    42    1000000    9223372036854775807
```

#### Floating-Point Literals

A sequence of digits containing a decimal point, with an optional exponent:

```
3.14    0.5    1.0e10    2.5E3    1.0e-5
```

Parsed as `f64`.

#### Boolean Literals

```
true    false
```

#### String Literals

Enclosed in double quotes. The following escape sequences are supported:

| Escape | Character |
|--------|-----------|
| `\n` | Newline (LF) |
| `\t` | Horizontal tab |
| `\r` | Carriage return |
| `\\` | Backslash |
| `\"` | Double quote |

```iris
"hello world"
"line1\nline2"
"tab\there"
```

#### Interpolated String Literals (f-strings)

Prefixed with `f`, containing `{identifier}` interpolation placeholders:

```iris
val name = "IRIS"
val greeting = f"Hello, {name}!"  // "Hello, IRIS!"
```

Each `{ident}` is replaced by `to_str(ident)` at compile time, and
adjacent segments are joined with `concat`.

> **Note:** Only simple identifiers are allowed inside `{}` — field access
> expressions like `{point.x}` are not supported. Extract to a local
> binding first:
>
> ```iris
> val x = point.x
> val msg = f"x = {x}"
> ```

### 2.6 Operators and Punctuation

#### Two-Character Tokens

```
->   ==   !=   <=   >=   =>   &&   ||   ..   ..=
```

#### Single-Character Tokens

```
(  )  {  }  [  ]  <  >  ,  :  ;  =  +  -  *  /  %  !  .  |  ?  @
```

### 2.7 Identifiers

An identifier starts with an ASCII letter or underscore, followed by zero or
more ASCII letters, digits, or underscores:

```
IDENT := [a-zA-Z_][a-zA-Z0-9_]*
```

Identifiers are case-sensitive. `foo`, `Foo`, and `FOO` are distinct.

---

## 3. Types

### 3.1 Scalar Types

| Type | Size | Description |
|------|------|-------------|
| `i8` | 8-bit | Signed integer |
| `u8` | 8-bit | Unsigned integer |
| `i32` | 32-bit | Signed integer |
| `u32` | 32-bit | Unsigned integer |
| `i64` | 64-bit | Signed integer (default integer type) |
| `u64` | 64-bit | Unsigned integer |
| `usize` | Pointer-width | Unsigned size type |
| `f32` | 32-bit | IEEE 754 single-precision float |
| `f64` | 64-bit | IEEE 754 double-precision float (default float type) |
| `bool` | 1-bit logical | Boolean: `true` or `false` |

### 3.2 String Type

```
str
```

UTF-8 encoded, immutable, heap-allocated string. String values support
comparison (`==`, `!=`) and concatenation via `concat()`.

### 3.3 Tensor Type

```
tensor<DTYPE, [DIM, DIM, ...]>
```

A multi-dimensional array with a fixed element type and shape. Dimensions
may be literal integers or symbolic names:

```iris
tensor<f32, [3, 4]>       // 3x4 matrix of f32
tensor<f64, [M, K]>       // symbolic dimensions M and K
tensor<f32, [batch, 784]> // named batch dimension
```

### 3.4 Array Type

```
[ELEM_TYPE; LENGTH]
```

A fixed-length, stack-allocated array:

```iris
[i64; 5]      // array of 5 i64 values
[f32; 10]     // array of 10 f32 values
```

### 3.5 Tuple Type

```
(TYPE, TYPE, ...)
```

An ordered, heterogeneous, fixed-size collection:

```iris
(i64, f64, bool)          // a 3-tuple
(str, i64)                // a 2-tuple (pair)
```

Elements are accessed by zero-based index: `val x = t.0`.

### 3.6 Record Type (Struct)

```
record NAME { FIELD: TYPE, FIELD: TYPE, ... }
```

A named product type with named fields:

```iris
record Point { x: f64, y: f64 }
```

Fields are accessed with dot notation: `point.x`.

### 3.7 Enum Type (Choice)

```
choice NAME { VARIANT, VARIANT(TYPE, ...), ... }
```

A tagged union (sum type). Variants may carry payload data:

```iris
choice Shape { Circle, Square, Triangle }
choice Option { Some(i64), None }
choice Result { Ok(i64), Err(str) }
```

Enum values are constructed with `EnumName.Variant` or `EnumName.Variant(args)`.

### 3.8 Option Type

```
option<T>
```

Built-in optional type equivalent to `Some(T) | None`:

- `some(value)` — wraps a value
- `none` — absence of value

Supports the `?` operator for early return (see §12).

### 3.9 Result Type

```
result<T, E>
```

Built-in error-handling type:

- `ok(value)` — success value of type `T`
- `err(error)` — error value of type `E`

Supports the `?` operator for error propagation (see §12).

### 3.10 Channel Type

```
channel<T>
```

A FIFO concurrent message-passing channel (see §11).

### 3.11 Atomic Type

```
atomic<T>
```

A lock-free atomically-accessible scalar value (see §11).

### 3.12 Mutex Type

```
mutex<T>
```

A mutex-protected value for shared mutable state (see §11).

### 3.13 Grad Type

```
grad<T>
```

A dual number for forward-mode automatic differentiation. Carries both a
primal value and a derivative (see §13).

### 3.14 Sparse Type

```
sparse<T>
```

A sparse representation of a tensor or array, storing only non-zero elements
(see §14).

### 3.15 List Type

```
list<T>
```

A dynamic, heap-allocated, growable sequence:

```iris
val nums: list<i64> = list()
push(nums, 42)
```

### 3.16 Map Type

```
map<K, V>
```

A hash map from keys of type `K` to values of type `V`:

```iris
val ages: map<str, i64> = map()
map_set(ages, "alice", 30)
```

### 3.17 Function Type

```
fn(PARAM_TYPES) -> RETURN_TYPE
```

The type of a function value or closure:

```iris
fn(i64) -> i64           // takes i64, returns i64
fn(i64, bool) -> str     // takes i64 and bool, returns str
```

### 3.18 Type Aliases

```
type Name = ExistingType
```

Creates a new name for an existing type:

```iris
type Matrix = tensor<f64, [3, 3]>
type Pair = (i64, i64)
```

---

## 4. Declarations

### 4.1 Function Declaration

```
[pub] [async] def NAME[TYPE_PARAMS](PARAMS) -> RETURN_TYPE BLOCK
```

```iris
def add(a: i64, b: i64) -> i64 { a + b }
pub def square(x: f64) -> f64 { x * x }
async def fetch(url: str) -> str { /* ... */ }
```

Parameters may have default values:

```iris
def greet(name: str = "world") -> str { concat("Hello, ", name) }
```

### 4.2 Record Declaration

```
[pub] record NAME { FIELD: TYPE, FIELD: TYPE, ... }
```

```iris
record Point { x: f64, y: f64 }
pub record Color { r: i64, g: i64, b: i64 }
```

### 4.3 Enum Declaration

```
[pub] choice NAME { VARIANT [( TYPES )], ... }
```

```iris
choice Direction { North, South, East, West }
choice Expr { Num(i64), Add(i64, i64) }
```

### 4.4 Constant Declaration

```
[pub] const NAME [: TYPE] = EXPR
```

```iris
const PI: f64 = 3.14159265358979
const MAX_SIZE = 1024
```

Constants are evaluated at compile time and must be literal values or
constant expressions.

### 4.5 Type Alias Declaration

```
[pub] type NAME = TYPE
```

```iris
type Vec3 = (f64, f64, f64)
```

### 4.6 Trait Declaration

```
trait NAME { METHOD_SIG METHOD_SIG ... }
```

```iris
trait Printable {
    def to_string(self: Self) -> str
}
```

### 4.7 Impl Declaration

```
impl TRAIT_NAME for TYPE_NAME { METHOD_DEF ... }
```

```iris
impl Printable for Point {
    def to_string(self: Point) -> str {
        concat(concat(to_str(self.x), ", "), to_str(self.y))
    }
}
```

### 4.8 Extern Function Declaration

```
extern def NAME(PARAMS) -> RETURN_TYPE
```

Declares a C-linkage function defined outside the IRIS source:

```iris
extern def puts(s: str) -> i64
```

### 4.9 Visibility

The `pub` keyword makes a declaration visible to importing modules. Without
`pub`, declarations are module-private. Applicable to: `def`, `record`,
`choice`, `const`, `type`.

---

## 5. Expressions

All expressions produce a value. IRIS is expression-oriented: blocks,
if/else, and when are all expressions.

### 5.1 Literal Expressions

```iris
42            // i64
3.14          // f64
true          // bool
"hello"       // str
f"x = {x}"   // interpolated str
```

### 5.2 Identifier Expression

A bare identifier evaluates to the value of the binding in scope:

```iris
val x = 10
x              // evaluates to 10
```

### 5.3 Binary Operations

```
EXPR OP EXPR
```

Arithmetic: `+`, `-`, `*`, `/`, `%`
Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
Logical: `&&` (short-circuit AND), `||` (short-circuit OR)

### 5.4 Unary Operations

```
-EXPR     // arithmetic negation
!EXPR     // boolean NOT
```

### 5.5 Function Call

```
NAME(ARGS)
```

```iris
add(1, 2)
print("hello")
sqrt(2.0)
```

### 5.6 Method Call

```
EXPR.METHOD(ARGS)
```

```iris
point.distance(other)
```

### 5.7 Field Access

```
EXPR.FIELD
```

```iris
point.x
color.r
```

### 5.8 Tuple Index

```
EXPR.INDEX
```

```iris
val t = (10, 20, 30)
t.0    // 10
t.2    // 30
```

### 5.9 Array/Tensor Index

```
EXPR[INDEX, INDEX, ...]
```

```iris
arr[0]
matrix[i, j]
```

### 5.10 Cast Expression

```
EXPR to TYPE
```

```iris
val x: i32 = 42 to i32
val y: f64 = x to f64
```

### 5.11 If Expression

> **Note:** IRIS does not have a dedicated `else if` construct. Chained
> conditions are expressed by nesting an `if` inside the `else` block:
>
> ```iris
> if x > 0 {
>     "positive"
> } else {
>     if x < 0 {
>         "negative"
>     } else {
>         "zero"
>     }
> }
> ```

```
if COND BLOCK [else BLOCK]
```

If/else is an expression — both branches must produce the same type:

```iris
val abs_x = if x < 0 { -x } else { x }
```

If without `else` returns `i64` (value 0) or can be used as a statement.

> **Caveat:** Using `if` without `else` inside loop bodies for mutable
> variable updates may produce unexpected results. Always provide an `else`
> branch when the `if` guards a mutation:
>
> ```iris
> // Correct: always assign from both branches
> var total = 0
> for x in xs {
>     total = if x > 0 { total + x } else { total }
> }
> ```

### 5.12 Block Expression

```
{ STMTS... TAIL_EXPR }
```

The value of a block is the value of its tail expression (the last expression
not followed by `;`):

```iris
val result = {
    val a = 10;
    val b = 20;
    a + b          // tail expression — block value is 30
}
```

### 5.13 When Expression

```
when SCRUTINEE { PATTERN => EXPR, ... }
```

Pattern matching (see §7).

### 5.14 Struct Literal

```
NAME { FIELD: EXPR, ... }
```

```iris
val p = Point { x: 1.0, y: 2.0 }
```

### 5.15 Array Literal

```
[EXPR, EXPR, ...]
```

```iris
val arr = [1, 2, 3, 4, 5]
```

### 5.16 Tuple Literal

```
(EXPR, EXPR, ...)
```

```iris
val pair = (42, "hello")
```

### 5.17 Lambda (Closure)

```
|PARAM: TYPE, ...| BODY_EXPR
```

```iris
val double = |x: i64| x * 2
val add = |a: i64, b: i64| a + b
```

Closures capture variables from the enclosing scope by value (lambda-lifted).

### 5.18 Try Expression

```
EXPR?
```

Propagates errors from `option<T>` or `result<T,E>` values (see §12).

### 5.19 Await Expression

```
await EXPR
```

Awaits an asynchronous computation (see §11).

### 5.20 Operator Precedence

From highest to lowest:

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `.` `[...]` | Left |
| 2 | `?` (postfix) | — |
| 3 | `-` `!` (prefix) | Right |
| 4 | `*` `/` `%` | Left |
| 5 | `+` `-` | Left |
| 6 | `to` (cast) | Left |
| 7 | `==` `!=` `<` `<=` `>` `>=` | Left |
| 8 | `&&` | Left |
| 9 | `\|\|` | Left |

---

## 6. Statements

Statements appear inside blocks. They are separated by `;` or newlines.

### 6.1 Immutable Binding (val)

```
val NAME [: TYPE] = EXPR;
```

```iris
val x = 42;
val name: str = "IRIS";
```

Once bound, `val` bindings cannot be reassigned.

### 6.2 Mutable Binding (var)

```
var NAME [: TYPE] = EXPR;
```

```iris
var count = 0;
count = count + 1;
```

### 6.3 Assignment

```
TARGET = EXPR;
```

Only valid for `var` bindings, array elements, and struct fields:

```iris
var x = 0;
x = 10;
arr[0] = 42;
point.x = 3.14;
```

### 6.4 Tuple Destructuring

```
val (NAME, NAME, ...) = EXPR;
```

```iris
val (a, b) = (10, 20);
```

### 6.5 While Loop

```
while COND { BODY }
```

```iris
var i = 0;
while i < 10 {
    print(to_str(i));
    i = i + 1;
}
```

### 6.6 Loop (Infinite)

```
loop { BODY }
```

Exit with `break`. Skip iteration with `continue`.

### 6.7 For Range Loop

```
for VAR in START..END { BODY }
```

Iterates `VAR` from `START` (inclusive) to `END` (exclusive):

```iris
for i in 0..10 { print(to_str(i)); }
```

### 6.8 For-Each Loop

```
for VAR in LIST_EXPR { BODY }
```

Iterates over elements of a list:

```iris
val items = list();
push(items, 1); push(items, 2); push(items, 3);
for item in items { print(to_str(item)); }
```

### 6.9 Parallel For

```
par for VAR in START..END { BODY }
```

Parallel range loop — iterations may execute concurrently (see §11).

### 6.10 Spawn

```
spawn { BODY }
```

Launches a concurrent task (see §11).

### 6.11 Return

```
return [EXPR];
```

Early return from a function. If `EXPR` is omitted, returns the zero value
of the return type.

### 6.12 Break and Continue

```
break;
continue;
```

`break` exits the immediately enclosing `while`, `loop`, or `for`.
`continue` skips to the next iteration.

---

## 7. Pattern Matching

The `when` expression performs exhaustive pattern matching:

```iris
when scrutinee {
    PATTERN [if GUARD] => EXPR,
    PATTERN => EXPR,
    ...
}
```

### 7.1 Pattern Types

| Pattern | Syntax | Matches |
|---------|--------|---------|
| Enum variant | `EnumName.Variant` | Tag-only variant |
| Enum with payload | `EnumName.Variant(a, b)` | Variant, binds payload to `a`, `b` |
| Option some | `some(x)` | `option` containing a value |
| Option none | `none` | Empty `option` |
| Result ok | `ok(x)` | `result` success |
| Result err | `err(e)` | `result` error |
| Integer literal | `0`, `1`, `42` | Exact integer match |
| Boolean literal | `true`, `false` | Exact boolean match |
| String literal | `"hello"` | Exact string match |
| Tuple | `(a, b)` | Destructures a tuple, binds elements |
| Range | `1..=5` | Inclusive integer range |
| Wildcard | `_` | Matches anything (catch-all) |

### 7.2 Guards

A guard expression follows the pattern with `if`:

```iris
when x {
    n if n > 0 => "positive",
    n if n < 0 => "negative",
    _ => "zero",
}
```

> **Implementation note (v0.5.0):** Guard expressions are fully supported and
> integrated natively at runtime in both JIT and compiled binary builds.

### 7.3 Exhaustiveness

The compiler checks that all possible values are covered. If a `when`
expression is non-exhaustive, a compile-time error is emitted. Use `_`
as a catch-all pattern.

---

## 8. Functions

### 8.1 Function Definition

```iris
def function_name(param1: Type1, param2: Type2) -> ReturnType {
    // body
    tail_expression
}
```

The return value is the value of the tail expression (the last expression in
the block without a trailing `;`). The `return` keyword provides early exit.

### 8.2 Recursion

Functions may call themselves recursively:

```iris
def fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
```

### 8.3 Higher-Order Functions

Functions can accept and return function values:

```iris
def apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

def main() -> i64 {
    val double = |x: i64| x * 2;
    apply(double, 21)    // 42
}
```

### 8.4 Default Parameters

```iris
def greet(name: str = "world") -> str {
    concat("Hello, ", name)
}
```

Parameters with defaults must appear after parameters without defaults.

### 8.5 Async Functions

```iris
async def fetch_data(url: str) -> str {
    // asynchronous operations
    http_get(url)
}
```

Async functions return a future that must be `await`-ed by the caller.

### 8.6 Attributes

Functions may be annotated with `@attribute`:

```iris
@differentiable
def f(x: grad<f64>) -> grad<f64> { x * x }
```

---

## 9. Generics and Traits

### 9.1 Generic Functions

Type parameters are declared in square brackets:

```iris
def identity[T](x: T) -> T { x }
def pair[A, B](a: A, b: B) -> (A, B) { (a, b) }
```

Generic functions are monomorphized: the compiler generates a specialized
version for each concrete type used at call sites.

### 9.2 Trait Definition

Traits define a set of method signatures that types must implement:

```iris
trait Printable {
    def to_string(self: Self) -> str
}

trait Comparable {
    def compare(self: Self, other: Self) -> i64
}
```

### 9.3 Trait Implementation

```iris
impl Printable for Point {
    def to_string(self: Point) -> str {
        format("({}, {})", self.x, self.y)
    }
}
```

---

## 10. Modules and Imports

### 10.1 File-Based Modules

Each `.iris` file is a module. The module name is derived from the filename.

### 10.2 Import Syntax

```
bring MODULE_PATH
```

Three forms:

```iris
bring std.math            // stdlib module
bring std.string          // stdlib module
bring "path/to/file.iris" // file-relative import
bring other_module        // project-local module
```

### 10.3 Using Imported Symbols

After `bring`, public symbols are accessed with dot notation:

```iris
bring std.math

def main() -> i64 {
    val g = math.gcd(24, 36);
    g
}
```

### 10.4 Public Exports

Only declarations marked `pub` are visible to importing modules:

```iris
// math.iris
pub def square(x: i64) -> i64 { x * x }
def helper() -> i64 { 42 }          // private — not importable
```

---

## 11. Concurrency

### 11.1 Channels

```iris
val ch = channel()         // create a channel
send(ch, 42)               // send a value
val msg = recv(ch)         // receive (blocking)
val maybe = chan_try_recv(ch)  // non-blocking receive (returns option)
val n = chan_len(ch)        // number of buffered messages
```

### 11.2 Spawn

```iris
spawn {
    // runs concurrently
    send(ch, compute_result())
}
```

`spawn` launches a concurrent task. Communication between tasks is done
through channels.

### 11.3 Parallel For

```iris
par for i in 0..n {
    output[i] = input[i] * 2.0
}
```

Iterations execute concurrently. Each iteration must be independent (no shared
mutable state within the loop body).

### 11.4 Atomics

```iris
val counter = atomic(0)
atomic_store(counter, 10)
val v = atomic_load(counter)
atomic_add(counter, 1)
```

### 11.5 Select

```iris
val result = select(ch1, ch2, ch3)   // wait on multiple channels
```

### 11.6 Timeout

```iris
val result = timeout(ch, 1000)       // recv with 1000ms timeout
```

### 11.7 Async/Await

```iris
async def slow_op() -> i64 {
    sleep(1000);
    42
}

def main() -> i64 {
    val future = slow_op();
    val result = await future;
    result
}
```

---

## 12. Error Handling

### 12.1 Option Type

```iris
def safe_div(a: i64, b: i64) -> option<i64> {
    if b == 0 { none } else { some(a / b) }
}
```

### 12.2 Result Type

```iris
def parse_number(s: str) -> result<i64, str> {
    val n = parse_i64(s);
    // parse_i64 returns result<i64, str>
    n
}
```

### 12.3 The `?` Operator

The `?` operator propagates errors upward. When applied to:

- `option<T>`: if `none`, immediately returns `none` from the enclosing function
- `result<T,E>`: if `err(e)`, immediately returns `err(e)` from the enclosing function

```iris
def process(s: str) -> result<i64, str> {
    val n = parse_i64(s)?;    // returns err() if parse fails
    val doubled = n * 2;
    ok(doubled)
}
```

### 12.4 Pattern Matching on Errors

```iris
when safe_div(10, x) {
    some(result) => print(to_str(result)),
    none         => print("division by zero"),
}
```

---

## 13. Automatic Differentiation

### 13.1 Dual Numbers

The `grad<T>` type carries a primal value and its derivative:

```iris
def f(x: grad<f64>) -> grad<f64> {
    x * x + x
}
```

### 13.2 Sparsify/Densify

```iris
val arr = [1.0, 0.0, 0.0, 3.0];
val sparse = sparsify(arr);     // sparse representation
val dense = densify(sparse);    // back to dense
```

### 13.3 Differentiable Attribute

```iris
@differentiable
def loss(x: grad<f64>, target: grad<f64>) -> grad<f64> {
    val diff = x - target;
    diff * diff
}
```

---

## 14. Tensors and ML

### 14.1 Tensor Operations

```iris
def matmul(
    a: tensor<f32, [M, K]>,
    b: tensor<f32, [K, N]>,
) -> tensor<f32, [M, N]> {
    a @ b                       // matrix multiplication via einsum
}
```

### 14.2 Model DSL

The model DSL provides a declarative way to define neural network architectures:

```iris
model MLP {
    input x: tensor<f32, [batch, 784]>
    layer h1 Linear(x, in_features=784, out_features=128)
    layer a1 ReLU(h1)
    layer h2 Linear(a1, in_features=128, out_features=10)
    output h2
}
```

### 14.3 Shape Checking

The compiler verifies tensor shape compatibility at compile time:

- Matrix multiplication: `[M, K] @ [K, N] → [M, N]`
- Element-wise operations: shapes must match exactly
- Broadcasting: supported for scalar-tensor operations

---

## 15. Foreign Function Interface

### 15.1 C FFI

```iris
bring std.ffi

val lib = ffi_open("libm.so")
val result = ffi_call_f64(lib, "sqrt", 144.0)
ffi_close(lib)
```

Functions: `ffi_open`, `ffi_call_i64`, `ffi_call_f64`, `ffi_call_str`,
`ffi_call_void`, `ffi_close`.

### 15.2 Python FFI

```iris
val result = python_eval("2 ** 10")         // evaluate expression
python_exec("import numpy as np")           // execute statement
val output = python_call("np.dot", a, b)    // call function
val ver = python_version()                  // Python version string
```

### 15.3 Rust FFI

```iris
val lib = rust_lib_open("mylib.dll")
val n = rust_call_i64(lib, "compute", 42)
```

### 15.4 Extern Declarations

```iris
extern def c_function(x: i64) -> i64
```

Declares a function with C linkage that is resolved at link time.

---

## 16. Standard Library

IRIS ships with 34 standard library modules, imported via `bring std.NAME`:

| Module | Contents |
|--------|----------|
| `std.math` | `gcd`, `lcm`, `abs_i64`, `is_even`, `is_odd`, `factorial`, `pow_i64` |
| `std.string` | `pad_left`, `pad_right`, `words`, `lines`, `title_case`, `snake_case` |
| `std.fmt` | `sprintf`, `pad_int`, `zero_pad_int`, `format_table` |
| `std.fs` | `read_text`, `write_text`, `path_exists`, `file_lines` |
| `std.json` | `json_stringify`, `json_parse` |
| `std.csv` | `csv_parse_row`, `csv_emit_row` |
| `std.http` | `http_get`, `http_post` |
| `std.time` | `now`, `sleep`, `elapsed` |
| `std.stochastic` | `normal`, `brownian_path`, `gbm_path` |
| `std.crypto` | `sha256`, `uuid`, `hex_encode`, `hex_decode` |
| `std.ffi` | `ffi_open`, `ffi_call_*`, `python_*`, `rust_*` |
| `std.os` | `env_get`, `env_set`, `exec_cmd`, `pid`, `exit_code` |
| `std.testing` | `assert_eq`, `assert_ne`, `assert_true`, `assert_false`, `assert_str_eq` |
| `std.log` | `log_info`, `log_warn`, `log_error`, `log_debug` |
| `std.iter` | `map_list`, `filter_list`, `reduce_list`, `zip_list` |
| `std.set` | Set operations (add, remove, contains, union, intersect) |
| `std.queue` | FIFO queue (enqueue, dequeue, peek) |
| `std.heap` | Priority queue / min-heap (insert, extract_min) |
| `std.deque` | Double-ended queue (push_front, push_back, pop_front, pop_back) |
| `std.kv` | Key-value store (SQLite-backed persistent storage) |
| `std.table` | Tabular data operations |
| `std.dataset` | ML dataset abstraction (load, split, batch) |
| `std.dataframe` | DataFrame-like API |
| `std.path` | Path manipulation (join, parent, extension) |
| `std.async` | Async runtime helpers |
| `std.bitset` | Bit array operations (set, clear, test, count) |

---

## 17. Built-in Functions

Built-in functions are available without `bring` and are resolved by the
compiler. They are grouped by category:

### 17.1 Math

`sin`, `cos`, `tan`, `exp`, `log`, `log2`, `sqrt`, `abs`, `floor`, `ceil`,
`round`, `sign`, `pow`, `min`, `max`, `clamp`, `math_pi`, `math_e`,
`is_nan`, `is_inf`

### 17.2 String

`len`, `concat`, `contains`, `starts_with`, `ends_with`, `to_upper`,
`to_lower`, `trim`, `repeat`, `to_str`, `format`, `split`, `join`, `find`,
`slice`, `str_index`, `str_replace`, `str_reverse`, `char_at`,
`str_pad_left`, `str_pad_right`, `str_chars`, `str_bytes`, `str_count`

### 17.3 I/O

`print`, `read_line`, `read_i64`, `read_f64`

### 17.4 Collections

`list`, `push`, `pop`, `list_get`, `list_set`, `list_len`, `list_map`,
`list_filter`, `list_reduce`, `list_any`, `list_all`, `list_zip`,
`list_enumerate`, `list_flatten`, `list_unique`, `list_reverse`,
`list_sorted`, `list_sum`, `list_min`, `list_max`

### 17.5 Map

`map`, `map_get`, `map_set`, `map_contains`, `map_remove`, `map_keys`,
`map_values`, `map_len`

### 17.6 Parsing

`parse_i64`, `parse_f64`, `json_stringify`, `regex_match`,
`regex_find_all`, `regex_replace`

### 17.7 System

`cwd`, `list_dir`, `mkdir`, `remove_file`, `path_join`, `env_get`,
`env_set`, `exec_cmd`, `pid`, `exit_code`, `type_of`

### 17.8 Random

`random`, `random_range`, `uuid`

### 17.9 Crypto

`sha256`, `hash`, `hex_encode`, `hex_decode`, `base64_encode`,
`base64_decode`

### 17.10 Concurrency

`channel`, `send`, `recv`, `spawn`, `chan_try_recv`, `chan_len`, `select`,
`timeout`, `thread_count`, `atomic`, `atomic_load`, `atomic_store`,
`atomic_add`

### 17.11 Date/Time

`datetime_now`, `datetime_timestamp`, `datetime_format`

---

## 18. Compiler Pipeline

### 18.1 Phases

```text
Source (.iris)
    |
    v
  Lexer         — source text → token stream
    |
    v
  Parser        — tokens → Abstract Syntax Tree (AST)
    |
    v
  Lowerer       — AST → SSA Intermediate Representation (IrModule)
    |
    v
  Pass Pipeline — analysis and optimization passes
    |
    v
  Code Generator — one of:
    ├── Interpreter    (--emit eval)
    ├── IR Printer     (--emit ir)
    ├── LLVM IR        (--emit llvm)
    ├── Native Binary  (--emit binary, via clang)
    ├── ONNX           (--emit onnx)
    ├── CUDA           (--emit cuda)
    └── SIMD           (--emit simd)
```

### 18.2 SSA IR Design

IRIS uses a block-parameter SSA form (MLIR-style):

- No phi nodes — branch arguments carry values directly
- Blocks have typed parameters
- Instructions produce named values (`%0`, `%1`, ...)
- Each value is defined exactly once (single static assignment)

### 18.3 Optimization Passes

Applied in order:

| Pass | Purpose |
|------|---------|
| `ValidatePass` | Structural SSA correctness |
| `TypeInferPass` | Type consistency and inference |
| `ConstFoldPass` | Constant arithmetic and identity simplification |
| `OpExpandPass` | Expand element-wise calls to tensor operations |
| `DcePass` | Dead code elimination |
| `CsePass` | Common subexpression elimination |
| `ShapeCheckPass` | Tensor shape consistency |
| `InlinePass` | Function inlining (small functions) |
| `LoopUnrollPass` | Loop unrolling for small fixed-trip-count loops |
| `StrengthReducePass` | Replace expensive operations with cheaper equivalents |
| `ExhaustivePass` | Pattern match exhaustiveness checking |
| `GcAnnotatePass` | Garbage collection annotation |

### 18.4 Native Compilation

`iris build` emits LLVM IR and invokes `clang` to produce a native binary.
The C runtime ([iris_runtime.c](src/runtime/iris_runtime.c)) provides:

- Heap-allocated values (tagged union `IrisVal`)
- Reference counting
- List, map, and string operations
- Channel and concurrency primitives
- FFI bridge functions

---

## 19. Runtime Semantics

### 19.1 Evaluation Order

Expressions are evaluated left-to-right. Function arguments are evaluated
left-to-right before the call. Short-circuit operators (`&&`, `||`) skip
the right operand when the result is determined.

### 19.2 Memory Management

IRIS uses reference counting for heap-allocated values (strings, lists, maps,
channels). Stack-allocated values (scalars, small tuples, arrays) are value
types.

### 19.3 Integer Overflow

Integer arithmetic (`i64`) wraps on overflow (two's complement). This
matches the behavior of the underlying C runtime.

### 19.4 Floating-Point Semantics

IEEE 754 double-precision (for `f64`) and single-precision (for `f32`).
`NaN`, `+Inf`, `-Inf` are representable. `NaN != NaN`.

### 19.5 String Semantics

Strings are immutable UTF-8 byte sequences. String operations that produce
new strings allocate fresh memory. Comparison is bytewise.

### 19.6 Panic

Unrecoverable errors (division by zero with integers, index out of bounds)
cause a runtime panic with a diagnostic message. The program terminates with
a non-zero exit code.

---

## Appendix A: Complete Grammar (BNF)

```bnf
module      ::= { top_level }
top_level   ::= function_def
              | record_def
              | enum_def
              | const_def
              | type_alias
              | trait_def
              | impl_def
              | bring_decl
              | extern_def
              | model_def

bring_decl  ::= "bring" bring_path
bring_path  ::= IDENT { "." IDENT }
              | STRING_LIT

function_def ::= [ "pub" ] [ "async" ] "def" IDENT [ type_params ] "(" params ")" "->" type block
type_params  ::= "[" IDENT { "," IDENT } "]"
params       ::= [ param { "," param } ]
param        ::= IDENT ":" type [ "=" expr ]

record_def  ::= [ "pub" ] "record" IDENT "{" field_defs "}"
field_defs  ::= field_def { "," field_def }
field_def   ::= IDENT ":" type

enum_def    ::= [ "pub" ] "choice" IDENT "{" variant_defs "}"
variant_defs ::= variant_def { "," variant_def }
variant_def  ::= IDENT [ "(" type { "," type } ")" ]

const_def   ::= [ "pub" ] "const" IDENT [ ":" type ] "=" expr

type_alias  ::= [ "pub" ] "type" IDENT "=" type

trait_def   ::= "trait" IDENT "{" { trait_method } "}"
trait_method ::= "def" IDENT "(" params ")" "->" type

impl_def    ::= "impl" IDENT "for" IDENT "{" { function_def } "}"

extern_def  ::= "extern" "def" IDENT "(" params ")" "->" type

model_def   ::= "model" IDENT "{" { model_item } "}"
model_item  ::= "input" IDENT ":" type
              | "layer" IDENT IDENT [ "(" layer_args ")" ]
              | "output" IDENT
layer_args  ::= layer_arg { "," layer_arg }
layer_arg   ::= IDENT "=" expr | IDENT

(* Statements *)
block       ::= "{" { stmt } [ expr ] "}"
stmt        ::= let_stmt
              | assign_stmt
              | while_stmt
              | loop_stmt
              | for_stmt
              | par_for_stmt
              | spawn_stmt
              | return_stmt
              | break_stmt
              | continue_stmt
              | expr ";"

let_stmt    ::= "val" IDENT [ ":" type ] "=" expr ";"
              | "var" IDENT [ ":" type ] "=" expr ";"
              | "val" "(" IDENT { "," IDENT } ")" "=" expr ";"
assign_stmt ::= expr "=" expr ";"
while_stmt  ::= "while" expr block
loop_stmt   ::= "loop" block
for_stmt    ::= "for" IDENT "in" expr ".." expr block
              | "for" IDENT "in" expr block
par_for_stmt ::= "par" "for" IDENT "in" expr ".." expr block
spawn_stmt  ::= "spawn" block
return_stmt ::= "return" [ expr ] ";"
break_stmt  ::= "break" ";"
continue_stmt ::= "continue" ";"

(* Expressions — from lowest to highest precedence *)
expr        ::= or_expr
or_expr     ::= and_expr { "||" and_expr }
and_expr    ::= cmp_expr { "&&" cmp_expr }
cmp_expr    ::= add_expr { ( "==" | "!=" | "<" | "<=" | ">" | ">=" ) add_expr }
add_expr    ::= mul_expr { ( "+" | "-" ) mul_expr }
mul_expr    ::= cast_expr { ( "*" | "/" | "%" ) cast_expr }
cast_expr   ::= unary_expr [ "to" type ]
unary_expr  ::= [ "-" | "!" ] postfix_expr
postfix_expr ::= primary { "." IDENT [ "(" args ")" ] | "." INT_LIT | "[" args "]" | "?" }

primary     ::= INT_LIT
              | FLOAT_LIT
              | BOOL_LIT
              | STRING_LIT
              | FSTRING_LIT
              | IDENT [ "::" IDENT ] [ "(" args ")" ]
              | IDENT "{" field_inits "}"
              | "(" expr { "," expr } ")"
              | "[" [ expr { "," expr } ] "]"
              | "|" params "|" expr
              | "if" expr block [ "else" block ]
              | "when" expr "{" when_arms "}"
              | "await" expr
              | block

args        ::= [ expr { "," expr } ]
field_inits ::= [ IDENT ":" expr { "," IDENT ":" expr } ]

when_arms   ::= when_arm { "," when_arm }
when_arm    ::= pattern [ "if" expr ] "=>" expr
pattern     ::= IDENT "." IDENT [ "(" bindings ")" ]
              | "some" "(" IDENT ")"
              | "none"
              | "ok" "(" IDENT ")"
              | "err" "(" IDENT ")"
              | INT_LIT [ "..=" INT_LIT ]
              | BOOL_LIT
              | STRING_LIT
              | "(" pattern { "," pattern } ")"
              | "_"
bindings    ::= [ IDENT { "," IDENT } ]

(* Types *)
type        ::= scalar_type
              | "tensor" "<" scalar_type "," "[" dims "]" ">"
              | "option" "<" type ">"
              | "result" "<" type "," type ">"
              | "channel" "<" type ">"
              | "atomic" "<" type ">"
              | "mutex" "<" type ">"
              | "grad" "<" type ">"
              | "sparse" "<" type ">"
              | "list" "<" type ">"
              | "map" "<" type "," type ">"
              | "[" type ";" INT_LIT "]"
              | "(" type { "," type } ")"
              | "(" [ type { "," type } ] ")" "->" type
              | IDENT  (* named struct/enum/alias *)

scalar_type ::= "i8" | "u8" | "i32" | "u32" | "i64" | "u64" | "usize"
              | "f32" | "f64" | "bool" | "str"

dims        ::= dim { "," dim }
dim         ::= INT_LIT | IDENT
```

---

## Appendix B: Reserved Words

All keywords listed in §2.3 and §2.4 are reserved and may not be used as
identifiers.

Additionally, the following words are reserved for future use:

```
match   enum   struct   fn   let   mut   use   mod   self   Self
super   crate  where    as   ref   move  dyn   box   yield  macro
```

---

## Appendix C: Operator Precedence

| Precedence | Category | Operators | Associativity |
|------------|----------|-----------|---------------|
| 1 (highest) | Postfix | `.field` `.method()` `[index]` `?` | Left |
| 2 | Prefix | `-` (negate) `!` (not) | Right |
| 3 | Multiplicative | `*` `/` `%` | Left |
| 4 | Additive | `+` `-` | Left |
| 5 | Cast | `to` | Left |
| 6 | Comparison | `==` `!=` `<` `<=` `>` `>=` | Left, non-chaining |
| 7 | Logical AND | `&&` | Left, short-circuit |
| 8 (lowest) | Logical OR | `\|\|` | Left, short-circuit |

---

*This specification is a living document. It will be updated as the language
evolves toward 1.0. For questions or clarifications, open a GitHub issue
with the tag `spec`.*
