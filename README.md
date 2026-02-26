# IRIS — Intermediate Representation Inference System

IRIS is a compiled, statically-typed ML-focused DSL written in Rust.
The goal is to be the "C of ML": low-level control, high-level ML ergonomics,
first-class tensor/gradient/sparsity types, and static safety guarantees for
parallel training workloads.

Source files use the `.iris` extension and compile through a full pipeline:
`.iris` → Lexer → Parser → AST → Lowerer → SSA IR → Passes → Codegen/Interpreter

---

## Language Overview

### Types

| Type | Syntax | Notes |
| ---- | ------ | ----- |
| Scalars | `i32`, `i64`, `f32`, `f64`, `bool` | |
| Tensors | `tensor<f32, [M, K]>` | Symbolic + literal dims |
| Strings | `str` | UTF-8, immutable |
| Arrays | `[i64; 5]` | Fixed-size |
| Tuples | `(i64, f64, bool)` | Heterogeneous |
| Records | `record Point { x: f64, y: f64 }` | Named fields |
| Enums | `choice Color { Red, Green }` | Sum types |
| Closures | `\|x: i64\| x * 2` | Lambda-lifted |
| Options | `option<T>` | `some(v)` / `none` |
| Results | `result<T, E>` | `ok(v)` / `err(e)` |
| Lists | `list<T>` | Dynamic, heap-allocated |
| Maps | `map<K, V>` | Hash map |
| Channels | `channel<T>` | Concurrent message passing |
| Grad | `grad<T>` | Dual numbers for autodiff |
| Sparse | `sparse<T>` | Sparse tensor wrapper |
| Atomics | `atomic<T>` | Lock-free scalar |

### Functions and Bindings

```iris
def add(a: i64, b: i64) -> i64 {
    a + b
}

def example() -> i64 {
    val x = 10          // immutable binding
    var count = 0       // mutable binding
    count = count + 1
    add(x, count)       // tail expression is return value
}
```

### Control Flow

```iris
// if/else (expression)
val abs_x = if x < 0 { -x } else { x }

// while loop
while count < 10 {
    count = count + 1
}

// loop with break / continue
loop {
    if done { break }
}

// for range
for i in 0..n {
    output[i] = relu(input[i])
}

// parallel for
par for i in 0..n {
    output[i] = input[i] * 2.0
}

// early return
def find(arr: [i64; 10], target: i64) -> i64 {
    for i in 0..10 {
        if arr[i] == target { return i }
    }
    -1
}
```

### Records and Enums

```iris
record Point { x: f64, y: f64 }

def midpoint(a: Point, b: Point) -> Point {
    Point { x: (a.x + b.x) / 2.0, y: (a.y + b.y) / 2.0 }
}

choice Shape { Circle, Square, Triangle }

def describe(s: Shape) -> i64 {
    when s {
        Shape.Circle   => 0,
        Shape.Square   => 1,
        Shape.Triangle => 2,
    }
}
```

### Tensors and Einsum

```iris
def matmul(a: tensor<f32, [M, K]>, b: tensor<f32, [K, N]>) -> tensor<f32, [M, N]> {
    einsum("mk,kn->mn", a, b)
}
```

### Options and Results

```iris
def safe_div(a: i64, b: i64) -> option<i64> {
    if b == 0 { none } else { some(a / b) }
}

def use_result(r: result<i64, str>) -> i64 {
    when r {
        ok(v) => v,
        err(e) => 0,
    }
}

// ? operator propagates errors
def parse_and_add(s: str) -> result<i64, str> {
    val n = parse_i64(s)?
    ok(n + 1)
}
```

### Generics and Traits

```iris
def identity[T](x: T) -> T { x }

trait Show {
    def show(self: Self) -> str
}

impl Show for i64 {
    def show(self: i64) -> str { to_str(self) }
}
```

### Modules

```iris
// math.iris
pub def square(x: i64) -> i64 { x * x }

// main.iris
bring math
def f() -> i64 { math.square(5) }
```

### Closures

```iris
def apply(f: fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}

def double_it() -> i64 {
    val double = |x: i64| x * 2
    apply(double, 21)   // 42
}
```

### Concurrency

```iris
// Channels and spawn
def producer(ch: channel<i64>) -> i64 {
    send(ch, 42)
    0
}

def main() -> i64 {
    val ch = channel()
    spawn { producer(ch) }
    recv(ch)
}

// Async/await
async def fetch() -> i64 { 42 }

def run() -> i64 {
    await fetch()
}
```

### Autodiff and Sparse

```iris
def f(x: grad<f64>) -> grad<f64> {
    x * x
}

def sparse_example(arr: [f64; 4]) -> [f64; 4] {
    val s = sparsify(arr)
    densify(s)
}
```

### Global Constants and Type Aliases

```iris
const MAX_SIZE: i64 = 1024

type Matrix = tensor<f32, [N, M]>
```

### Collections

```iris
def list_example() -> i64 {
    val xs = list()
    push(xs, 1)
    push(xs, 2)
    list_len(xs)   // 2
}

def map_example() -> i64 {
    val m = map()
    map_set(m, "key", 99)
    map_get(m, "key")
}
```

### Builtins

**Math:** `sin`, `cos`, `tan`, `exp`, `log`, `log2`, `sqrt`, `abs`, `floor`, `ceil`,
`round`, `sign`, `pow`, `min`, `max`, `clamp(x, lo, hi)`

**Bitwise:** `band`, `bor`, `bxor`, `shl`, `shr`, `bitnot`

**String:** `len`, `concat`, `contains`, `starts_with`, `ends_with`, `to_upper`,
`to_lower`, `trim`, `repeat`, `to_str`, `format("Hello {}", name)`,
`str_index`, `slice`, `find`, `str_replace`

**I/O:** `print(v)`, `read_line()`, `read_i64()`, `read_f64()`

**Parse:** `parse_i64(s)` → `option<i64>`, `parse_f64(s)` → `option<f64>`

**Control:** `panic(msg)`, `assert(cond)`

---

## Compiler Pipeline

```text
.iris source
    │
    ▼
Lexer (src/parser/lexer.rs)
    │  tokens
    ▼
Parser (src/parser/parse.rs)
    │  AST
    ▼
Lowerer (src/lower/mod.rs)
    │  IrModule (SSA IR)
    ▼
Pass Pipeline (src/pass/)
    │  1. ValidatePass    — SSA invariants, rejects IrType::Infer
    │  2. TypeInferPass   — type consistency
    │  3. ConstFoldPass   — constant arithmetic + identity folding
    │  4. OpExpandPass    — activation calls → TensorOp::Unary
    │  5. DcePass         — dead code elimination
    │  6. CsePass         — common subexpression elimination
    │  7. ShapeCheckPass  — einsum notation validation
    ▼
Codegen / Interpreter
    ├── --emit ir      IR text printer (deterministic)
    ├── --emit llvm    LLVM IR text (with globals, declares, GEP for strings)
    ├── --emit onnx    ONNX binary protobuf
    └── --emit eval    Tree-walking IR interpreter
```

**IR design:** Block-parameter SSA (MLIR-style). No phi nodes — branch
arguments carry values directly. Index-based arenas (`Vec<T>` indexed by
newtype IDs `BlockId(u32)`, `ValueId(u32)`).

---

## Project Structure

```text
src/
  main.rs          Binary entry point (CLI)
  lib.rs           Library root; exports compile(), compile_multi()
  cli.rs           Argument parsing, EmitKind dispatch
  error.rs         All error types: ParseError, LowerError, PassError, CodegenError, InterpError
  parser/
    lexer.rs       Token stream
    ast.rs         AST node types
    parse.rs       Recursive-descent parser
  ir/
    mod.rs         Re-exports
    types.rs       IrType enum (Scalar, Tensor, Str, Tuple, Array, Fn, ...)
    instr.rs       IrInstr enum — central IR node (all passes touch this)
    block.rs       IrBlock + block parameters
    function.rs    IrFunction (flat Vec<IrBlock>)
    module.rs      IrModule + IrFunctionBuilder (builder pattern)
    value.rs       ValueId newtype + ValueDef
  lower/
    mod.rs         AST → IR lowering; all intrinsics; generics; traits; modules
  pass/
    mod.rs         Pass trait + PassManager
    validate.rs    SSA structural validation
    type_infer.rs  Type consistency checking
    const_fold.rs  Constant folding
    opt.rs         DCE + CSE
    expand.rs      OpExpand
    shape.rs       ShapeCheck
  interp/
    mod.rs         Tree-walking IR interpreter; IrValue enum
  codegen/
    printer.rs     Deterministic IR text emitter
    llvm_stub.rs   LLVM IR emitter (global strings, GEP, runtime declares)
    onnx.rs        ONNX binary protobuf emitter
  proto/
    encode.rs      Varint + protobuf field encoding
tests/
  ir_construction.rs   IR builder API tests
  parse_lower.rs       Parser + lowerer integration
  pass_pipeline.rs     Pass manager tests
  graph_lower.rs       Graph lowering tests
  model_parse.rs       Model DSL tests
  phase4.rs .. phase48.rs   Phase integration tests (411 tests total)
```

---

## Building

Requires Rust stable. No external dependencies other than `thiserror`.

```sh
cargo build
cargo build --release
```

Run the test suite (411 tests, all passing):

```sh
cargo test
```

---

## CLI Usage

```sh
# Emit SSA IR text
cargo run -- --emit ir examples/mlp.iris

# Emit LLVM IR
cargo run -- --emit llvm examples/mlp.iris

# Emit ONNX binary
cargo run -- --emit onnx examples/mlp.iris

# Evaluate (interpret) — prints return value
cargo run -- --emit eval examples/mlp.iris
```

---

## Programmatic API

```rust
use iris::{compile, compile_multi, EmitKind};

// Single module
let ir = compile(src, "my_module", EmitKind::Ir)?;
let result = compile(src, "my_module", EmitKind::Eval)?;

// Multi-module (imports resolved in order)
let result = compile_multi(&[
    ("math", math_src),
    ("main", main_src),
], EmitKind::Eval)?;
```

---

## Implementation Status

| Phase | Feature | Status |
| ----- | ------- | ------ |
| 1–3 | Lexer, parser, lowerer core | Done |
| 4 | Graph ops, ONNX emission | Done |
| 5 | DCE, CSE, OpExpand, ShapeCheck | Done |
| 6 | ConstFold, if-else, BatchNorm | Done |
| 7 | Unary ops, LLVM stub, Conv2D | Done |
| 8 | CmpNe/Gt/Ge, LLVM Load/Store, TypeInfer | Done |
| 9 | `while` / `loop` / `break` / `continue` | Done |
| 10 | Tensor indexing, modulo, casts | Done |
| 11 | IR interpreter, `EmitKind::Eval` | Done |
| 12 | Diagnostics, CLI, `render_error` | Done |
| 13 | Record types, `MakeStruct`/`GetField` | Done |
| 14 | ONNX binary protobuf encoding | Done |
| 15 | Enum types (`choice`/`when`), `MakeVariant` | Done |
| 16 | `var` keyword, mutable rebinding | Done |
| 17 | Inter-function calls, cross-fn eval | Done |
| 18 | `for i in start..end` range loops | Done |
| 19 | Tuple types, element access | Done |
| 20 | Logical `&&` / `\|\|`, early `return` | Done |
| 21 | String type (`str`, `concat`, `len`, `print`) | Done |
| 22 | Array types (`[T; N]`, literals, indexing) | Done |
| 23 | Closures (`\|x\| expr`, lambda-lifting) | Done |
| 24 | `option<T>`: `some`, `none`, `is_some`, `unwrap`, `when` | Done |
| 25 | `result<T,E>`: `ok`, `err`, `is_ok`, `?` operator | Done |
| 26 | Channels: `channel()`, `send`, `recv`, `spawn` | Done |
| 27 | `par for` parallel range loops | Done |
| 28 | `async`/`await` desugaring | Done |
| 29 | `barrier()`, `parallel_reduce` | Done |
| 30 | `grad<T>` forward-mode autodiff, dual numbers | Done |
| 31 | `sparse<T>`: `sparsify`/`densify` | Done |
| 32 | Math builtins: `sqrt`, `abs`, `floor`, `ceil`, `pow`, `min`, `max` | Done |
| 33 | Extended string ops: `contains`, `starts_with`, `to_upper`, `trim`, … | Done |
| 34 | Bitwise ops: `band`, `bor`, `bxor`, `shl`, `shr`, `bitnot` | Done |
| 35 | Global constants: `const NAME: Type = expr` | Done |
| 36 | Extended math: `sin`, `cos`, `exp`, `log`, `round`, `sign`, `clamp` | Done |
| 37 | `panic` / `assert` | Done |
| 38 | Type aliases: `type Name = Type` | Done |
| 39 | `to_str`, `format("Hello {}", x)` | Done |
| 40 | User input: `read_line`, `read_i64`, `read_f64` | Done |
| 41 | `parse_i64` / `parse_f64` → `option<T>` | Done |
| 42 | `str_index`, `slice`, `find`, `str_replace` | Done |
| 43 | `list<T>`: `list()`, `push`, `list_len`, `list_get`, `list_set`, `list_pop` | Done |
| 44 | `map<K,V>`: `map()`, `map_set`, `map_get`, `map_contains`, `map_remove`, `map_len` | Done |
| 45 | Generic functions: `def f[T](...)`, monomorphization | Done |
| 46 | Trait system: `trait`/`impl`, static dispatch | Done |
| 47 | Module system: `bring mod`, `pub def`, `compile_multi` | Done |
| 48 | LLVM IR: target triple/datalayout, global strings, GEP, runtime declares | Done |

---

## License

GNU General Public License v2.0 or later. See [LICENSE](LICENSE).
