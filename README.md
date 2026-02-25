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
| Arrays | `[i64; 5]` | Fixed-size, stack-allocated |
| Tuples | `(i64, f64, bool)` | Heterogeneous |
| Records | `record Point { x: f64, y: f64 }` | Named fields |
| Enums | `choice Color { Red, Green, Blue }` | Sum types |
| Closures | `\|x: i64\| x * 2` | Lambda-lifted to top-level |

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

// loop with break
loop {
    if done { break }
}

// for range
for i in 0..n {
    output[i] = relu(input[i])
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

### Strings and Arrays

```iris
def greet(name: str) -> str {
    concat("Hello, ", name)
}

def sum_array() -> i64 {
    val arr = [1, 2, 3, 4, 5]
    var total = 0
    for i in 0..5 { total = total + arr[i] }
    total
}
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
    ├── --emit llvm    LLVM IR text stub
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
  lib.rs           Library root; exports compile()
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
    mod.rs         AST → IR lowering; einsum intrinsic; lambda lifting
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
    llvm_stub.rs   LLVM IR text stub emitter
    onnx.rs        ONNX binary protobuf emitter
  proto/
    encode.rs      Varint + protobuf field encoding
tests/
  ir_construction.rs   IR builder API tests
  parse_lower.rs       Parser + lowerer integration
  pass_pipeline.rs     Pass manager tests
  graph_lower.rs       Graph lowering tests
  model_parse.rs       Model DSL tests
  phase4.rs  .. phase22.rs   Phase integration tests
```

---

## Building

Requires Rust stable. No external dependencies other than `thiserror`.

```sh
cargo build
cargo build --release
```

Run the test suite (203 tests, all passing):

```sh
cargo test
```

---

## CLI Usage

```sh
# Emit SSA IR text
cargo run -- --emit ir examples/mlp.iris

# Emit LLVM IR stub
cargo run -- --emit llvm examples/mlp.iris

# Emit ONNX binary
cargo run -- --emit onnx examples/mlp.iris

# Evaluate (interpret) — prints return value
cargo run -- --emit eval examples/mlp.iris
```

---

## Programmatic API

```rust
use iris::{compile, EmitKind};

let src = r#"
def add(a: i64, b: i64) -> i64 { a + b }
"#;

let ir_text = compile(src, "my_module", EmitKind::Ir)?;
let result  = compile(src, "my_module", EmitKind::Eval)?;
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
| 9 | while / loop / break / continue | Done |
| 10 | Tensor indexing, modulo, casts | Done |
| 11 | IR interpreter, EmitKind::Eval | Done |
| 12 | Diagnostics, CLI, render_error | Done |
| 13 | Record types, MakeStruct/GetField | Done |
| 14 | ONNX binary protobuf encoding | Done |
| 15 | Enum types (choice/when), MakeVariant | Done |
| 16 | `var` keyword, mutable rebinding | Done |
| 17 | Inter-function calls, cross-fn eval | Done |
| 18 | `for i in start..end` range loops | Done |
| 19 | Tuple types, destructuring | Done |
| 20 | Logical `&&` / `\|\|`, early `return` | Done |
| 21 | String type (`str`, `concat`, `len`, `print`) | Done |
| 22 | Array types (`[T; N]`, literals, indexing) | Done |
| 23 | Closures (`\|x\| expr`, lambda-lifting) | IR done, tests pending |
| 24–25 | Generic functions, Option type | Planned |
| 26–31 | Concurrency (channels, async, par-for, atomic, ownership) | Planned |
| 32–37 | ML types (`grad<T>`, sparse, quantized, device, distributions, autograd) | Planned |
| 38–42 | Module system, traits, pattern matching, panic, math builtins | Planned |
| 43–47 | Backends (real LLVM IR, CUDA, SIMD, JIT, PGO) | Planned |

---

## License

GNU General Public License v2.0 or later. See [LICENSE](LICENSE).
