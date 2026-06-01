# IRIS — Intermediate Representation for Intelligent Systems.

**A compiled, statically-typed systems programming language engineered for Autonomous Intelligent Systems (AIS).**

IRIS bridges low-level hardware control and execution efficiency with high-level machine learning and cognitive ergonomics. It compiles directly to native binaries via an LLVM pipeline, utilizes a deterministic reference-counting garbage collector, and features built-in multi-dimensional tensors, tape-based automatic differentiation, and a multi-level capability-based security sandbox.

[![CI](https://github.com/moon9t/iris/actions/workflows/ci.yml/badge.svg)](https://github.com/moon9t/iris/actions)
[![Release](https://img.shields.io/github/v/release/moon9t/iris?label=release)](https://github.com/moon9t/iris/releases)
[![License](https://img.shields.io/badge/license-GPL--2.0--or--later-blue.svg)](LICENSE)

---

## Technical Architecture Overview

IRIS is designed to be the foundational substrate for next-generation intelligent agents, robocar controllers, and high-throughput decision-making systems.

```text
.iris source
     │
     ▼
Lexer → Parser → AST → Lowerer
                         │
                         ▼
                    Block-Parameter SSA IR
                         │
                         ▼
                    Pass Pipeline (15 Optimizer Passes)
                         │
                         ▼
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
     LLVM Codegen  CUDA Backend    ONNX Protobuf
           │             │              │
           ▼             ▼              ▼
        Clang          NVPTX       ONNX Binary
           │
           ▼
     Native Binary
```

### Key Architectural Pillars

1. **Native Performance & Optimization:** Compiles to highly optimized native machine code using LLVM. The compiler pipeline includes 15 distinct passes (such as LICM, CSE, DCE, and Strength Reduction) and supports SIMD auto-vectorization and a CUDA/NVPTX backend for GPU kernel execution.
2. **First-Class Compute Engine:** Tensors are native types (`tensor<f32, [M, K]>`) with compile-time symbolic shape-checking. Includes a general Einstein Summation (`einsum`) contraction engine, dense/sparse tensor conversions, and tape-based reverse-mode automatic differentiation.
3. **Deterministic Memory Management:** Employs a zero-pause reference-counting garbage collector embedded in the C runtime, utilizing an efficient side-table tracing mechanism (`iris_retain`, `iris_release`) with deep-free semantics to prevent latency spikes common in tracing GCs.
4. **Capability-Based Security Sandbox:** A robust security audit layer allows fine-grained, run-time restriction of system capabilities (filesystem reads/writes, outbound connections, process spawning, and FFI). Includes path traversal sanitization and audit log exports.
5. **Actor-Model & Async Concurrency:** First-class async tasks, awaitable futures, light-weight thread spawning, atomic variables, and native thread-safe channels (`channel<T>`) for secure concurrent message passing.
6. **Multi-Platform FFI:** Zero-cost dynamic linking with C libraries, a Python runtime wrapper for embedded scripting, and native Rust cdylib integrations.

---

## Quick Start

### Installation

For development, compile IRIS directly from source using Cargo (requires Rust 1.75+ and Clang 17+ on PATH):

```sh
# Clone and compile in release mode
git clone https://github.com/moon9t/IRIS.git
cd IRIS
cargo install --path .

# Verify the compiler installation
iris --version
```

### Running Your First Program

Write a simple program to `hello.iris`:

```iris
def main() -> i64 {
    print("Hello, IRIS!");
    0
}
```

Execute it instantly using the LLVM JIT engine, or build a native binary:

```sh
# Run via JIT
iris run hello.iris

# Compile to a native executable
iris build hello.iris
./hello
```

---

## Feature Matrix

| Category | Supported Capabilities |
| :--- | :--- |
| **Type System** | `i32`, `i64`, `f32`, `f64`, `bool`, `str`, fixed arrays, lists, maps, tuples, records, enums, generics, and traits |
| **Numeric & Math** | Extended integer types (`u8`, `i8`, `u32`, `u64`, `usize`), IEEE-754 float precision, math constants, and stochastic paths |
| **ML & Tensors** | First-class `tensor<DType, Shape>` with symbolic dims, dense/sparse representations, tape-based reverse-mode autodiff, and einsum |
| **AIS Framework** | Agent execution loop, multi-channel perception pipelines, decision policies, and reinforcement learning primitives |
| **Concurrency** | Actor-style `spawn`, parallel `par for` range loops, `async/await`, atomic scalar primitives, mutexes, and channels |
| **Error Handling** | Monadic `option<T>` and `result<T, E>` types, propagating `?` operator, and exhaustive match pattern checking |
| **Sandboxing** | Fine-grained global security policies, path traversal defense, FFI capability checks, and real-time audit logging |
| **FFI Integrations** | Dynamic C FFI (`dlopen`/`dlsym`), embedded Python evaluation engine, and native Rust cdylib adapters |
| **Tooling & IDE** | Fully-featured LSP server, DAP debugger protocol, verbose execution profiler, package manager, and a REPL |

---

## Language Specifications

### Type Declarations & Control Flow

IRIS supports strong, static type inference with options for explicit annotations.

```iris
record Point {
    x: f64,
    y: f64
}

choice Shape {
    Circle(f64),
    Rect(Point, Point),
    Unknown
}

def describe_shape(s: Shape) -> i64 {
    when s {
        Shape.Circle(r) => {
            print("Circle radius");
            0
        }
        Shape.Rect(p1, p2) => {
            print("Rectangle boundary");
            1
        }
        Shape.Unknown => -1,
    }
}
```

### Machine Learning, Tensors, & Autodiff

Create, manipulate, and differentiate mathematical models natively.

```iris
bring std.ml
bring std.tensor

def train_step(x: tensor<f32, [1, 4]>, y: f32, weights: tensor<f32, [4, 1]>) -> tensor<f32, [4, 1]> {
    // Autodiff dual numbers can track gradients through operations
    val target = grad(y);
    
    // Convert arrays or tensors to sparse matrices for memory efficiency
    val sparse_data = sparsify([0.0, 1.5, 0.0, 4.2]);
    val dense_back = densify(sparse_data);
    
    // Matrix contractions via Einstein Summation
    val pred = einsum("ik,kj->ij", x, weights);
    
    weights
}
```

### Parallel & Concurrent Programming

IRIS supports native channels, atomic operations, and parallel execution.

```iris
def parallel_computation(n: i64) -> [f64; 1000] {
    var output = [0.0; 1000];
    
    // Run loop iterations concurrently across available processor cores
    par for i in 0..1000 {
        output[i] = random() * 100.0;
    }
    
    output
}

def main() -> i64 {
    val ch = channel();
    
    spawn {
        val calculated = 42;
        send(ch, calculated);
    }
    
    val result = recv(ch);
    print("Received concurrent token");
    0
}
```

### capability-Based Sandboxing

Enforce strict operational constraints at runtime.

```iris
bring std.os
bring std.fs

def main() -> i64 {
    // Program can be run with `--sandbox` flag to restrict capabilities.
    // Filesystem traversal like "../../../etc/passwd" is caught by the compiler
    // and C-runtime path verification before any disk access.
    
    val file_content = fs.read_text("safe_local_data.txt");
    0
}
```

---

## Compiler Pass Pipeline

The IRIS compiler utilizes a block-parameter SSA (Static Single Assignment) intermediate representation resembling MLIR, completely eliminating phi nodes by passing arguments directly across CFG branches.

Before codegen, the `PassManager` drives **15 structural and optimization passes** to ensure correctness and maximize runtime efficiency:

1. **`HmTypeInferPass`** — Resolves type variables and implicit placeholders using Hindley-Milner union-find unification.
2. **`ValidatePass`** — Verifies SSA invariants, dominance criteria, and CFG structural integrity.
3. **`TypeInferPass`** — Confirms global type consistency and enforces static constraints.
4. **`ConstFoldPass`** — Performs aggressive constant folding (evaluating static math expressions at compile time) and algebraic simplifications.
5. **`StrengthReducePass`** — Replaces costly operations with cheaper equivalents (e.g., converting powers to multiplications, or division by constants to multiplications).
6. **`CopyPropPass`** — Dedupes redundant constants and propagates values transitively across registers to reduce stack pressure.
7. **`OpExpandPass`** — Expands high-level element-wise tensor operations into optimized execution loops.
8. **`LicmPass`** — Performs Loop-Invariant Code Motion, hoisting invariant expressions out of loop bodies into preheaders.
9. **`InlinePass`** — Inlines small, non-recursive functions to eliminate call frame overhead.
10. **`LoopUnrollPass`** — Unrolls small loops with constant limits (up to 8 iterations) to eliminate branch penalties.
11. **`ExhaustivePass`** — Validates pattern matching `when` expressions, ensuring enums and choices are handled exhaustively.
12. **`DcePass`** — Eliminates dead code, dead parameters, and unreachable execution blocks.
13. **`CsePass`** — Resolves and merges common subexpressions within the same dominance hierarchy.
14. **`ShapeCheckPass`** — Statically verifies that tensor shapes align across contractions and matrix operations.
15. **`GcAnnotatePass`** — Analyzes reference lifespans and inserts optimal `Retain`/`Release` calls to drive the reference-counting GC.

---

## Standard Library (35 Modules)

IRIS provides a production-ready standard library embedded directly into the compiler executable:

* **Computation:** `std.math` (GCD, LCM, trig, constants), `std.tensor` (matmul, contractions, slicing), `std.stochastic` (normal/Brownian distribution generators), `std.ml` (regression, normalizers, loss), `std.nn` (neural network layers, conv2d, LSTM), `std.rl` (Q-learning, SARSA, replay buffers).
* **Data Structures:** `std.iter` (HOF transformers), `std.set` (hash-sets), `std.queue` (FIFO), `std.heap` (min-heap/priority), `std.deque` (double-ended queue), `std.bitset` (compact bit arrays), `std.table` (in-memory relational operations), `std.dataframe` (structured analysis), `std.dataset` (batch loaders).
* **Systems & I/O:** `std.fs` (file read/write), `std.path` (filepath manipulation), `std.os` (environment, command execution), `std.time` (timestamps, performance gauges), `std.log` (structured runtime logging).
* **Serialization & Networking:** `std.json` (stringify, parse), `std.csv` (row emission, parsing), `std.http` (client engine), `std.http_server` (threaded HTTP server framework), `std.sql` (SQLite driver), `std.kv` (embedded SQLite key-value store).
* **Security & FFI:** `std.crypto` (SHA256, UUID, encoding), `std.ffi` (C dynamic linking, Python wrappers, Rust cdylib), `std.async` (asynchronous schedulers).
* **Graphics & Testing:** `std.svg` (vector graphics), `std.termplot` (terminal terminal graphing), `std.testing` (assertion suites).

---

## Compiler Tooling Ecosystem

IRIS includes modern development tools out of the box:

### Language Server Protocol (LSP)

The embedded LSP server powers syntax coloring, diagnostics, type tooltips on hover, and code actions (quick-fixes and best-practice tips):
```sh
iris lsp
```

### Debug Adapter Protocol (DAP)

A native DAP server allows standard debugger integration to set breakpoints, step through code, inspect variables, and watch active memory states:
```sh
iris dap
```

### Performance Profiler

Run any program with the verbose profiler to record function call counts, runtime allocations, execution durations, and generate interactive SVG flame graphs:
```sh
# Generate a folded stack file for speedscope or flamegraph.pl
iris profile --folded program.iris

# Emit a completed SVG Flame Graph directly
iris profile --svg output.svg program.iris
```

### Package Manager

Create new projects, handle remote dependencies, and compile libraries:
```sh
# Initialize a new workspace
iris pkg init my_agent

# Add a dependency
iris pkg add serde

# Build the project dependencies and code
iris pkg build
```

### Interactive REPL

Experiment with expressions and inspect types interactively with balanced multiline evaluation:
```sh
iris repl
```

*REPL Meta-commands:* Use `:help` to list commands, `:env` to dump active bindings, `:type <expr>` to see type signatures, and `:ir <expr>` to inspect generated SSA IR.

---

## Verification Pipeline

IRIS maintains a rigorous testing harness to ensure compiler correctness and prevent performance regressions:

* **Regression Tests:** 149 integration test suites (~1,100 specific assertions) in the `tests/` directory verifying everything from type unification to FFI boundaries.
* **Continuous Fuzzing:** Embedded fuzzing targets in `fuzz/` test the robustness of the lexer, parser, and IR lowerer under anomalous inputs.
* **Performance Regressions:** Benchmarks in `benches/` continuously profile tree traversals, matrix multiplications, database indexing, and mathematical integration algorithms.

---

## License

This project is licensed under the **GNU General Public License v2.0 (or later)**. See the [LICENSE](LICENSE) file for the complete text.

Copyright (C) 2024-2026 Moon & IRIS Project Contributors.
