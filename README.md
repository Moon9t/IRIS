# IRIS — Intermediate Representation for Intelligent Systems

<p align="center">
  <strong>A compiled, statically-typed systems &amp; ML language written in Rust.</strong><br/>
  Low-level control. High-level ML ergonomics. First-class tensor, gradient, and sparsity types.
</p>

<p align="center">
  <a href="https://github.com/moon9t/iris/actions"><img src="https://github.com/moon9t/iris/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/moon9t/iris/releases"><img src="https://img.shields.io/github/v/release/moon9t/iris?label=release" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--2.0--or--later-blue.svg" alt="License"></a>
</p>

---

## Quick Start :

```sh
# Install (or download from Releases)
cargo install --path .

# Hello world
echo 'def main() -> i64 { print("Hello, IRIS!"); 0 }' > hello.iris
iris run hello.iris

# Build native binary
iris build hello.iris        # produces hello.exe (Windows) or ./hello
./hello

# Check compiler version
iris --version
```

`iris run`, `--emit eval`, and `--emit jit` execute through the LLVM/native
pipeline. There is no silent interpreter fallback in those user-facing paths.
`iris build --target <preset>` can now emit cross-target binaries when the
matching clang/sysroot toolchain is installed.

**Output of `iris --version`:**

```
iris 0.3.0 (abc1234 2026-03-13)
IRIS — Intermediate Representation for Intelligent Systems
Copyright (C) 2024-2026 Moon & IRIS Project Contributors
License: GPL-2.0-or-later <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>

Compiler:
  Version:       0.3.0
  Git commit:    abc1234567890abcdef1234567890abcdef123456
  Git branch:    main
  Build date:    2026-03-13

Platform:
  Target:        x86_64-pc-windows-msvc
  Host:          x86_64-pc-windows-msvc
  Thread model:  win32

Build:
  Profile:       release
  Opt level:     3
  Rust edition:  2021
  Built with:    rustc 1.78.0 (9b00956e5 2024-04-29)
```

---

## Features at a Glance

| Category | Highlights |
|----------|-----------|
| **Type System** | `i32` `i64` `f32` `f64` `bool` `str`, tensors, arrays, tuples, records, enums, generics, traits |
| **Collections** | `list<T>`, `map<K,V>`, deque, sorted set, bitset, heap, queue |
| **ML Built-ins** | `tensor<f32,[M,K]>`, `einsum`, `grad<T>` autodiff, `sparse<T>` |
| **Concurrency** | `channel<T>`, `spawn`, `par for`, `async/await`, `atomic<T>`, `mutex` |
| **Error Handling** | `option<T>`, `result<T,E>`, `?` operator, pattern matching (`when`) |
| **FFI** | C FFI (dlopen/dlsym), Python FFI (eval/exec/call), Rust cdylib FFI |
| **Native Compilation** | LLVM IR → clang → native binary (Windows, Linux, macOS) |
| **Package Manager** | `iris pkg init/add/remove/install/build/run` |
| **Tooling** | LSP server, DAP debugger, REPL, VS Code extension |
| **Standard Library** | 25 modules: math, string, fmt, fs, json, csv, http, crypto, ffi, os, testing, … |

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
val abs_x = if x < 0 { -x } else { x }

while count < 10 { count = count + 1 }

for i in 0..n { output[i] = relu(input[i]) }

par for i in 0..n { output[i] = input[i] * 2.0 }
```

### Records, Enums, and Pattern Matching

```iris
record Point { x: f64, y: f64 }

choice Shape { Circle, Square, Triangle }

def describe(s: Shape) -> i64 {
    when s {
        Shape.Circle   => 0,
        Shape.Square   => 1,
        Shape.Triangle => 2,
    }
}
```

### Closures and Generics

```iris
def apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

def double_it() -> i64 {
    val double = |x: i64| x * 2
    apply(double, 21)   // 42
}

def identity[T](x: T) -> T { x }
```

### Concurrency

```iris
def main() -> i64 {
    val ch = channel()
    spawn { send(ch, 42) }
    recv(ch)
}
```

### FFI (C, Python, Rust)

```iris
bring std.ffi

def main() -> i64 {
    // C FFI
    val lib = ffi_open("libm.so")
    val result = ffi_call_f64(lib, "sqrt", 144.0)

    // Python FFI
    val py_result = python_eval("2 ** 10")

    // Rust cdylib FFI
    val rlib = rust_lib_open("mylib.dll")
    val n = rust_call_i64(rlib, "compute", 42)
    0
}
```

### Options and Results

```iris
def safe_div(a: i64, b: i64) -> option<i64> {
    if b == 0 { none } else { some(a / b) }
}

// ? operator propagates errors
def parse_and_add(s: str) -> result<i64, str> {
    val n = parse_i64(s)?
    ok(n + 1)
}
```

### Autodiff and Sparse

```iris
def f(x: grad<f64>) -> grad<f64> { x * x }

def sparse_example(arr: [f64; 4]) -> [f64; 4] {
    val s = sparsify(arr)
    densify(s)
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

---

## Compiler Pipeline

```text
.iris source
    │
    ▼
  Lexer → Parser → AST → Lowerer → SSA IR → Pass Pipeline → Codegen
                                                                │
                                    ┌───────────────────────────┤
                                    ▼               ▼           ▼
                                 Native exec    LLVM IR      ONNX binary
                                                   │
                                                 clang
                                                   │
                                              Native binary
```

**IR design:** Block-parameter SSA (MLIR-style). No phi nodes — branch arguments carry values directly.

**Optimization passes:** Validate → TypeInfer → ConstFold → OpExpand → DCE → CSE → ShapeCheck

---

## Standard Library (25 modules)

```iris
bring std.math       // gcd, lcm, abs_i64, is_even, is_odd, ...
bring std.string     // pad_left, pad_right, words, lines, ...
bring std.fmt        // sprintf, pad_int, zero_pad_int, ...
bring std.fs         // read_text, write_text, path_exists, ...
bring std.json       // json_stringify, json_parse, ...
bring std.csv        // csv_parse_row, csv_emit_row, ...
bring std.http       // http_get, http_post, ...
bring std.time       // now, sleep, elapsed, ...
bring std.crypto     // sha256, uuid, hex_encode, hex_decode
bring std.ffi        // ffi_open, ffi_call_*, python_*, rust_*
bring std.os         // env_get, env_set, exec_cmd, pid, exit_code
bring std.testing    // assert_eq, assert_ne, assert_true, ...
bring std.log        // log_info, log_warn, log_error, ...
bring std.iter       // map_list, filter_list, reduce_list, ...
bring std.set        // set operations
bring std.queue      // FIFO queue
bring std.heap       // priority queue / min-heap
bring std.deque      // double-ended queue
bring std.kv         // key-value store (SQLite-backed)
bring std.table      // tabular data operations
bring std.dataset    // ML dataset abstraction
bring std.dataframe  // DataFrame-like API
bring std.path       // path manipulation
bring std.async      // async runtime helpers
bring std.bitset     // bit array operations
```

---

## Built-in Functions

**Math:** `sin`, `cos`, `tan`, `exp`, `log`, `log2`, `sqrt`, `abs`, `floor`, `ceil`, `round`, `sign`, `pow`, `min`, `max`, `clamp`, `math_pi`, `math_e`, `is_nan`, `is_inf`

**String:** `len`, `concat`, `contains`, `starts_with`, `ends_with`, `to_upper`, `to_lower`, `trim`, `repeat`, `to_str`, `format`, `split`, `join`, `find`, `slice`, `str_index`, `str_replace`, `str_reverse`, `char_at`, `str_pad_left`, `str_pad_right`, `str_chars`, `str_bytes`, `str_count`

**I/O:** `print`, `read_line`, `read_i64`, `read_f64`

**Collections:** `list`, `push`, `pop`, `list_get`, `list_set`, `list_len`, `list_map`, `list_filter`, `list_reduce`, `list_any`, `list_all`, `list_zip`, `list_enumerate`, `list_flatten`, `list_unique`, `list_reverse`, `list_sorted`, `list_sum`, `list_min`, `list_max`

**Map:** `map`, `map_get`, `map_set`, `map_contains`, `map_remove`, `map_keys`, `map_values`, `map_len`

**Parsing:** `parse_i64`, `parse_f64`, `json_stringify`, `regex_match`, `regex_find_all`, `regex_replace`

**System:** `cwd`, `list_dir`, `mkdir`, `remove_file`, `path_join`, `env_get`, `env_set`, `exec_cmd`, `pid`, `exit_code`, `type_of`

**Random:** `random`, `random_range`, `uuid`

**Crypto:** `sha256`, `hash`, `hex_encode`, `hex_decode`, `base64_encode`, `base64_decode`

**FFI:** `ffi_open`, `ffi_call`, `ffi_close`, `ffi_call_i64`, `ffi_call_f64`, `ffi_call_str`, `ffi_call_void`, `python_eval`, `python_exec`, `python_call`, `python_version`, `rust_lib_open`, `rust_call_i64`, `rust_call_f64`, `rust_call_void`

**Concurrency:** `channel`, `send`, `recv`, `spawn`, `chan_try_recv`, `chan_len`, `select`, `timeout`, `thread_count`, `atomic`, `atomic_load`, `atomic_store`, `atomic_add`

**DateTime:** `datetime_now`, `datetime_timestamp`, `datetime_format`

---

## CLI Usage

```sh
iris <file.iris>                    # Emit IR (default)
iris build <file.iris>              # Build native binary
iris run <file.iris>                # Build and run
iris repl                           # Interactive REPL
iris lsp                            # Start LSP server (stdin/stdout)
iris dap                            # Start DAP debugger (stdin/stdout)
iris pkg <cmd>                      # Package manager

# Flags
iris --emit ir|llvm|eval|binary|onnx|cuda|simd <file.iris>
iris build <file.iris> -o <output>
iris --version | -V
iris --help | -h
```

---

## Tooling

### VS Code Extension

The official `iris-lang` extension provides:

- **Syntax highlighting** — full TextMate grammar
- **LSP integration** — hover, completions, diagnostics, go-to-definition, rename, references, formatting, inlay hints, code actions (quick fixes + best practice hints)
- **DAP debugger** — breakpoints, step in/over/out, step back, variables, call stack, hover evaluation
- **Code lenses** — inline ▷ Run / ⬡ Debug buttons on zero-arg functions
- **REPL** — integrated terminal REPL
- **Status bar** — shows IRIS version, git commit, build info; click for server actions

Install: `code --install-extension iris-lang-0.2.0.vsix`

### REPL Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `:help` | `:h` | Show command reference |
| `:env` | `:e` | List active definitions and bindings |
| `:type <expr>` | `:t <expr>` | Show inferred type of expression |
| `:bring <mod>` | `:b <mod>` | Load a stdlib module |
| `:time` | | Show elapsed time of last evaluation |
| `:history` | | Show numbered input history |
| `:ir <expr>` | | Show compiled IR for an expression |
| `:clear` | | Clear the terminal screen |
| `:reset` | | Clear all session state |
| `:quit` | `:q` | Exit the REPL |

### Package Manager

```sh
iris pkg init myproject          # Create iris.toml + main.iris
iris pkg add serde               # Add dependency
iris pkg install                 # Fetch all dependencies
iris pkg build                   # Build the project
iris pkg run                     # Build and run
iris pkg list                    # List dependencies
```

---

## Project Structure

```text
src/
  main.rs          CLI entry point
  lib.rs           Library root (compile, compile_multi)
  cli.rs           Argument parsing
  error.rs         Error types
  lsp.rs           Language Server Protocol
  dap.rs           Debug Adapter Protocol
  repl.rs          Interactive REPL
  pkg.rs           Package manager
  compiler.rs      Native compilation pipeline
  diagnostics.rs   Rich error rendering
  parser/          Lexer, AST, recursive-descent parser
  ir/              SSA IR types, blocks, functions, modules
  lower/           AST → IR lowering
  pass/            Optimization passes
  interp/          Tree-walking interpreter
  codegen/         LLVM IR, ONNX, IR printer
  runtime/         C runtime (iris_runtime.c/.h)
  stdlib/          25 standard library modules
  proto/           Protobuf encoding (ONNX)
stdlib/            External stdlib (file.iris, io.iris)
examples/          29 example programs
tests/             110+ integration test suites
vscode-iris/       VS Code extension
installer/         Windows installer (Inno Setup)
```

---

## Building from Source

Requires Rust 1.75+ stable. For native binary compilation, clang 17+ must be in PATH.

```sh
cargo build                # Debug build
cargo build --release      # Release build (optimized)
cargo test                 # Run all 110+ test suites (~850 tests)
```

---

## Implementation Status

| Phase | Feature | Status |
| ----- | ------- | ------ |
| 1–10 | Core: lexer, parser, lowerer, types, control flow, tensors | ✅ |
| 11–20 | Interpreter, diagnostics, records, enums, vars, functions, tuples | ✅ |
| 21–30 | Strings, arrays, closures, options, results, channels, par, async, autodiff | ✅ |
| 31–48 | Sparse, builtins, constants, collections, generics, traits, modules, LLVM codegen | ✅ |
| 100 | Complete LLVM IR codegen + native binaries via clang | ✅ |
| 101 | Rich error diagnostics, LSP, DAP, REPL | ✅ |
| 102 | VS Code extension, Windows installer | ✅ |
| 103 | Compiler pipeline rewrite (clang-only), C runtime | ✅ |
| 104 | HTTP, JSON, Regex, DateTime, OS, Random, Hash, Base64 builtins | ✅ |
| 105 | 60+ builtins, 9 stdlib modules, package manager | ✅ |
| 106 | C/Python/Rust FFI, LSP code actions, verbose --version, git info | ✅ |

---

## License

This program is free software; you can redistribute it and/or modify it under the terms of the **GNU General Public License v2.0** (or later) as published by the Free Software Foundation.

See [LICENSE](LICENSE) for the full text.

Copyright (C) 2024-2026 Moon

---

<p align="center">
  <a href="https://github.com/moon9t/iris">github.com/moon9t/iris</a>
</p>
