# IRIS 0.4.0 Current Language Lock

This document captures the current ready-to-use IRIS language surface for the 0.4.0 build. It is the reference for examples, LSP/DAP support, editor grammar, installer dependency checks, and ML pipeline examples.

## Locked Language Surface

- Entry point: `def main() -> i64`.
- Bindings: `val` and `var` are both inferred bindings when no explicit type annotation is written. `val` is inferred immutable; `var` is inferred mutable.
- Control flow: `if/else`, `while`, `loop`, `break`, `continue`, `for i in a..b`, `for item in list`, `par for`, `return`, and expression tails.
- Pattern matching: `when` over `choice` enums, `option<T>`, `result<T,E>`, literals, tuples, ranges, and `_`.
- Declarations: `def`, `pub def`, `async def`, `record`, `choice`, `const`, `type`, `trait`, `impl`, `extern def`, `bring`, and `model`.
- Types: `i64`, `i32`, `i8`, `u8`, `u32`, `u64`, `usize`, `f64`, `f32`, `bool`, `str`, `tensor<dtype, [dims]>`, `[T; N]`, tuples, `list<T>`, `map<K,V>`, `option<T>`, `result<T,E>`, `chan<T>`, `atomic<T>`, `mutex<T>`, `grad<T>`, `sparse<T>`, named records/enums, and function types.
- Expressions: calls, method calls, field access, indexing, tuple indexing, arrays, casts with `to`, lambdas, `await`, and `?`.
- Model DSL: `model Name { input x: T layer y Op(...) output y }` with contextual `input`, `layer`, and `output`.
- Standard library: embedded `std.math`, `std.string`, `std.fmt`, `std.fs`, `std.json`, `std.csv`, `std.http`, `std.kv`, `std.table`, `std.dataset`, `std.dataframe`, `std.iter`, `std.deque`, `std.bitset`, `std.crypto`, `std.os`, `std.ffi`, `std.async`, `std.testing`, `std.log`, `std.ml`, `std.nn`, `std.tensor`, and `std.http_server`.

## Tooling Contract

- LSP: diagnostics, hover, rich completions, go to definition, document symbols, signature help, formatting, quick fixes, inlay hints, references, and rename.
- DAP: launch, line/conditional/log/hit-count breakpoints, continue, next, step in, step out, step back, pause, restart, loaded sources, stack trace, locals, set variable, watch/evaluate, exception info, and debug-console completions.
- VS Code extension: grammar, snippets, run/build/debug commands, inline run/debug code lenses, REPL, IR/LLVM viewers, LSP status bar, and settings for executable path, formatting, inlay hints, timing, and stop-on-entry debugging.
- ML pipeline: `examples/ml_full_pipeline.iris` demonstrates mixed-source ingestion through adapters, cleaning, supervised dataset construction, training, scoring, streaming re-ingest, retraining, prediction, and backend tensor handoff.

## Binary Install Dependencies

The `iris` binary itself includes the Rust compiler frontend, embedded runtime C source, and embedded IRIS stdlib sources. Running source files through user-facing commands uses the LLVM/native pipeline, so execution dependencies are:

| Use case | Required on target device |
| --- | --- |
| `iris --version`, parsing, diagnostics, LSP, DAP startup, REPL startup | `iris` binary and the OS runtime needed by that binary |
| `iris run`, `iris build`, `--emit eval`, `--emit jit`, `--emit binary` | `iris` plus `clang`/LLVM tools and a linkable target sysroot |
| Windows native execution | LLVM `clang`/`lld` plus a MinGW UCRT64 sysroot with headers, CRT objects, import libs, and GCC runtime libs |
| Linux native execution | LLVM/clang/lld plus compatible libc development files and system libs such as `libm` and pthreads |
| macOS native execution | Apple Command Line Tools or a bundled LLVM that can see the macOS SDK |
| Python FFI | A compatible Python installation available at runtime |
| C/Rust FFI | The user-provided shared libraries (`.dll`, `.so`, `.dylib`) and their transitive dependencies |
| Native ONNX/TensorFlow backend handoff | Backend SDK env vars plus `IRIS_NATIVE_ML_BACKENDS=1` at build/run time |
| VS Code extension | VS Code and the `iris` binary on PATH or configured through `iris.executablePath` |

## Bundling Policy

Installers can bundle dependencies. The current installer layout already supports a `toolchain/` payload:

- Windows full installers should bundle `toolchain/llvm` and `toolchain/ucrt64` so `iris run` works on a clean machine.
- Portable Windows zips can include the same `toolchain/` folder next to `iris.exe`.
- Linux and macOS packages may bundle LLVM, but should still detect or install the host sysroot/SDK through the package manager or Xcode Command Line Tools because those are platform-specific and large.
- Compact installers may install only `iris` and then print the exact missing dependency command for native execution.
