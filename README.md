
# IRIS: Intermediate Representation Infrastructure in Rust

IRIS is a modular, extensible compiler infrastructure written in Rust. It provides a rich intermediate representation (IR) framework, a suite of compiler passes, and support for ONNX and LLVM code generation. IRIS is designed for research, experimentation, and building custom compilers or analysis tools.

## Key Features

- **Modular IR**: Flexible IR with support for blocks, functions, graphs, and values.
- **Graph-Based Optimizations**: Includes dead node elimination, constant folding, shape/type inference, and more.
- **ONNX & LLVM Codegen**: Convert IR to ONNX or LLVM for interoperability and native code generation.
- **Pass Pipeline**: Customizable pass manager for IR and graph transformations.
- **Interpreter**: Execute IR directly for rapid prototyping and testing.
- **Diagnostics**: Built-in error handling and diagnostic reporting.
- **CLI Tooling**: Command-line interface for compiling, running, and analyzing models.

## Project Structure

- `src/`
  - `cli.rs`: Command-line interface and entry point.
  - `codegen/`: ONNX and LLVM code generation backends.
  - `interp/`: IR interpreter.
  - `ir/`: Core IR data structures (blocks, functions, graphs, instructions, types, values).
  - `lower/`: Lowering from graph to IR.
  - `parser/`: Lexer, parser, and AST for source input.
  - `pass/`: Optimization and analysis passes (const fold, DCE, type/shape inference, validation, etc).
  - `proto/`: Protobuf/ONNX utilities.
  - `main.rs`, `lib.rs`: Main library and binary entry points.
- `tests/`: Comprehensive test suite for all phases and passes.
- `Cargo.toml`: Rust project manifest.

## Building and Running

1. **Build the project:**
   ```sh
   cargo build --release
   ```
2. **Run the CLI:**
   ```sh
   cargo run -- [OPTIONS] <input>
   ```
3. **Run tests:**
   ```sh
   cargo test
   ```

## Example Usage

Compile a model or IR file:
```sh
cargo run -- model.onnx
```

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open issues or pull requests.

## License

This project is licensed under the GNU General Public License v2.0 or (at your option) any later version. See [LICENSE](LICENSE) for details.
