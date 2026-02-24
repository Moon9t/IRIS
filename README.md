# IRIS

IRIS is a Rust-based project for intermediate representation and compilation research. It includes modules for parsing, lowering, code generation, interpretation, and various optimization passes. The project is structured for extensibility and experimentation with compiler techniques.

## Features
- Modular IR design
- ONNX and LLVM codegen support
- Graph-based optimizations
- Shape/type inference and validation
- CLI for compilation and diagnostics

## Getting Started
1. **Build:**
   ```sh
   cargo build
   ```
2. **Run:**
   ```sh
   cargo run -- [options]
   ```
3. **Test:**
   ```sh
   cargo test
   ```

## Directory Structure
- `src/` - Main source code
- `tests/` - Test cases
- `Cargo.toml` - Project manifest

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
