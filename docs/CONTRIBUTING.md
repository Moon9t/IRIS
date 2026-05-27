Contributing to IRIS — notes for Linguist submissions
=====================================================

If you're updating the TextMate grammar, language metadata, or preparing another submission to GitHub Linguist, follow these steps to keep the process reproducible.

Local preparation (recommended via Docker)
1. Update the grammar source in this repository: `iris-tmLanguage/syntaxes/iris.tmLanguage.json`.
2. Regenerate the unified diff for Linguist:

```sh
python tools/generate_iris_linguist_patch.py
```

3. Prepare a local checkout of `github-linguist/linguist` (fork if needed).
4. Apply the generated patch to your local Linguist checkout (located in `tools/iris-linguist.patch`) and add the vendored grammar under `vendor/grammars/iris-tmLanguage`.

Using Docker to run Linguist helpers (recommended)
1. Build the helper image and run a container that mounts your local Linguist checkout (example helper included under `tools/linguist-docker/`).
2. Inside the container, run:

```sh
bundle install
script/update-ids
# optionally: rake test
```

Notes
- `script/update-ids` generates the `language_id` used by `languages.yml`; run it after adding your language block.
- Forked PRs to Linguist will not run Actions until a repository maintainer approves and triggers them; be prepared to ask maintainers to "Approve and run" Actions for your PR.
- Keep vendor licenses and README files with the grammar when vendoring.

If you want me to run the Docker helper locally or create the patch for you, say so and I will proceed.
# Contributing to IRIS

Thank you for your interest in contributing to IRIS! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Project Structure](#project-structure)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/iris.git
   cd iris
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/moon9t/iris.git
   ```

## Development Setup

### Prerequisites

- **Rust** 1.75+ (install via [rustup](https://rustup.rs/))
- **LLVM 18** (for native compilation features)
- **Git**
- **Node.js 18+** and **npm** (only for VS Code extension development)

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run the REPL
cargo run -- repl

# Run a file
cargo run -- run examples/hello.iris
```

### Running Tests

```bash
# Run all tests
cargo test

# Run a specific test
cargo test phase42

# Run tests with output
cargo test -- --nocapture
```

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/my-bugfix
   ```

2. **Make your changes** — keep commits focused and atomic.

3. **Write tests** for new functionality. Every new feature or bugfix should include tests.

4. **Ensure all tests pass**:
   ```bash
   cargo test
   ```

5. **Format and lint your code**:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   ```

## Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your branch** to your fork:
   ```bash
   git push origin feature/my-feature
   ```

3. **Open a Pull Request** against `main` on the upstream repository.

4. **Fill in the PR template** — describe what changed and why.

5. **Wait for review** — a maintainer will review your PR. Address any feedback promptly.

6. **CI must pass** — all tests and lints must succeed before merging.

## Coding Standards

### Rust Code

- Follow standard Rust conventions (`rustfmt` defaults)
- Use descriptive variable and function names
- Document public APIs with `///` doc comments
- Avoid `unwrap()` in library code — use proper error handling
- Keep functions focused and reasonably short

### IRIS Code (stdlib, examples)

- Use `snake_case` for functions and variables
- Use `PascalCase` for types (`record`, `choice`)
- Add comments explaining non-obvious logic
- Keep examples self-contained and runnable

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add pattern matching for tuples
fix: correct string escape handling in parser
docs: update Chapter 5 with new examples
test: add tests for closure captures
refactor: simplify IR lowering pass
```

## Testing

### Test Organization

- **Unit tests**: In each source file via `#[cfg(test)]` modules
- **Integration tests**: In the `tests/` directory (`phase*.rs` files)
- **Example programs**: In `examples/` — should all run without errors

### Writing Tests

Integration tests follow this pattern:

```rust
#[test]
fn my_feature_works() {
    let src = r#"
def main() -> i64 {
    // Your test code here
    val result = my_feature(42);
    print(result);
    0
}
"#;
    let (exit, stdout, _stderr) = iris::run_str(src);
    assert_eq!(exit, 0);
    assert_eq!(stdout.trim(), "expected output");
}
```

## Project Structure

```
src/
├── main.rs          # CLI entry point
├── lib.rs           # Library root
├── cli.rs           # Command-line interface
├── compiler.rs      # Native compilation (LLVM)
├── dap.rs           # Debug Adapter Protocol server
├── debugger.rs      # Debugger core
├── diagnostics.rs   # Error rendering
├── error.rs         # Error types
├── lsp.rs           # Language Server Protocol
├── pkg.rs           # Package manager
├── repl.rs          # Interactive REPL
├── codegen/         # LLVM code generation
├── interp/          # IR interpreter
├── ir/              # Intermediate representation
├── lower/           # AST → IR lowering
├── parser/          # Lexer + parser
├── pass/            # Optimization passes
├── runtime/         # C runtime for native binaries
└── stdlib/          # Built-in standard library modules

stdlib/              # IRIS standard library (.iris files)
tests/               # Integration tests
examples/            # Example programs
vscode-iris/         # VS Code extension
```

## Areas Where Help is Welcome

- **Bug reports** — found a crash or incorrect behavior? File an issue!
- **Documentation** — typos, unclear explanations, missing examples
- **Standard library** — new utility functions, improving existing ones
- **Examples** — interesting programs that showcase IRIS features
- **Error messages** — making compiler errors clearer and more helpful
- **Performance** — optimizer passes, interpreter speed improvements
- **Platform support** — testing on Linux, macOS, ARM

## Questions?

Open a [Discussion](https://github.com/moon9t/iris/discussions) on GitHub or file an issue if you're unsure about something. We're happy to help!
