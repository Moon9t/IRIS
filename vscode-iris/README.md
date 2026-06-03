# IRIS Language for Visual Studio Code

Full-featured IDE support for the [IRIS programming language](https://github.com/moon9t/iris) — syntax highlighting, Language Server Protocol, debugging, REPL, and more.

![VS Code](https://img.shields.io/badge/VS%20Code-1.85+-blue?logo=visualstudiocode)
![License](https://img.shields.io/badge/license-GPL--2.0--or--later-green)
![Version](https://img.shields.io/badge/version-1.0.0-rc1-orange)

---

## Features

### Syntax Highlighting

Rich TextMate grammar for `.iris` files — keywords, types, strings, f-strings, comments, operators, and builtins are all highlighted accurately.

### Language Server Protocol (LSP)

Powered by the IRIS compiler's built-in language server:

- **Hover** — type info and doc comments on any symbol
- **Completions** — context-aware suggestions for functions, types, builtins, and keywords
- **Diagnostics** — real-time error and warning reporting, plus best-practice hints (long functions, missing doc comments, naming conventions)
- **Go to Definition** / **Peek Definition**
- **Document Symbols** — outline view and breadcrumbs
- **Signature Help** — parameter hints as you type
- **Formatting** — format on save (configurable)
- **Inlay Hints** — inline inferred binding hints on `val` / `var`
- **Code Actions** — auto-fix missing semicolons, type casts, naming conventions, and more

### Debug Adapter Protocol (DAP)

Step-through debugging with the built-in IRIS debugger:

- Breakpoints (line, conditional, logpoint, and hit-count)
- Step In / Step Over / Step Out / Step Back / Continue / Restart / Pause
- Variables inspector (locals)
- Watch expressions
- Debug Console evaluation and completions

### Commands

| Command | Keybinding | Description |
|---------|------------|-------------|
| **IRIS: Run File** | `Ctrl+F5` | Run the current `.iris` file |
| **IRIS: Build Binary** | — | Compile to a native executable |
| **IRIS: Debug File** | `F5` | Launch the current `.iris` file under the DAP debugger |
| **IRIS: Open REPL** | — | Launch an interactive IRIS session |
| **IRIS: Restart Language Server** | — | Restart the LSP server |
| **IRIS: Stop Language Server** | — | Stop the LSP server |
| **IRIS: Show IR Output** | — | Display the compiler's SSA IR |
| **IRIS: Show LLVM IR Output** | — | Display the generated LLVM IR |
| **IRIS: Show Version Info** | — | Show compiler version, git commit, build date, and target |

### Snippets

Quickly scaffold common patterns: `def`, `record`, `choice`, `val`, `var`, `if`, `while`, `for`, `when`, `bring`, error handling, FFI calls, and more.

### Code Lens & Status Bar

- **Status bar** shows the active IRIS compiler version with a rich tooltip (git commit, branch, build date, target triple, rustc version).

---

## Requirements

- **VS Code** ≥ 1.85
- **IRIS compiler** installed and available on `PATH` (or configure `iris.executablePath`)

Install IRIS from [github.com/moon9t/iris/releases](https://github.com/moon9t/iris/releases) or build from source:

```bash
git clone https://github.com/moon9t/iris.git
cd iris
cargo build --release
```

---

## Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `iris.executablePath` | `"iris"` | Path to the `iris` binary |
| `iris.maxNumberOfProblems` | `100` | Maximum diagnostics shown in the Problems panel |
| `iris.formatOnSave` | `true` | Auto-format `.iris` files on save |
| `iris.trace.server` | `"off"` | Trace LSP communication (`off` / `messages` / `verbose`) |
| `iris.inlayHints.enabled` | `true` | Enable inlay hints |
| `iris.inlayHints.typeHints` | `true` | Show inferred binding hints on `val` / `var` bindings |
| `iris.debug.stopOnEntry` | `false` | Stop at the first executable trace point when debugging |
| `iris.showTimingOnRun` | `true` | Show compile + run elapsed time |

---

## Quick Start

1. Install the IRIS compiler and ensure `iris` is on your PATH.
2. Install this extension from the VS Code Marketplace.
3. Open any `.iris` file — the language server starts automatically.
4. Press `Ctrl+F5` to run, or use `F5` to debug.

```iris
// hello.iris
def main() -> i64 {
    print("Hello, IRIS!");
    0
}
```

---

## IRIS Language at a Glance

```iris
// Records, generics, and pattern matching
record Point { x: f64, y: f64 }

choice Shape {
    Circle(f64),
    Rect(f64, f64),
}

def area(s: Shape) -> f64 {
    when s {
        Shape.Circle(r) => 3.14159 * r * r,
        Shape.Rect(w, h) => w * h,
    }
}

def main() -> i64 {
    val c = Shape.Circle(5.0);
    val r = Shape.Rect(3.0, 4.0);
    print(concat("Circle area: ", to_str(area(c))));
    print(concat("Rect area: ", to_str(area(r))));
    0
}
```

IRIS features strong static typing, algebraic data types, closures, generics, pattern matching, multi-module projects, an LLVM-backed native compiler, and a growing standard library.

---

## Known Issues

- Native concurrency (`spawn` / `channel`) is under active development.
- `iris run` uses the LLVM/native pipeline, so a working clang/sysroot setup is required.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](https://github.com/moon9t/iris/blob/main/CONTRIBUTING.md).

---

## License

[GPL-2.0-or-later](https://github.com/moon9t/iris/blob/main/LICENSE)
