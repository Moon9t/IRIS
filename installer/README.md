# IRIS Language — Windows Installer

## Prerequisites

- Windows 10 or later (x64)
- **No internet connection required** — all toolchains are bundled
- **No manual dependency installation required** — everything is included

## Installation

1. **Download** `IRIS-0.2.0-windows-x64-setup.exe`
2. **Run the installer** — no administrator privileges required (installs per-user by default)
3. Choose **Full** (recommended) to install everything, or **Compact** for the compiler only
4. **Restart your terminal** (or VSCode) to pick up the updated PATH
5. **Verify:**

   ```bash
   iris --version
   iris repl
   ```

## What gets installed

| Component | Install Location | Description |
| --- | --- | --- |
| `iris.exe` | `%LOCALAPPDATA%\Programs\IRIS` | Compiler, REPL, LSP server, DAP server |
| LLVM (clang + lld) | `...\IRIS\toolchain\llvm\bin` | For native binary compilation |
| MinGW sysroot | `...\IRIS\toolchain\ucrt64` | C headers + static libs for linking |
| VSCode extension | VSCode extensions dir | Syntax highlighting, diagnostics, debugging |
| User PATH | `%LOCALAPPDATA%\Programs\IRIS` | Added automatically (optional) |

## Installer Options

The setup wizard offers three installation types:

- **Full** — Compiler + LLVM/clang + MinGW sysroot + VSCode extension (~50 MB compressed, ~300 MB on disk)
- **Compact** — Compiler only (requires clang + MinGW already installed on the system)
- **Custom** — Choose individual components

Optional tasks (all enabled by default):

- Add IRIS to user PATH
- Add LLVM to user PATH
- Install VSCode extension (if VSCode is detected)

## VSCode Integration

After installation, open any `.iris` file in VSCode. The extension provides:

- Syntax highlighting
- Error diagnostics as you type
- Hover for type info, completions, goto-definition
- Inline **Run** / **Debug** buttons above zero-argument functions
- `Ctrl+F5` to run the current file
- `IRIS: Open REPL` command in the Command Palette
- `IRIS: Show IR Output` — inspect compiled IR
- Format on save

## Uninstallation

Use **Add or Remove Programs** in Windows Settings, or run the uninstaller from the Start Menu group.

The uninstaller removes PATH entries and registry keys automatically.

## Quick Start

```bash
# Interpreter mode (fast, no C compiler needed)
iris run hello.iris

# Native binary (requires clang + lld, bundled in Full install)
iris build hello.iris -o hello.exe
./hello.exe

# Interactive REPL
iris repl

# Inspect compiler output
iris --emit ir hello.iris
iris --emit llvm hello.iris
```

## Example: hello.iris

```iris
def main() -> i64 {
    print("Hello from IRIS!");
    0
}
```

## Building the Installer

Requires [Inno Setup 6](https://jrsoftware.org/isdl.php) and MSYS2/ucrt64.

```powershell
# From the project root:
powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1

# Skip the cargo build step:
powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1 -SkipBuild
```

Output: `installer\dist\IRIS-0.2.0-windows-x64-setup.exe`

## Troubleshooting

**`iris: command not found`** — restart your terminal after install, or run:

```powershell
$env:PATH += ";$env:LOCALAPPDATA\Programs\IRIS"
```

**`'clang' not found`** — make sure you chose "Full" installation. If you picked "Compact", install LLVM manually from <https://releases.llvm.org/>

**LSP not connecting in VSCode** — set `iris.executablePath` in VSCode settings to the full path of `iris.exe`
