# IRIS Language — Windows Installer

## Prerequisites

- Windows 10 or later (x64)
- **For `iris run`/`iris repl`**: no additional dependencies
- **For `iris build` (native binaries)**: requires LLVM/clang + MSYS2
  - LLVM: <https://releases.llvm.org/> (install to `C:\Program Files\LLVM`)
  - MSYS2: <https://www.msys2.org/> (install to `C:\msys64`)

## Installation

1. **Download** `IRIS-0.1.0-windows-x64.zip` and unzip it
2. **Open PowerShell** as your regular user (not administrator required)
3. **Run the installer:**

   ```powershell
   powershell -ExecutionPolicy Bypass -File install.ps1
   ```

4. **Restart your terminal** (or VSCode) to pick up the updated PATH
5. **Verify:**

   ```bash
   iris --version
   iris repl
   ```

## What gets installed

| Location | Contents |
| --- | --- |
| `%USERPROFILE%\.iris\bin\iris.exe` | The IRIS compiler and runtime |
| Windows user PATH | `%USERPROFILE%\.iris\bin` added permanently |
| VSCode extension | `iris-lang` extension (if VSCode is detected) |

## VSCode Integration

After installation, open any `.iris` file in VSCode. The extension provides:

- Syntax highlighting
- Error diagnostics as you type
- Hover for type info, completions, goto-definition
- Inline **▷ Run** / **⬡ Debug** buttons above zero-argument functions
- `Ctrl+F5` to run the current file
- `IRIS: Open REPL` command in the Command Palette
- `IRIS: Show IR Output` — inspect compiled IR
- Format on save

## Uninstallation

```powershell
powershell -ExecutionPolicy Bypass -File uninstall.ps1
```

## Quick Start

```bash
# Interpreter mode (fast, no C compiler needed)
iris run hello.iris

# Native binary (requires clang + MSYS2)
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

## Troubleshooting

**`iris: command not found`** — restart your terminal after install, or run:

```powershell
$env:PATH += ";$env:USERPROFILE\.iris\bin"
```

**`'clang' not found`** — install LLVM from <https://releases.llvm.org/> to `C:\Program Files\LLVM`

**LSP not connecting in VSCode** — set `iris.executablePath` in VSCode settings to the full path of `iris.exe`
