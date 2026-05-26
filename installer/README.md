# IRIS Language — Installer & Packaging

Cross-platform installers, package builders, and one-line install scripts for the
IRIS programming language.

## Quick Install

### Linux

```bash
curl -fsSL https://raw.githubusercontent.com/moon9t/iris/master/installer/linux/install.sh | bash
```

### macOS

```bash
curl -fsSL https://raw.githubusercontent.com/moon9t/iris/master/installer/macos/install.sh | bash
```

### Windows

Download `IRIS-0.5.0-windows-x64-setup.exe` from the
[latest release](https://github.com/moon9t/iris/releases/latest), or use
PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File installer\install.ps1
```

---

## Directory Layout

```
installer/
├── build-all.sh              # Cross-platform build orchestrator
├── install.ps1               # Windows PowerShell installer
├── uninstall.ps1             # Windows PowerShell uninstaller
├── build_installer.ps1       # Windows Inno Setup .exe builder
├── iris.iss                  # Inno Setup 6 script
├── README.md                 # This file
├── linux/
│   ├── install.sh            # One-line Linux installer
│   ├── uninstall.sh          # Linux uninstaller
│   ├── build-deb.sh          # Debian .deb package builder
│   ├── build-rpm.sh          # RPM package builder
│   └── build-appimage.sh     # AppImage builder
├── macos/
│   ├── install.sh            # One-line macOS installer
│   ├── uninstall.sh          # macOS uninstaller
│   ├── build-pkg.sh          # macOS .pkg installer builder
│   └── build-dmg.sh          # macOS .dmg disk image builder
└── windows/
    ├── build_portable.ps1    # Portable .zip builder
    └── build_msi.ps1         # WiX .msi builder
```

---

## Platform Details

### Linux

**One-line install** — downloads the latest release, installs to `~/.iris/bin`,
configures PATH in `.bashrc`/`.zshrc`/`.profile`:

```bash
curl -fsSL https://raw.githubusercontent.com/moon9t/iris/master/installer/linux/install.sh | bash
```

**Debian/Ubuntu (.deb):**

```bash
sudo dpkg -i iris_0.5.0_x86_64.deb
# or
sudo apt install ./iris_0.5.0_x86_64.deb
```

**Fedora/RHEL (.rpm):**

```bash
sudo rpm -i iris-0.5.0-1.x86_64.rpm
# or
sudo dnf install ./iris-0.5.0-1.x86_64.rpm
```

**AppImage (portable, no root):**

```bash
chmod +x IRIS-0.5.0-x86_64.AppImage
./IRIS-0.5.0-x86_64.AppImage --version
```

**Uninstall:**

```bash
# Shell install:
bash installer/linux/uninstall.sh

# System package:
sudo dpkg -r iris          # Debian
sudo rpm -e iris            # RPM
```

### macOS

**One-line install** — downloads the latest release, installs to `~/.iris/bin`,
configures PATH in `.zshrc`/`.bash_profile`:

```bash
curl -fsSL https://raw.githubusercontent.com/moon9t/iris/master/installer/macos/install.sh | bash
```

**macOS Installer Package (.pkg):**

Double-click the `.pkg` file, or from the terminal:

```bash
sudo installer -pkg IRIS-0.5.0-macos-aarch64.pkg -target /
```

Installs to `/usr/local/bin/iris` with stdlib at `/usr/local/share/iris/`.

**Disk Image (.dmg):**

Mount the DMG image and run the included `Install` script.

**Uninstall:**

```bash
# Shell install:
bash installer/macos/uninstall.sh

# .pkg install:
sudo rm -rf /usr/local/bin/iris /usr/local/share/iris
sudo pkgutil --forget dev.moon9t.iris
```

### Windows

**Inno Setup installer (.exe)** — the recommended method:

1. Download `IRIS-0.5.0-windows-x64-setup.exe` from the latest release
2. Run the installer (no admin required — per-user install)
3. Choose **Full** for compiler + LLVM + MinGW, or **Compact** for compiler only

**PowerShell installer:**

```powershell
powershell -ExecutionPolicy Bypass -File installer\install.ps1
```

**Portable .zip** — extract and add to PATH:

```powershell
Expand-Archive IRIS-0.5.0-windows-x86_64-portable.zip -DestinationPath C:\iris
$env:PATH += ";C:\iris"
```

**Uninstall:**

- Via **Add or Remove Programs** (Inno Setup install)
- Or: `powershell -ExecutionPolicy Bypass -File installer\uninstall.ps1`

---

## What Gets Installed

| Component | Linux / macOS | Windows |
|-----------|--------------|---------|
| `iris` binary | `~/.iris/bin/iris` | `%LOCALAPPDATA%\Programs\IRIS\iris.exe` |
| Standard library | `~/.iris/stdlib/` | `...\IRIS\stdlib\` |
| Examples | `~/.iris/examples/` | `...\IRIS\examples\` |
| PATH config | `.bashrc` / `.zshrc` | User PATH environment variable |
| VSCode extension | Auto-installed if `code` detected | Auto-installed if VSCode detected |

System packages (.deb / .rpm / .pkg) install to `/usr/bin/iris` and
`/usr/share/iris/` instead.

---

## Building Installers

### Build all formats for the current platform

```bash
bash installer/build-all.sh
```

Options:
- `--version 0.5.0` — set the version (default: 0.5.0)
- `--skip-build` — use an existing `target/release/iris` binary
- `--format deb` — build only a specific format (deb, rpm, appimage, pkg, dmg, portable)

### Build specific formats

**Linux .deb:**
```bash
bash installer/linux/build-deb.sh --version 0.5.0 --arch x86_64
```

**Linux .rpm:**
```bash
bash installer/linux/build-rpm.sh --version 0.5.0 --arch x86_64
```

**Linux AppImage:**
```bash
bash installer/linux/build-appimage.sh --version 0.5.0
```

**macOS .pkg:**
```bash
bash installer/macos/build-pkg.sh --version 0.5.0
```

**macOS .dmg:**
```bash
bash installer/macos/build-dmg.sh --version 0.5.0
```

**Windows Inno Setup .exe:**
```powershell
powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1
```

**Windows portable .zip:**
```powershell
powershell -ExecutionPolicy Bypass -File installer\windows\build_portable.ps1
```

**Windows .msi (requires WiX Toolset):**
```powershell
powershell -ExecutionPolicy Bypass -File installer\windows\build_msi.ps1
```

All outputs go to `installer/dist/`.

---

## CI/CD

The [release workflow](../.github/workflows/release.yml) automatically builds
all installer formats when a version tag (`v*`) is pushed:

```bash
git tag v0.5.0
git push origin v0.5.0
```

The GitHub Release will include:
- Binary archives (`.tar.gz` / `.zip`) for all 6 targets
- `.deb` packages (x64 + ARM64)
- `.rpm` packages (x64 + ARM64)
- `.pkg` installers (x64 + ARM64)
- `.dmg` disk images (x64 + ARM64)
- Portable `.zip` for Windows (x64 + ARM64)
- VS Code extension (`.vsix`)
- `SHA256SUMS.txt` for integrity verification

---

## Troubleshooting

**`iris: command not found`** — restart your terminal after install, or source
your profile:

```bash
source ~/.bashrc   # Linux
source ~/.zshrc    # macOS
```

On Windows: restart your terminal or run
`$env:PATH += ";$env:LOCALAPPDATA\Programs\IRIS"`

**`clang not found`** — `iris run`, `iris build`, `--emit eval`, and `--emit jit`
use the LLVM/native pipeline. Install clang and the target sysroot/linker:

```bash
sudo apt install clang lld         # Debian/Ubuntu
sudo dnf install clang lld         # Fedora
brew install llvm                  # macOS
# Windows: use the Full installer, which bundles LLVM and MinGW UCRT64
```

See [../docs/current-language-lock.md](../docs/current-language-lock.md#binary-install-dependencies)
for the full binary dependency matrix and bundling policy.

**macOS Gatekeeper blocks the binary** — remove the quarantine attribute:

```bash
xattr -d com.apple.quarantine ~/.iris/bin/iris
```

**LSP not connecting in VSCode** — set `iris.executablePath` in VSCode settings
to the full path of the `iris` binary.
