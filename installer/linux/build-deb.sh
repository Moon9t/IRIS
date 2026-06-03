#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# build-deb.sh — Build a .deb package for IRIS Language
# ──────────────────────────────────────────────────────────────────────────
# Usage:
#   ./installer/linux/build-deb.sh [--version 1.0.0-rc1] [--arch amd64]
#
# Produces: installer/dist/iris_<version>_<arch>.deb
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="1.0.0-rc1"
ARCH="amd64"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --arch)    ARCH="$2";    shift 2 ;;
        *)         echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Map arch names
case "$ARCH" in
    x86_64|amd64)    DEB_ARCH="amd64" ;;
    aarch64|arm64)   DEB_ARCH="arm64" ;;
    *)               echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

echo "Building IRIS .deb package v${VERSION} (${DEB_ARCH})"

# ── Paths ─────────────────────────────────────────────────────────────────
PKG_NAME="iris"
PKG_DIR="$ROOT/installer/dist/deb-staging"
DIST_DIR="$ROOT/installer/dist"
DEB_FILE="${PKG_NAME}_${VERSION}_${DEB_ARCH}.deb"

# ── Clean previous build ─────────────────────────────────────────────────
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/share/iris/stdlib"
mkdir -p "$PKG_DIR/usr/share/iris/examples"
mkdir -p "$PKG_DIR/usr/share/iris/toolchain"
mkdir -p "$PKG_DIR/usr/share/doc/iris"
mkdir -p "$PKG_DIR/usr/share/man/man1"
mkdir -p "$DIST_DIR"

# ── Build release binary (if not already built) ──────────────────────────
IRIS_BIN="$ROOT/target/release/iris"
if [[ ! -f "$IRIS_BIN" ]]; then
    echo "[1/5] Building release binary..."
    (cd "$ROOT" && cargo build --release)
else
    echo "[1/5] Using existing release binary."
fi

# ── Stage files ───────────────────────────────────────────────────────────
echo "[2/5] Staging files..."

# Binary
cp "$IRIS_BIN" "$PKG_DIR/usr/bin/iris"
chmod 755 "$PKG_DIR/usr/bin/iris"
strip "$PKG_DIR/usr/bin/iris" 2>/dev/null || true

# Get installed size in KB
INSTALLED_SIZE=$(du -sk "$PKG_DIR" | awk '{print $1}')

# Stdlib
cp -r "$ROOT/stdlib/"* "$PKG_DIR/usr/share/iris/stdlib/" 2>/dev/null || true

# Examples
cp -r "$ROOT/examples/"* "$PKG_DIR/usr/share/iris/examples/" 2>/dev/null || true

# Bundled LLVM toolchain (enables `iris build` without system clang)
TOOLCHAIN_SRC="$ROOT/toolchain"
if [[ -d "$TOOLCHAIN_SRC" ]]; then
    cp -r "$TOOLCHAIN_SRC"/* "$PKG_DIR/usr/share/iris/toolchain/" 2>/dev/null || true
    echo "  Bundled LLVM toolchain"
fi

# Documentation
cp "$ROOT/LICENSE" "$PKG_DIR/usr/share/doc/iris/copyright"
cp "$ROOT/README.md" "$PKG_DIR/usr/share/doc/iris/"

# ── Generate man page ────────────────────────────────────────────────────
echo "[3/5] Generating man page..."
cat > "$PKG_DIR/usr/share/man/man1/iris.1" << 'MANEOF'
.TH IRIS 1 "2026" "iris 1.0.0-rc1" "IRIS Language"
.SH NAME
iris \- IRIS programming language compiler and toolchain
.SH SYNOPSIS
.B iris
[\fIOPTIONS\fR] [\fICOMMAND\fR] [\fIFILE\fR]
.SH DESCRIPTION
IRIS (Intermediate Representation for Intelligent Systems) is a
programming language designed for machine learning and systems
programming. The \fBiris\fR command provides a compiler, REPL,
REPL, language server (LSP), and debug adapter (DAP).
.SH COMMANDS
.TP
.B run \fIFILE\fR
Compile and run an IRIS source file through the LLVM/native pipeline.
.TP
.B build \fIFILE\fR \-o \fIOUTPUT\fR
Compile an IRIS source file to a native binary.
.TP
.B repl
Start an interactive REPL session.
.TP
.B lsp
Start the Language Server Protocol server.
.TP
.B dap
Start the Debug Adapter Protocol server.
.SH OPTIONS
.TP
.B \-\-version
Print version information and exit.
.TP
.B \-\-help
Print help information and exit.
.TP
.B \-\-emit \fITYPE\fR
Emit compiler output: ir, llvm, asm, jit, eval.
.SH EXAMPLES
.nf
iris run hello.iris
iris build hello.iris -o hello
iris repl
iris --emit ir hello.iris
.fi
.SH FILES
.TP
.I /usr/share/iris/stdlib/
Standard library modules.
.TP
.I /usr/share/iris/examples/
Example programs.
.SH AUTHOR
IRIS Language Project <https://github.com/moon9t/iris>
.SH SEE ALSO
.UR https://github.com/moon9t/iris
IRIS on GitHub
.UE
MANEOF

gzip -9 "$PKG_DIR/usr/share/man/man1/iris.1"

# ── DEBIAN control file ──────────────────────────────────────────────────
echo "[4/5] Creating package metadata..."

cat > "$PKG_DIR/DEBIAN/control" << EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: devel
Priority: optional
Architecture: ${DEB_ARCH}
Installed-Size: ${INSTALLED_SIZE}
Maintainer: IRIS Language Project <iris@moon9t.dev>
Homepage: https://github.com/moon9t/iris
Description: IRIS programming language compiler and toolchain
 IRIS (Intermediate Representation for Intelligent Systems) is a
 programming language designed for machine learning and systems
 programming. Includes compiler, REPL, LSP server,
 and DAP server.
Depends: libc6
EOF

# Post-install script
cat > "$PKG_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/sh
set -e
echo ""
echo "  IRIS Language installed successfully!"
echo "  Run 'iris --version' to verify."
echo ""
EOF
chmod 755 "$PKG_DIR/DEBIAN/postinst"

# Pre-removal script
cat > "$PKG_DIR/DEBIAN/prerm" << 'EOF'
#!/bin/sh
set -e
# Nothing special needed — dpkg handles file removal
EOF
chmod 755 "$PKG_DIR/DEBIAN/prerm"

# ── Build .deb ────────────────────────────────────────────────────────────
echo "[5/5] Building .deb package..."

# Fix permissions
find "$PKG_DIR" -type d -exec chmod 755 {} \;
find "$PKG_DIR" -type f -exec chmod 644 {} \;
chmod 755 "$PKG_DIR/usr/bin/iris"
chmod 755 "$PKG_DIR/DEBIAN/postinst"
chmod 755 "$PKG_DIR/DEBIAN/prerm"

if command -v dpkg-deb &>/dev/null; then
    dpkg-deb --build --root-owner-group "$PKG_DIR" "$DIST_DIR/$DEB_FILE"
elif command -v fakeroot &>/dev/null; then
    fakeroot dpkg-deb --build "$PKG_DIR" "$DIST_DIR/$DEB_FILE"
else
    echo "Warning: dpkg-deb not found. Creating .deb with tar+ar..."
    # Manual .deb construction
    (cd "$PKG_DIR" && tar czf "$DIST_DIR/data.tar.gz" --owner=0 --group=0 -C "$PKG_DIR" usr)
    (cd "$PKG_DIR" && tar czf "$DIST_DIR/control.tar.gz" --owner=0 --group=0 -C "$PKG_DIR/DEBIAN" .)
    echo "2.0" > "$DIST_DIR/debian-binary"
    (cd "$DIST_DIR" && ar rcs "$DEB_FILE" debian-binary control.tar.gz data.tar.gz)
    rm -f "$DIST_DIR/data.tar.gz" "$DIST_DIR/control.tar.gz" "$DIST_DIR/debian-binary"
fi

# Cleanup staging
rm -rf "$PKG_DIR"

echo ""
echo "  Package ready: $DIST_DIR/$DEB_FILE"
SIZE_MB=$(du -m "$DIST_DIR/$DEB_FILE" | awk '{print $1}')
echo "  Size: ${SIZE_MB} MB"
echo ""
echo "  Install with:"
echo "    sudo dpkg -i $DEB_FILE"
echo "    # or"
echo "    sudo apt install ./$DEB_FILE"
echo ""
