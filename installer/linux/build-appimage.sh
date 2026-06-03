#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# build-appimage.sh — Build an AppImage for IRIS Language
# ──────────────────────────────────────────────────────────────────────────
# Usage:
#   ./installer/linux/build-appimage.sh [--version 0.6.1]
#
# Produces: installer/dist/IRIS-<version>-<arch>.AppImage
# Self-contained, runs on any Linux distro (glibc 2.17+).
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="0.6.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        *)         echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ARCH="$(uname -m)"
DIST_DIR="$ROOT/installer/dist"
APPDIR="$DIST_DIR/IRIS.AppDir"

echo "Building IRIS AppImage v${VERSION} (${ARCH})"

# ── Clean ─────────────────────────────────────────────────────────────────
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/iris/stdlib"
mkdir -p "$APPDIR/usr/share/iris/examples"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$DIST_DIR"

# ── Build release binary ─────────────────────────────────────────────────
IRIS_BIN="$ROOT/target/release/iris"
if [[ ! -f "$IRIS_BIN" ]]; then
    echo "[1/5] Building release binary..."
    (cd "$ROOT" && cargo build --release)
else
    echo "[1/5] Using existing release binary."
fi

# ── Stage files ───────────────────────────────────────────────────────────
echo "[2/5] Staging files..."

cp "$IRIS_BIN" "$APPDIR/usr/bin/iris"
chmod 755 "$APPDIR/usr/bin/iris"
strip "$APPDIR/usr/bin/iris" 2>/dev/null || true

cp -r "$ROOT/stdlib/"* "$APPDIR/usr/share/iris/stdlib/" 2>/dev/null || true
cp -r "$ROOT/examples/"* "$APPDIR/usr/share/iris/examples/" 2>/dev/null || true

# Icon
if [[ -f "$ROOT/logo/iris-logo.png" ]]; then
    cp "$ROOT/logo/iris-logo.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/iris.png"
    cp "$ROOT/logo/iris-logo.png" "$APPDIR/iris.png"
else
    # Generate a minimal placeholder icon
    echo "  [i] No logo found, skipping icon."
fi

# ── Desktop entry ─────────────────────────────────────────────────────────
echo "[3/5] Creating desktop entry..."
cat > "$APPDIR/iris.desktop" << EOF
[Desktop Entry]
Type=Application
Name=IRIS Language
Comment=IRIS programming language compiler and toolchain
Exec=iris
Icon=iris
Categories=Development;IDE;
Terminal=true
MimeType=text/x-iris;
EOF

# ── AppRun script ─────────────────────────────────────────────────────────
echo "[4/5] Creating AppRun..."
cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
SELF="$(readlink -f "$0")"
HERE="${SELF%/*}"
export IRIS_STDLIB="${HERE}/usr/share/iris/stdlib"
export PATH="${HERE}/usr/bin:${PATH}"
exec "${HERE}/usr/bin/iris" "$@"
EOF
chmod 755 "$APPDIR/AppRun"

# ── Build AppImage ────────────────────────────────────────────────────────
echo "[5/5] Packaging AppImage..."

APPIMAGETOOL="$DIST_DIR/appimagetool"
APPIMAGE_NAME="IRIS-${VERSION}-${ARCH}.AppImage"

if [[ ! -f "$APPIMAGETOOL" ]]; then
    echo "  Downloading appimagetool..."
    APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"
    if command -v curl &>/dev/null; then
        curl -fSL "$APPIMAGETOOL_URL" -o "$APPIMAGETOOL"
    elif command -v wget &>/dev/null; then
        wget -q "$APPIMAGETOOL_URL" -O "$APPIMAGETOOL"
    else
        echo "Error: Neither curl nor wget found."
        exit 1
    fi
    chmod +x "$APPIMAGETOOL"
fi

ARCH="$ARCH" "$APPIMAGETOOL" "$APPDIR" "$DIST_DIR/$APPIMAGE_NAME"

# Cleanup staging
rm -rf "$APPDIR"

echo ""
echo "  AppImage ready: $DIST_DIR/$APPIMAGE_NAME"
if [[ -f "$DIST_DIR/$APPIMAGE_NAME" ]]; then
    SIZE_MB=$(du -m "$DIST_DIR/$APPIMAGE_NAME" | awk '{print $1}')
    echo "  Size: ${SIZE_MB} MB"
fi
echo ""
echo "  Run with:"
echo "    chmod +x $APPIMAGE_NAME && ./$APPIMAGE_NAME --version"
echo ""
