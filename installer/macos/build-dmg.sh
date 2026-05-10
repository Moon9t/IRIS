#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# build-dmg.sh — Build a macOS .dmg disk image for IRIS Language
# ──────────────────────────────────────────────────────────────────────────
# Usage:
#   ./installer/macos/build-dmg.sh [--version 0.3.0]
#
# Produces: installer/dist/IRIS-<version>-macos-<arch>.dmg
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="0.3.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        *)         echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)  ARCH_LABEL="x64"   ;;
    aarch64|arm64) ARCH_LABEL="arm64" ;;
    *) echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

echo "Building IRIS macOS .dmg v${VERSION} (${ARCH_LABEL})"

# ── Paths ─────────────────────────────────────────────────────────────────
DIST_DIR="$ROOT/installer/dist"
DMG_DIR="$DIST_DIR/dmg-staging"
DMG_FILE="IRIS-${VERSION}-macos-${ARCH_LABEL}.dmg"
VOL_NAME="IRIS ${VERSION}"

mkdir -p "$DIST_DIR"
rm -rf "$DMG_DIR"
mkdir -p "$DMG_DIR"

# ── Build release binary ─────────────────────────────────────────────────
IRIS_BIN="$ROOT/target/release/iris"
if [[ ! -f "$IRIS_BIN" ]]; then
    echo "[1/4] Building release binary..."
    (cd "$ROOT" && cargo build --release)
else
    echo "[1/4] Using existing release binary."
fi

# ── Stage files ───────────────────────────────────────────────────────────
echo "[2/4] Staging files..."

cp "$IRIS_BIN" "$DMG_DIR/iris"
chmod 755 "$DMG_DIR/iris"
strip "$DMG_DIR/iris" 2>/dev/null || true
xattr -cr "$DMG_DIR/iris" 2>/dev/null || true

# Copy install script
cp "$SCRIPT_DIR/install.sh" "$DMG_DIR/install.sh"
chmod 755 "$DMG_DIR/install.sh"

# Stdlib and examples
cp -R "$ROOT/stdlib" "$DMG_DIR/stdlib" 2>/dev/null || true
cp -R "$ROOT/examples" "$DMG_DIR/examples" 2>/dev/null || true

# Bundled LLVM toolchain
if [[ -d "$ROOT/toolchain" ]]; then
    cp -R "$ROOT/toolchain" "$DMG_DIR/toolchain" 2>/dev/null || true
    echo "  Bundled LLVM toolchain"
fi

cp "$ROOT/LICENSE" "$DMG_DIR/"
cp "$ROOT/README.md" "$DMG_DIR/"

# VSCode extension
VSIX="$(find "$ROOT/vscode-iris" -name 'iris-lang-*.vsix' 2>/dev/null | sort -rV | head -1)"
if [[ -n "$VSIX" ]]; then
    cp "$VSIX" "$DMG_DIR/"
fi

# Create a quick-start README for the DMG
cat > "$DMG_DIR/INSTALL.txt" << EOF
IRIS Language v${VERSION}
========================

Quick Install:
  1. Open Terminal
  2. cd to this disk image
  3. Run: bash install.sh

Manual Install:
  1. Copy 'iris' to /usr/local/bin/ (or ~/.iris/bin/)
  2. Add the directory to your PATH
  3. Run: iris --version

For native compilation:
  xcode-select --install

Documentation: https://github.com/moon9t/iris
EOF

# ── Create DMG ────────────────────────────────────────────────────────────
echo "[3/4] Creating DMG..."

# Create a temporary DMG, then convert to compressed
TEMP_DMG="$DIST_DIR/iris-temp.dmg"
hdiutil create \
    -volname "$VOL_NAME" \
    -srcfolder "$DMG_DIR" \
    -ov \
    -format UDRW \
    "$TEMP_DMG"

echo "[4/4] Compressing DMG..."
hdiutil convert \
    "$TEMP_DMG" \
    -format UDZO \
    -imagekey zlib-level=9 \
    -o "$DIST_DIR/$DMG_FILE"

rm -f "$TEMP_DMG"
rm -rf "$DMG_DIR"

echo ""
echo "  DMG ready: $DIST_DIR/$DMG_FILE"
if [[ -f "$DIST_DIR/$DMG_FILE" ]]; then
    SIZE_MB=$(du -m "$DIST_DIR/$DMG_FILE" | awk '{print $1}')
    echo "  Size: ${SIZE_MB} MB"
fi
echo ""
