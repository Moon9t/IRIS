#!/usr/bin/env bash
# build-all.sh — Cross-platform build orchestrator for IRIS installers
# ─────────────────────────────────────────────────────────────────────
# Detects the current OS/arch and builds all applicable installer formats.
#
# Usage:
#   ./installer/build-all.sh [--version 0.4.0] [--skip-build] [--format deb|rpm|appimage|pkg|dmg|all]
#
# Produces all outputs in installer/dist/
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

VERSION="0.4.0"
SKIP_BUILD=false
FORMAT="all"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)  VERSION="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --format)   FORMAT="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--version X.Y.Z] [--skip-build] [--format deb|rpm|appimage|pkg|dmg|all]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Detect platform ─────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)  PLATFORM="linux" ;;
    Darwin*) PLATFORM="macos" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
    *)       echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|amd64)  ARCH_LABEL="x86_64" ;;
    aarch64|arm64)  ARCH_LABEL="aarch64" ;;
    *)              echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

echo "════════════════════════════════════════════════════════════"
echo "  IRIS Installer Builder v${VERSION}"
echo "  Platform: ${PLATFORM} (${ARCH_LABEL})"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Ensure dist directory ────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/dist"

# ── Build release binary ─────────────────────────────────────────────
if [ "$SKIP_BUILD" = false ]; then
    echo "[BUILD] Compiling release binary..."
    cd "$ROOT_DIR"
    cargo build --release
    echo "[BUILD] Done."
    echo ""
fi

IRIS_BIN="$ROOT_DIR/target/release/iris"
if [ "$PLATFORM" = "windows" ]; then
    IRIS_BIN="${IRIS_BIN}.exe"
fi

if [ ! -f "$IRIS_BIN" ]; then
    echo "ERROR: iris binary not found at $IRIS_BIN"
    echo "       Run without --skip-build, or build manually first."
    exit 1
fi

# ── Track results ────────────────────────────────────────────────────
BUILT=()
SKIPPED=()
FAILED=()

run_builder() {
    local label="$1"
    local script="$2"
    shift 2

    echo "────────────────────────────────────────────────────────"
    echo "[${label}] Building..."
    if bash "$script" "$@" 2>&1; then
        BUILT+=("$label")
        echo "[${label}] SUCCESS"
    else
        FAILED+=("$label")
        echo "[${label}] FAILED"
    fi
    echo ""
}

# ── Linux formats ────────────────────────────────────────────────────
if [ "$PLATFORM" = "linux" ]; then
    if [ "$FORMAT" = "all" ] || [ "$FORMAT" = "deb" ]; then
        if command -v dpkg-deb &>/dev/null || command -v ar &>/dev/null; then
            run_builder "DEB" "$SCRIPT_DIR/linux/build-deb.sh" \
                --version "$VERSION" --arch "$ARCH_LABEL"
        else
            SKIPPED+=("DEB (dpkg-deb/ar not found)")
        fi
    fi

    if [ "$FORMAT" = "all" ] || [ "$FORMAT" = "rpm" ]; then
        if command -v rpmbuild &>/dev/null; then
            run_builder "RPM" "$SCRIPT_DIR/linux/build-rpm.sh" \
                --version "$VERSION" --arch "$ARCH_LABEL"
        else
            SKIPPED+=("RPM (rpmbuild not found)")
        fi
    fi

    if [ "$FORMAT" = "all" ] || [ "$FORMAT" = "appimage" ]; then
        run_builder "AppImage" "$SCRIPT_DIR/linux/build-appimage.sh" \
            --version "$VERSION"
    fi

    # Always build the tar.gz portable archive
    if [ "$FORMAT" = "all" ]; then
        echo "────────────────────────────────────────────────────────"
        echo "[TAR.GZ] Building portable archive..."
        ARCHIVE_NAME="IRIS-${VERSION}-linux-${ARCH_LABEL}.tar.gz"
        STAGE_DIR=$(mktemp -d)
        mkdir -p "$STAGE_DIR/iris-${VERSION}"
        cp "$IRIS_BIN" "$STAGE_DIR/iris-${VERSION}/"
        cp "$ROOT_DIR/LICENSE" "$STAGE_DIR/iris-${VERSION}/"
        cp "$ROOT_DIR/README.md" "$STAGE_DIR/iris-${VERSION}/"
        [ -d "$ROOT_DIR/stdlib" ] && cp -r "$ROOT_DIR/stdlib" "$STAGE_DIR/iris-${VERSION}/"
        [ -d "$ROOT_DIR/examples" ] && cp -r "$ROOT_DIR/examples" "$STAGE_DIR/iris-${VERSION}/"
        cp "$SCRIPT_DIR/linux/install.sh" "$STAGE_DIR/iris-${VERSION}/"
        cp "$SCRIPT_DIR/linux/uninstall.sh" "$STAGE_DIR/iris-${VERSION}/"
        tar czf "$SCRIPT_DIR/dist/$ARCHIVE_NAME" -C "$STAGE_DIR" "iris-${VERSION}"
        rm -rf "$STAGE_DIR"
        BUILT+=("TAR.GZ")
        echo "[TAR.GZ] SUCCESS"
        echo ""
    fi
fi

# ── macOS formats ────────────────────────────────────────────────────
if [ "$PLATFORM" = "macos" ]; then
    if [ "$FORMAT" = "all" ] || [ "$FORMAT" = "pkg" ]; then
        if command -v pkgbuild &>/dev/null; then
            run_builder "PKG" "$SCRIPT_DIR/macos/build-pkg.sh" \
                --version "$VERSION"
        else
            SKIPPED+=("PKG (pkgbuild not found)")
        fi
    fi

    if [ "$FORMAT" = "all" ] || [ "$FORMAT" = "dmg" ]; then
        if command -v hdiutil &>/dev/null; then
            run_builder "DMG" "$SCRIPT_DIR/macos/build-dmg.sh" \
                --version "$VERSION"
        else
            SKIPPED+=("DMG (hdiutil not found)")
        fi
    fi

    # Always build the tar.gz portable archive
    if [ "$FORMAT" = "all" ]; then
        echo "────────────────────────────────────────────────────────"
        echo "[TAR.GZ] Building portable archive..."
        ARCHIVE_NAME="IRIS-${VERSION}-macos-${ARCH_LABEL}.tar.gz"
        STAGE_DIR=$(mktemp -d)
        mkdir -p "$STAGE_DIR/iris-${VERSION}"
        cp "$IRIS_BIN" "$STAGE_DIR/iris-${VERSION}/"
        cp "$ROOT_DIR/LICENSE" "$STAGE_DIR/iris-${VERSION}/"
        cp "$ROOT_DIR/README.md" "$STAGE_DIR/iris-${VERSION}/"
        [ -d "$ROOT_DIR/stdlib" ] && cp -r "$ROOT_DIR/stdlib" "$STAGE_DIR/iris-${VERSION}/"
        [ -d "$ROOT_DIR/examples" ] && cp -r "$ROOT_DIR/examples" "$STAGE_DIR/iris-${VERSION}/"
        cp "$SCRIPT_DIR/macos/install.sh" "$STAGE_DIR/iris-${VERSION}/"
        cp "$SCRIPT_DIR/macos/uninstall.sh" "$STAGE_DIR/iris-${VERSION}/"
        tar czf "$SCRIPT_DIR/dist/$ARCHIVE_NAME" -C "$STAGE_DIR" "iris-${VERSION}"
        rm -rf "$STAGE_DIR"
        BUILT+=("TAR.GZ")
        echo "[TAR.GZ] SUCCESS"
        echo ""
    fi
fi

# ── Windows (under MSYS/Git Bash) ────────────────────────────────────
if [ "$PLATFORM" = "windows" ]; then
    echo "On Windows, use the PowerShell scripts directly:"
    echo "  - installer\\build_installer.ps1   (Inno Setup .exe)"
    echo "  - installer\\windows\\build_portable.ps1  (Portable .zip)"
    echo "  - installer\\windows\\build_msi.ps1       (WiX .msi)"
    echo ""
    echo "Or call from Git Bash:"

    if command -v powershell.exe &>/dev/null; then
        if [ "$FORMAT" = "all" ] || [ "$FORMAT" = "portable" ]; then
            echo "────────────────────────────────────────────────────────"
            echo "[PORTABLE ZIP] Building via PowerShell..."
            if powershell.exe -ExecutionPolicy Bypass -File \
                "$SCRIPT_DIR/windows/build_portable.ps1" \
                -Version "$VERSION" -SkipBuild 2>&1; then
                BUILT+=("PORTABLE ZIP")
            else
                FAILED+=("PORTABLE ZIP")
            fi
            echo ""
        fi
    else
        SKIPPED+=("Windows installers (PowerShell not available in this shell)")
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Build Summary"
echo "════════════════════════════════════════════════════════════"

if [ ${#BUILT[@]} -gt 0 ]; then
    echo ""
    echo "  Built successfully:"
    for item in "${BUILT[@]}"; do
        echo "    ✓ $item"
    done
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "  Skipped (missing tools):"
    for item in "${SKIPPED[@]}"; do
        echo "    - $item"
    done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  Failed:"
    for item in "${FAILED[@]}"; do
        echo "    ✗ $item"
    done
fi

echo ""
echo "  Output directory: $SCRIPT_DIR/dist/"
if [ -d "$SCRIPT_DIR/dist" ]; then
    echo ""
    ls -lh "$SCRIPT_DIR/dist/" 2>/dev/null | grep -v "^total"
fi
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
