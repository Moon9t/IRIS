#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# build-rpm.sh — Build an .rpm package for IRIS Language
# ──────────────────────────────────────────────────────────────────────────
# Usage:
#   ./installer/linux/build-rpm.sh [--version 0.4.0] [--arch x86_64]
#
# Produces: installer/dist/iris-<version>-1.<arch>.rpm
# Requires: rpmbuild (from rpm or rpm-build package)
#
# This script uses a direct BUILDROOT approach (no source tarball / %setup)
# so it works for cross-architecture builds on any host (e.g. building an
# aarch64 RPM on an x86_64 Ubuntu CI runner).
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="0.4.0"
ARCH="x86_64"
RELEASE="1"
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

# Map arch names for RPM
case "$ARCH" in
    x86_64|amd64)    RPM_ARCH="x86_64" ;;
    aarch64|arm64)   RPM_ARCH="aarch64" ;;
    *)               echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

echo "Building IRIS .rpm package v${VERSION} (${RPM_ARCH})"

# ── Paths ─────────────────────────────────────────────────────────────────
DIST_DIR="$ROOT/installer/dist"
RPM_TOPDIR="$ROOT/installer/dist/rpm-staging"
RPM_FILE="iris-${VERSION}-${RELEASE}.${RPM_ARCH}.rpm"

# ── Clean previous build ─────────────────────────────────────────────────
rm -rf "$RPM_TOPDIR"
mkdir -p "$RPM_TOPDIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS,BUILDROOT}
mkdir -p "$DIST_DIR"

# ── Locate release binary ────────────────────────────────────────────────
IRIS_BIN="$ROOT/target/release/iris"
if [[ ! -f "$IRIS_BIN" ]]; then
    echo "[1/4] Building release binary..."
    (cd "$ROOT" && cargo build --release)
else
    echo "[1/4] Using existing release binary."
fi

# ── Pre-populate BUILDROOT directly ──────────────────────────────────────
echo "[2/4] Staging files into BUILDROOT..."
BUILDROOT="$RPM_TOPDIR/BUILDROOT/iris-${VERSION}-${RELEASE}.${RPM_ARCH}"
mkdir -p "$BUILDROOT/usr/bin"
mkdir -p "$BUILDROOT/usr/share/iris/stdlib"
mkdir -p "$BUILDROOT/usr/share/iris/examples"
mkdir -p "$BUILDROOT/usr/share/iris/toolchain"
mkdir -p "$BUILDROOT/usr/share/doc/iris"

cp "$IRIS_BIN" "$BUILDROOT/usr/bin/iris"
chmod 755 "$BUILDROOT/usr/bin/iris"
cp -r "$ROOT/stdlib/"* "$BUILDROOT/usr/share/iris/stdlib/" 2>/dev/null || true
cp -r "$ROOT/examples/"* "$BUILDROOT/usr/share/iris/examples/" 2>/dev/null || true
if [[ -d "$ROOT/toolchain" ]]; then
    cp -r "$ROOT/toolchain/"* "$BUILDROOT/usr/share/iris/toolchain/" 2>/dev/null || true
fi
cp "$ROOT/LICENSE" "$BUILDROOT/usr/share/doc/iris/"
cp "$ROOT/README.md" "$BUILDROOT/usr/share/doc/iris/"

# Also create a staging area the spec %install can reference (rpmbuild
# cleans $BUILDROOT before running %install, so we need a second copy).
STAGED_BIN="$DIST_DIR/staged-bin"
STAGED_DATA="$DIST_DIR/staged-data"
mkdir -p "$STAGED_BIN" "$STAGED_DATA/stdlib" "$STAGED_DATA/examples" "$STAGED_DATA/toolchain"
cp "$IRIS_BIN" "$STAGED_BIN/iris"
chmod 755 "$STAGED_BIN/iris"
cp -r "$ROOT/stdlib/"* "$STAGED_DATA/stdlib/" 2>/dev/null || true
cp -r "$ROOT/examples/"* "$STAGED_DATA/examples/" 2>/dev/null || true
if [[ -d "$ROOT/toolchain" ]]; then
    cp -r "$ROOT/toolchain/"* "$STAGED_DATA/toolchain/" 2>/dev/null || true
fi
cp "$ROOT/LICENSE" "$STAGED_DATA/"
cp "$ROOT/README.md" "$STAGED_DATA/"

# ── Create spec file (binary-only — no %prep/%build) ─────────────────────
echo "[3/4] Creating RPM spec file..."
CHANGELOG_DATE="$(date '+%a %b %d %Y')"

cat > "$RPM_TOPDIR/SPECS/iris.spec" << SPECEOF
Name:           iris
Version:        ${VERSION}
Release:        ${RELEASE}
Summary:        IRIS programming language compiler and toolchain

License:        GPL-2.0-or-later
URL:            https://github.com/moon9t/iris

# Pre-built binary — skip debug package extraction, auto deps, and host strip
# (cross-arch builds: host strip cannot read foreign-arch ELF binaries)
%global debug_package %{nil}
%define __strip /bin/true
AutoReqProv:    no

%description
IRIS (Intermediate Representation for Intelligent Systems) is a
programming language designed for machine learning and systems
programming. Includes compiler, REPL, LSP server,
and DAP server.

# Binary package — files are staged directly into BUILDROOT by the build
# script. No source extraction, configuration, or compilation needed.
%prep
# nothing

%build
# nothing

%install
# rpmbuild cleans BUILDROOT before %install, so re-stage files here.
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/usr/share/iris/stdlib
mkdir -p %{buildroot}/usr/share/iris/examples
mkdir -p %{buildroot}/usr/share/iris/toolchain
mkdir -p %{buildroot}/usr/share/doc/iris
cp %{_topdir}/../staged-bin/iris %{buildroot}/usr/bin/iris
chmod 755 %{buildroot}/usr/bin/iris
cp -r %{_topdir}/../staged-data/stdlib/* %{buildroot}/usr/share/iris/stdlib/ 2>/dev/null || true
cp -r %{_topdir}/../staged-data/examples/* %{buildroot}/usr/share/iris/examples/ 2>/dev/null || true
cp -r %{_topdir}/../staged-data/toolchain/* %{buildroot}/usr/share/iris/toolchain/ 2>/dev/null || true
cp %{_topdir}/../staged-data/LICENSE %{buildroot}/usr/share/doc/iris/
cp %{_topdir}/../staged-data/README.md %{buildroot}/usr/share/doc/iris/

%files
%license /usr/share/doc/iris/LICENSE
%doc /usr/share/doc/iris/README.md
/usr/bin/iris
/usr/share/iris/

%post
echo ""
echo "  IRIS Language installed successfully!"
echo "  Run 'iris --version' to verify."
echo ""

%changelog
* ${CHANGELOG_DATE} IRIS Language Project <iris@moon9t.dev> - ${VERSION}-${RELEASE}
- Release v${VERSION}
SPECEOF

# ── Build .rpm ────────────────────────────────────────────────────────────
echo "[4/4] Building .rpm package..."

if ! command -v rpmbuild &>/dev/null; then
    echo "Error: rpmbuild not found."
    echo "Install it with:"
    echo "  Fedora/RHEL: sudo dnf install rpm-build"
    echo "  Ubuntu:      sudo apt install rpm"
    exit 1
fi

# Use --define to set architecture explicitly (avoids needing platform macros
# for cross-arch builds, e.g. building aarch64 on x86_64 Ubuntu CI).
rpmbuild \
    --define "_topdir $RPM_TOPDIR" \
    --define "_rpmdir $RPM_TOPDIR/RPMS" \
    --define "_build_name_fmt %%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.rpm" \
    --buildroot "$BUILDROOT" \
    --target "${RPM_ARCH}-linux" \
    -bb "$RPM_TOPDIR/SPECS/iris.spec"

# Copy the built RPM to dist
find "$RPM_TOPDIR/RPMS" -name '*.rpm' -exec cp {} "$DIST_DIR/" \;

# Cleanup staging
rm -rf "$RPM_TOPDIR" "$STAGED_BIN" "$STAGED_DATA"

# Find the output file
RPM_OUT="$(find "$DIST_DIR" -name 'iris-*.rpm' -type f 2>/dev/null | head -1)"
if [[ -z "$RPM_OUT" ]]; then
    RPM_OUT="$DIST_DIR/$RPM_FILE"
fi

echo ""
echo "  Package ready: $RPM_OUT"
if [[ -f "$RPM_OUT" ]]; then
    SIZE_MB=$(du -m "$RPM_OUT" | awk '{print $1}')
    echo "  Size: ${SIZE_MB} MB"
fi
echo ""
echo "  Install with:"
echo "    sudo rpm -i $RPM_FILE"
echo "    # or"
echo "    sudo dnf install ./$RPM_FILE"
echo ""
