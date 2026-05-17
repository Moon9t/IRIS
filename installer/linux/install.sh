#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# IRIS Language Installer for Linux
# ──────────────────────────────────────────────────────────────────────────
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/moon9t/iris/master/installer/linux/install.sh | bash
#   # or
#   bash install.sh
#
# Installs iris to ~/.iris/bin and adds it to PATH via shell profile.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="0.4.0"
INSTALL_DIR="${IRIS_INSTALL_DIR:-$HOME/.iris}"
BIN_DIR="$INSTALL_DIR/bin"

# ── Colours ───────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

banner() {
    echo ""
    echo -e "${CYAN}  ██╗██████╗ ██╗███████╗${RESET}"
    echo -e "${CYAN}  ██║██╔══██╗██║██╔════╝${RESET}"
    echo -e "${CYAN}  ██║██████╔╝██║███████╗${RESET}"
    echo -e "${CYAN}  ██║██╔══██╗██║╚════██║${RESET}"
    echo -e "${CYAN}  ██║██║  ██║██║███████║${RESET}"
    echo -e "${CYAN}  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝${RESET}"
    echo ""
    echo -e "  ${BOLD}IRIS Language Installer  v${VERSION}${RESET}"
    echo -e "  Intermediate Representation for Intelligent Systems"
    echo ""
}

step()  { echo -e "  ${YELLOW}-->${RESET} $1"; }
ok()    { echo -e "  ${GREEN}[OK]${RESET} $1"; }
info()  { echo -e "  ${CYAN}[i]${RESET}  $1"; }
warn()  { echo -e "  ${YELLOW}[!]${RESET}  $1"; }
err()   { echo -e "  ${RED}[X]${RESET}  $1"; }

# ── Detect architecture ──────────────────────────────────────────────────
detect_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64)  echo "x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
        *)             err "Unsupported architecture: $arch"; exit 1 ;;
    esac
}

# ── Find iris binary ─────────────────────────────────────────────────────
find_iris_binary() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # 1. Next to this script
    if [[ -f "$script_dir/iris" ]]; then
        echo "$script_dir/iris"
        return
    fi

    # 2. In parent's target/release
    local release="$script_dir/../../target/release/iris"
    if [[ -f "$release" ]]; then
        echo "$release"
        return
    fi

    # 3. Download from GitHub
    local target
    target="$(detect_arch)"
    local tag="v${VERSION}"
    local archive="iris-${tag}-${target}.tar.gz"
    local url="https://github.com/moon9t/iris/releases/download/${tag}/${archive}"

    step "Downloading IRIS ${tag} for ${target}..."
    local tmp
    tmp="$(mktemp -d)"
    if command -v curl &>/dev/null; then
        curl -fSL "$url" -o "$tmp/$archive"
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$tmp/$archive"
    else
        err "Neither curl nor wget found. Cannot download IRIS."
        exit 1
    fi

    tar xzf "$tmp/$archive" -C "$tmp"
    if [[ -f "$tmp/iris" ]]; then
        echo "$tmp/iris"
    else
        err "iris binary not found in downloaded archive."
        exit 1
    fi
}

# ── Add to shell profile ─────────────────────────────────────────────────
add_to_path() {
    local line="export PATH=\"$BIN_DIR:\$PATH\""
    local profiles=("$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile")

    for profile in "${profiles[@]}"; do
        if [[ -f "$profile" ]]; then
            if ! grep -qF "$BIN_DIR" "$profile" 2>/dev/null; then
                echo "" >> "$profile"
                echo "# IRIS Language" >> "$profile"
                echo "$line" >> "$profile"
                ok "Added to PATH in $(basename "$profile")"
            else
                info "PATH already configured in $(basename "$profile")"
            fi
        fi
    done

    # Also update current session
    export PATH="$BIN_DIR:$PATH"
}

# ── Install stdlib ────────────────────────────────────────────────────────
install_stdlib() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local stdlib_src=""

    # Check next to script
    if [[ -d "$script_dir/stdlib" ]]; then
        stdlib_src="$script_dir/stdlib"
    elif [[ -d "$script_dir/../../stdlib" ]]; then
        stdlib_src="$script_dir/../../stdlib"
    fi

    if [[ -n "$stdlib_src" ]]; then
        mkdir -p "$INSTALL_DIR/stdlib"
        cp -r "$stdlib_src"/* "$INSTALL_DIR/stdlib/" 2>/dev/null || true
        ok "Installed stdlib to $INSTALL_DIR/stdlib"
    fi
}

# ── Install examples ─────────────────────────────────────────────────────
install_examples() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local examples_src=""

    if [[ -d "$script_dir/examples" ]]; then
        examples_src="$script_dir/examples"
    elif [[ -d "$script_dir/../../examples" ]]; then
        examples_src="$script_dir/../../examples"
    fi

    if [[ -n "$examples_src" ]]; then
        mkdir -p "$INSTALL_DIR/examples"
        cp -r "$examples_src"/* "$INSTALL_DIR/examples/" 2>/dev/null || true
        ok "Installed examples to $INSTALL_DIR/examples"
    fi
}

# ── VSCode extension ─────────────────────────────────────────────────────
install_vscode_extension() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    local vsix
    vsix="$(find "$script_dir" -maxdepth 2 -name 'iris-lang-*.vsix' 2>/dev/null | head -1)"

    if [[ -z "$vsix" ]]; then
        vsix="$(find "$script_dir/../.." -maxdepth 2 -name 'iris-lang-*.vsix' 2>/dev/null | head -1)"
    fi

    if [[ -z "$vsix" ]]; then
        info "No .vsix found — skipping VSCode extension."
        return
    fi

    if command -v code &>/dev/null; then
        step "Installing VSCode extension..."
        code --install-extension "$vsix" --force 2>/dev/null && ok "VSCode extension installed." || \
            warn "Could not install VSCode extension automatically."
    else
        info "VSCode not detected — skipping extension install."
    fi
}

# ── Check clang ───────────────────────────────────────────────────────────
check_clang() {
    if command -v clang &>/dev/null; then
        local ver
        ver="$(clang --version 2>/dev/null | head -1)"
        ok "clang detected: $ver"
    else
        info "clang not found. Install it for iris run/build/eval/jit:"
        info "  Ubuntu/Debian: sudo apt install clang lld"
        info "  Fedora:        sudo dnf install clang lld"
        info "  Arch:          sudo pacman -S clang lld"
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
banner

# Step 1: Find binary
step "Locating iris binary..."
IRIS_BIN="$(find_iris_binary)"
ok "Found: $IRIS_BIN"

# Step 2: Create install directory
step "Creating install directory: $BIN_DIR"
mkdir -p "$BIN_DIR"
ok "Directory ready."

# Step 3: Install binary
step "Installing iris -> $BIN_DIR/iris"
cp "$IRIS_BIN" "$BIN_DIR/iris"
chmod +x "$BIN_DIR/iris"
ok "Binary installed."

# Step 4: Install stdlib + examples
step "Installing standard library and examples..."
install_stdlib
install_examples

# Step 5: Add to PATH
step "Configuring PATH..."
add_to_path

# Step 6: VSCode extension
install_vscode_extension

# Step 7: Check clang
echo ""
step "Checking for clang..."
check_clang

# Step 8: Verify
echo ""
step "Verifying installation..."
if "$BIN_DIR/iris" --version &>/dev/null; then
    ok "iris responds: $("$BIN_DIR/iris" --version 2>&1)"
else
    warn "Could not run iris to verify. Try opening a new terminal and running: iris --version"
fi

# Done
echo ""
echo -e "  ${GREEN}════════════════════════════════════════════════════════════${RESET}"
echo -e "  ${GREEN}  IRIS installed successfully!${RESET}"
echo -e "  ${GREEN}════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  Quick Start:"
echo -e "    ${CYAN}iris --version${RESET}                    # verify install"
echo -e "    ${CYAN}iris run hello.iris${RESET}               # run a program"
echo -e "    ${CYAN}iris build hello.iris -o hello${RESET}    # compile native binary"
echo -e "    ${CYAN}iris repl${RESET}                         # interactive REPL"
echo ""
echo -e "  ${YELLOW}NOTE: Open a new terminal for PATH changes to take effect.${RESET}"
echo ""
