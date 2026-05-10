#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# IRIS Language Installer for macOS
# ──────────────────────────────────────────────────────────────────────────
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/moon9t/iris/master/installer/macos/install.sh | bash
#   # or
#   bash install.sh
#
# Installs iris to ~/.iris/bin and adds it to PATH via shell profile.
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="0.3.0"
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
        x86_64|amd64)  echo "x86_64-apple-darwin" ;;
        aarch64|arm64) echo "aarch64-apple-darwin" ;;
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
    curl -fSL "$url" -o "$tmp/$archive"
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
    # macOS default shell is zsh since Catalina
    local profiles=("$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.bashrc" "$HOME/.bash_profile")

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

    # If no profile exists yet, create .zshrc (macOS default)
    local found=false
    for profile in "${profiles[@]}"; do
        if [[ -f "$profile" ]]; then found=true; break; fi
    done
    if [[ "$found" == "false" ]]; then
        echo "# IRIS Language" > "$HOME/.zshrc"
        echo "$line" >> "$HOME/.zshrc"
        ok "Created ~/.zshrc with PATH entry"
    fi

    export PATH="$BIN_DIR:$PATH"
}

# ── Install stdlib ────────────────────────────────────────────────────────
install_stdlib() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local stdlib_src=""

    if [[ -d "$script_dir/stdlib" ]]; then
        stdlib_src="$script_dir/stdlib"
    elif [[ -d "$script_dir/../../stdlib" ]]; then
        stdlib_src="$script_dir/../../stdlib"
    fi

    if [[ -n "$stdlib_src" ]]; then
        mkdir -p "$INSTALL_DIR/stdlib"
        cp -R "$stdlib_src"/* "$INSTALL_DIR/stdlib/" 2>/dev/null || true
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
        cp -R "$examples_src"/* "$INSTALL_DIR/examples/" 2>/dev/null || true
        ok "Installed examples to $INSTALL_DIR/examples"
    fi
}

# ── Install toolchain (bundled LLVM for native compilation) ──────────────
install_toolchain() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local tc_src=""

    if [[ -d "$script_dir/toolchain" ]]; then
        tc_src="$script_dir/toolchain"
    elif [[ -d "$script_dir/../../toolchain" ]]; then
        tc_src="$script_dir/../../toolchain"
    fi

    if [[ -n "$tc_src" ]]; then
        mkdir -p "$INSTALL_DIR/toolchain"
        cp -R "$tc_src"/* "$INSTALL_DIR/toolchain/" 2>/dev/null || true
        ok "Installed toolchain to $INSTALL_DIR/toolchain"
    else
        info "No bundled toolchain found — iris build will use system clang."
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

# ── Check clang (macOS ships with Apple Clang via Xcode CLT) ─────────────
check_clang() {
    if command -v clang &>/dev/null; then
        local ver
        ver="$(clang --version 2>/dev/null | head -1)"
        ok "clang detected: $ver"
    else
        info "clang not found. Install Xcode Command Line Tools:"
        info "  xcode-select --install"
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

# Remove quarantine attribute if present (Gatekeeper)
xattr -d com.apple.quarantine "$BIN_DIR/iris" 2>/dev/null || true
ok "Binary installed."

# Step 4: Install stdlib + examples + toolchain
step "Installing standard library, examples, and toolchain..."
install_stdlib
install_examples
install_toolchain

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
