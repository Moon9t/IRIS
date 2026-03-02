#Requires -Version 5.1
<#
.SYNOPSIS
    IRIS Language Installer for Windows
.DESCRIPTION
    Installs the IRIS programming language compiler, configures PATH,
    optionally installs the VSCode extension, and detects LLVM/clang.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Banner {
    Write-Host ""
    Write-Host "  ██╗██████╗ ██╗███████╗" -ForegroundColor Cyan
    Write-Host "  ██║██╔══██╗██║██╔════╝" -ForegroundColor Cyan
    Write-Host "  ██║██████╔╝██║███████╗" -ForegroundColor Cyan
    Write-Host "  ██║██╔══██╗██║╚════██║" -ForegroundColor Cyan
    Write-Host "  ██║██║  ██║██║███████║" -ForegroundColor Cyan
    Write-Host "  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  IRIS Language Installer  v0.2.0" -ForegroundColor White
    Write-Host "  Intermediate Representation for Intelligent Systems" -ForegroundColor DarkGray
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host "  --> $Message" -ForegroundColor Yellow
}

function Write-Ok {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "  [i]  $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host "  [!]  $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "  [X]  $Message" -ForegroundColor Red
}

function Write-Divider {
    Write-Host "  " + ("-" * 60) -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Locate iris.exe next to this script
# ---------------------------------------------------------------------------

function Find-IrisExe {
    $scriptDir = $PSScriptRoot
    if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }

    $candidate = Join-Path $scriptDir 'iris.exe'
    if (Test-Path $candidate) { return $candidate }

    # Fallback: ask the user
    Write-Warn "iris.exe not found next to this installer script."
    $userPath = Read-Host "  Enter full path to iris.exe (or press Enter to abort)"
    if ([string]::IsNullOrWhiteSpace($userPath)) {
        Write-Err "Installation aborted: iris.exe not provided."
        exit 1
    }
    if (-not (Test-Path $userPath)) {
        Write-Err "File not found: $userPath"
        exit 1
    }
    return $userPath
}

# ---------------------------------------------------------------------------
# Locate .vsix next to this script (optional)
# ---------------------------------------------------------------------------

function Find-Vsix {
    $scriptDir = $PSScriptRoot
    if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }

    $vsix = Get-ChildItem -Path $scriptDir -Filter 'iris-lang-*.vsix' -ErrorAction SilentlyContinue |
            Select-Object -First 1
    if ($vsix) { return $vsix.FullName }
    return $null
}

# ---------------------------------------------------------------------------
# PATH helpers
# ---------------------------------------------------------------------------

function Add-ToUserPath {
    param([string]$Dir)

    $regKey = 'HKCU:\Environment'
    $current = (Get-ItemProperty -Path $regKey -Name 'Path' -ErrorAction SilentlyContinue).Path
    if (-not $current) { $current = '' }

    $entries = $current -split ';' | Where-Object { $_ -ne '' }

    if ($entries -contains $Dir) {
        Write-Info "PATH already contains: $Dir"
        return
    }

    $newPath = ($entries + $Dir) -join ';'
    [System.Environment]::SetEnvironmentVariable('Path', $newPath, 'User')

    # Also update current session
    $env:PATH = "$env:PATH;$Dir"

    Write-Ok "Added to user PATH: $Dir"
}

# ---------------------------------------------------------------------------
# LLVM / clang detection
# ---------------------------------------------------------------------------

function Test-ClangInstalled {
    $standardPaths = @(
        'C:\Program Files\LLVM\bin\clang.exe',
        'C:\Program Files (x86)\LLVM\bin\clang.exe',
        "$env:USERPROFILE\scoop\shims\clang.exe",
        'C:\msys64\ucrt64\bin\clang.exe',
        'C:\msys64\mingw64\bin\clang.exe'
    )

    foreach ($p in $standardPaths) {
        if (Test-Path $p) { return $p }
    }

    # Check PATH
    $fromPath = Get-Command clang -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }

    return $null
}

# ---------------------------------------------------------------------------
# VSCode detection
# ---------------------------------------------------------------------------

function Get-CodeCmd {
    $candidates = @('code', 'code-insiders')
    foreach ($c in $candidates) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $c }
    }

    # Common installation paths
    $exePaths = @(
        "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd",
        "$env:LOCALAPPDATA\Programs\Microsoft VS Code Insiders\bin\code-insiders.cmd",
        'C:\Program Files\Microsoft VS Code\bin\code.cmd'
    )
    foreach ($p in $exePaths) {
        if (Test-Path $p) { return $p }
    }

    return $null
}

# ---------------------------------------------------------------------------
# Main installation logic
# ---------------------------------------------------------------------------

Write-Banner

# --- Step 1: Find iris.exe ---
Write-Step "Locating iris.exe..."
$irisExeSrc = Find-IrisExe
Write-Ok "Found: $irisExeSrc"

# --- Step 2: Create install directory ---
$installDir = Join-Path $env:USERPROFILE '.iris\bin'
Write-Step "Creating install directory: $installDir"
try {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    Write-Ok "Directory ready."
} catch {
    Write-Err "Failed to create directory: $_"
    exit 1
}

# --- Step 3: Copy iris.exe ---
$installDest = Join-Path $installDir 'iris.exe'
Write-Step "Installing iris.exe -> $installDest"
try {
    Copy-Item -Path $irisExeSrc -Destination $installDest -Force
    Write-Ok "Copied successfully."
} catch {
    Write-Err "Failed to copy iris.exe: $_"
    exit 1
}

# --- Step 4: Add to PATH ---
Write-Step "Configuring user PATH..."
Add-ToUserPath -Dir $installDir

# --- Step 5: VSCode extension ---
Write-Host ""
Write-Step "Checking for Visual Studio Code..."
$codeCmd = Get-CodeCmd

if ($codeCmd) {
    Write-Ok "VSCode detected: $codeCmd"
    $vsixPath = Find-Vsix
    if ($vsixPath) {
        Write-Step "Installing VSCode extension: $vsixPath"
        try {
            & $codeCmd --install-extension $vsixPath --force 2>&1 | Out-Null
            Write-Ok "VSCode extension installed."
        } catch {
            Write-Warn "Could not install VSCode extension automatically."
            Write-Warn "You can install it manually: code --install-extension iris-lang-0.2.0.vsix"
        }
    } else {
        Write-Info "No .vsix found next to installer — skipping VSCode extension."
        Write-Info "Download the extension from the IRIS releases page."
    }
} else {
    Write-Info "VSCode not detected — skipping extension install."
    Write-Info "Install VSCode from https://code.visualstudio.com/ for syntax highlighting."
}

# --- Step 6: Install bundled LLVM (clang + lld) ---
Write-Host ""
Write-Step "Installing bundled LLVM (clang + lld)..."

$scriptDir = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }

$BundledLlvmDir = Join-Path $scriptDir "toolchain\llvm\bin"
$LlvmInstallDir = "C:\Program Files\LLVM\bin"

$clangPath = Test-ClangInstalled
if ($clangPath) {
    Write-Ok "clang already present: $clangPath"
} elseif (Test-Path (Join-Path $BundledLlvmDir "clang.exe")) {
    try {
        New-Item -ItemType Directory -Force -Path $LlvmInstallDir | Out-Null
        Copy-Item "$BundledLlvmDir\*" $LlvmInstallDir -Force
        Write-Ok "Installed clang.exe + ld.lld.exe -> $LlvmInstallDir"
    } catch {
        Write-Warn "Could not copy to Program Files (may need admin)."
        Write-Info "Try running this installer as Administrator, or copy manually."
        $UserLlvm = Join-Path $env:USERPROFILE ".iris\llvm\bin"
        New-Item -ItemType Directory -Force -Path $UserLlvm | Out-Null
        Copy-Item "$BundledLlvmDir\*" $UserLlvm -Force
        $LlvmInstallDir = $UserLlvm
        Write-Ok "Installed clang + lld -> $UserLlvm (user-local fallback)"
    }
} else {
    Write-Warn "Bundled LLVM tools not found in installer package."
    Write-Info "Install LLVM manually from: https://releases.llvm.org/"
}

# --- Step 7: Install bundled MinGW sysroot (headers + libs) ---
Write-Host ""
Write-Step "Installing MinGW sysroot (headers + libraries)..."

$BundledUcrt64 = Join-Path $scriptDir "toolchain\ucrt64"
$Ucrt64InstallDir = "C:\msys64\ucrt64"

# Check if ucrt64 sysroot is already present (lib + include directories)
$existingSysroot = (Test-Path "$Ucrt64InstallDir\lib") -and (Test-Path "$Ucrt64InstallDir\include")

if ($existingSysroot) {
    Write-Ok "MinGW sysroot already present: $Ucrt64InstallDir"
} elseif (Test-Path $BundledUcrt64) {
    try {
        New-Item -ItemType Directory -Force -Path "$Ucrt64InstallDir\lib"     | Out-Null
        New-Item -ItemType Directory -Force -Path "$Ucrt64InstallDir\include" | Out-Null

        # Copy lib (CRT objects, static libs, GCC internal libs)
        if (Test-Path "$BundledUcrt64\lib") {
            Copy-Item "$BundledUcrt64\lib\*" "$Ucrt64InstallDir\lib\" -Force -Recurse
            Write-Ok "Copied MinGW libraries"
        }

        # Copy include (C headers)
        if (Test-Path "$BundledUcrt64\include") {
            Copy-Item "$BundledUcrt64\include\*" "$Ucrt64InstallDir\include\" -Force -Recurse
            Write-Ok "Copied MinGW headers"
        }
    } catch {
        Write-Warn "Could not install MinGW sysroot to $Ucrt64InstallDir (may need admin)."
        $UserUcrt64 = Join-Path $env:USERPROFILE ".iris\ucrt64"
        New-Item -ItemType Directory -Force -Path "$UserUcrt64\lib"     | Out-Null
        New-Item -ItemType Directory -Force -Path "$UserUcrt64\include" | Out-Null
        if (Test-Path "$BundledUcrt64\lib")     { Copy-Item "$BundledUcrt64\lib\*"     "$UserUcrt64\lib\"     -Force -Recurse }
        if (Test-Path "$BundledUcrt64\include") { Copy-Item "$BundledUcrt64\include\*" "$UserUcrt64\include\" -Force -Recurse }
        $Ucrt64InstallDir = $UserUcrt64
        Write-Ok "Installed MinGW sysroot -> $UserUcrt64 (user-local fallback)"
    }
} else {
    Write-Warn "Bundled MinGW sysroot not found in installer package."
    Write-Info "Install MSYS2 manually from: https://www.msys2.org/"
}

# Ensure LLVM bin dir is in user PATH (no need for ucrt64\bin — no GCC executables)
if (Test-Path $LlvmInstallDir)             { Add-ToUserPath -Dir $LlvmInstallDir }

# --- Step 8: Verify installation ---
Write-Host ""
Write-Step "Verifying installation..."
try {
    $versionOutput = & $installDest --version 2>&1
    Write-Ok "iris.exe responds: $versionOutput"
} catch {
    Write-Warn "Could not run iris.exe to verify. Try opening a new terminal and running: iris --version"
}

# --- Done ---
Write-Host ""
Write-Host ("  " + "=" * 60) -ForegroundColor Green
Write-Host "  IRIS installed successfully!" -ForegroundColor Green
Write-Host ("  " + "=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "  Quick Start:" -ForegroundColor White
Write-Host "    Open a new terminal, then:" -ForegroundColor DarkGray
Write-Host ""
Write-Host "    iris --version                    # verify install" -ForegroundColor Cyan
Write-Host "    iris --emit ir hello.iris         # show IR" -ForegroundColor Cyan
Write-Host "    iris --emit llvm hello.iris       # show LLVM IR" -ForegroundColor Cyan
Write-Host "    iris build hello.iris -o hello    # compile native binary" -ForegroundColor Cyan
Write-Host "    iris run hello.iris               # compile and run" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Documentation: https://github.com/iris-lang/iris" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  NOTE: You may need to open a new terminal for PATH changes to take effect." -ForegroundColor Yellow
Write-Host ""
