#Requires -Version 5.1
<#
.SYNOPSIS
    IRIS Language Uninstaller for Windows
.DESCRIPTION
    Removes the IRIS programming language compiler, its PATH entry,
    and optionally the VSCode extension.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File uninstall.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'SilentlyContinue'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Banner {
    Write-Host ""
    Write-Host "  IRIS Language Uninstaller  v0.1.0" -ForegroundColor White
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

# ---------------------------------------------------------------------------
# PATH helpers
# ---------------------------------------------------------------------------

function Remove-FromUserPath {
    param([string]$Dir)

    $regKey = 'HKCU:\Environment'
    $current = (Get-ItemProperty -Path $regKey -Name 'Path' -ErrorAction SilentlyContinue).Path
    if (-not $current) {
        Write-Info "User PATH is empty — nothing to remove."
        return
    }

    $entries = $current -split ';' | Where-Object { $_ -ne '' -and $_ -ne $Dir }
    $newPath = $entries -join ';'

    if ($newPath -eq $current) {
        Write-Info "PATH did not contain: $Dir"
        return
    }

    [System.Environment]::SetEnvironmentVariable('Path', $newPath, 'User')

    # Update current session too
    $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -ne $Dir }) -join ';'

    Write-Ok "Removed from user PATH: $Dir"
}

# ---------------------------------------------------------------------------
# VSCode extension removal
# ---------------------------------------------------------------------------

function Get-CodeCmd {
    $candidates = @('code', 'code-insiders')
    foreach ($c in $candidates) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $c }
    }

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

function Remove-VsCodeExtension {
    $codeCmd = Get-CodeCmd
    if (-not $codeCmd) {
        Write-Info "VSCode not detected — no extension to remove."
        return
    }

    Write-Step "Checking for installed IRIS VSCode extension..."
    try {
        $installed = & $codeCmd --list-extensions 2>&1 | Where-Object { $_ -match 'iris' }
        if ($installed) {
            foreach ($ext in $installed) {
                Write-Step "Uninstalling extension: $ext"
                & $codeCmd --uninstall-extension $ext 2>&1 | Out-Null
                Write-Ok "Extension removed: $ext"
            }
        } else {
            Write-Info "No IRIS extension found in VSCode."
        }
    } catch {
        Write-Warn "Could not query/remove VSCode extension: $_"
    }
}

# ---------------------------------------------------------------------------
# Confirmation prompt
# ---------------------------------------------------------------------------

function Confirm-Action {
    param([string]$Prompt)
    $answer = Read-Host "  $Prompt [y/N]"
    return $answer -match '^[yY]'
}

# ---------------------------------------------------------------------------
# Main uninstall logic
# ---------------------------------------------------------------------------

Write-Banner

$installDir = Join-Path $env:USERPROFILE '.iris\bin'
$irisRoot   = Join-Path $env:USERPROFILE '.iris'

Write-Host "  This will uninstall IRIS from your machine." -ForegroundColor White
Write-Host "  Install directory: $irisRoot" -ForegroundColor DarkGray
Write-Host ""

if (-not (Confirm-Action "Proceed with uninstallation?")) {
    Write-Host ""
    Write-Info "Uninstall cancelled."
    exit 0
}

Write-Host ""

# --- Step 1: Remove from PATH ---
Write-Step "Removing $installDir from user PATH..."
Remove-FromUserPath -Dir $installDir

# --- Step 2: Remove VSCode extension ---
Write-Host ""
Remove-VsCodeExtension

# --- Step 3: Remove install directory ---
Write-Host ""
Write-Step "Removing IRIS installation directory: $irisRoot"

if (Test-Path $irisRoot) {
    $confirmed = Confirm-Action "Delete $irisRoot and all its contents?"
    if ($confirmed) {
        try {
            Remove-Item -Recurse -Force -Path $irisRoot
            Write-Ok "Removed: $irisRoot"
        } catch {
            Write-Err "Failed to remove directory: $_"
            Write-Warn "You can delete it manually: Remove-Item -Recurse -Force '$irisRoot'"
        }
    } else {
        Write-Info "Skipped directory removal. Files remain at: $irisRoot"
    }
} else {
    Write-Info "Directory not found (already removed): $irisRoot"
}

# --- Done ---
Write-Host ""
Write-Host ("  " + "=" * 60) -ForegroundColor Green
Write-Host "  IRIS has been uninstalled." -ForegroundColor Green
Write-Host ("  " + "=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "  Open a new terminal for PATH changes to take effect." -ForegroundColor Yellow
Write-Host ""
