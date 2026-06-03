# setup_dependencies.ps1 - Smart dependency downloader and configurator for IRIS
# Checks for Git, LLVM, MinGW (ucrt64), and MSVC Build Tools. Downloads and configures them if missing.

param(
    [string]$InstallDir = "$env:LOCALAPPDATA\Programs\IRIS"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "         IRIS Dependency Checker & Downloader             " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Installation Directory: $InstallDir" -ForegroundColor Gray
Write-Host ""

# Helper to check if a command exists in PATH
function Test-CommandExists($command) {
    $item = Get-Command $command -ErrorAction SilentlyContinue
    return $null -ne $item
}

# Helper to add a directory to user PATH
function Add-ToPath($dir) {
    if (-not (Test-Path $dir)) { return }
    $regKey = 'HKCU:\Environment'
    $current = (Get-ItemProperty -Path $regKey -Name 'Path' -ErrorAction SilentlyContinue).Path
    if (-not $current) { $current = '' }
    $entries = $current -split ';' | Where-Object { $_ -ne '' }
    if (-not ($entries -contains $dir)) {
        $newPath = ($entries + $dir) -join ';'
        [System.Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        $env:PATH = "$env:PATH;$dir"
        Write-Host "  Added to user PATH: $dir" -ForegroundColor Green
    }
}

# Helper to download file with progress bar
function Download-FileWithProgress($url, $destPath) {
    Write-Host "Downloading $url..." -ForegroundColor Cyan
    $tempFile = $destPath
    $clnt = New-Object System.Net.WebClient
    $clnt.DownloadFile($url, $tempFile)
    Write-Host "Download complete: $destPath" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 1. Check Git
# ---------------------------------------------------------------------------
Write-Host "[1/4] Checking Git..." -ForegroundColor Yellow
if (Test-CommandExists "git") {
    $gitVer = (git --version)
    Write-Host "  Git is already installed: $gitVer" -ForegroundColor Green
} else {
    Write-Host "  Git not found. Installing MinGit (lightweight portable Git)..." -ForegroundColor Cyan
    $gitDir = Join-Path $InstallDir "toolchain\git"
    if (-not (Test-Path $gitDir)) { New-Item -ItemType Directory -Force -Path $gitDir | Out-Null }
    
    $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/MinGit-2.44.0-64-bit.zip"
    $tempZip = Join-Path $env:TEMP "mingit.zip"
    
    try {
        Download-FileWithProgress $gitUrl $tempZip
        Write-Host "  Extracting MinGit to $gitDir..." -ForegroundColor Cyan
        Expand-Archive -Path $tempZip -DestinationPath $gitDir -Force
        Remove-Item $tempZip
        
        $gitCmdDir = Join-Path $gitDir "cmd"
        Add-ToPath $gitCmdDir
        Write-Host "  MinGit configured successfully." -ForegroundColor Green
    } catch {
        Write-Host "  Warning: MinGit download/extraction failed: $_" -ForegroundColor Red
    }
}

# ---------------------------------------------------------------------------
# 2. Check LLVM (clang + lld)
# ---------------------------------------------------------------------------
Write-Host "`n[2/4] Checking LLVM (clang + lld)..." -ForegroundColor Yellow
$clangPath = $null
if (Test-CommandExists "clang") {
    $clangPath = (Get-Command clang).Source
} else {
    $stdPaths = @(
        "C:\Program Files\LLVM\bin\clang.exe",
        "C:\Program Files (x86)\LLVM\bin\clang.exe",
        (Join-Path $InstallDir "toolchain\llvm\bin\clang.exe")
    )
    foreach ($p in $stdPaths) {
        if (Test-Path $p) { $clangPath = $p; break }
    }
}

if ($null -ne $clangPath) {
    Write-Host "  LLVM/clang is already installed at: $clangPath" -ForegroundColor Green
    $llvmBinDir = Split-Path $clangPath
    Add-ToPath $llvmBinDir
} else {
    Write-Host "  LLVM not found. Downloading official LLVM installer..." -ForegroundColor Cyan
    $llvmUrl = "https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/LLVM-17.0.6-win64.exe"
    $tempExe = Join-Path $env:TEMP "llvm-setup.exe"
    
    try {
        Download-FileWithProgress $llvmUrl $tempExe
        Write-Host "  Running LLVM installer silently (destination C:\Program Files\LLVM)..." -ForegroundColor Cyan
        # Install LLVM silently
        $proc = Start-Process -FilePath $tempExe -ArgumentList "/S", "/D=C:\Program Files\LLVM" -Wait -PassThru -NoNewWindow
        Remove-Item $tempExe
        
        if ($proc.ExitCode -eq 0 -or (Test-Path "C:\Program Files\LLVM\bin\clang.exe")) {
            Write-Host "  LLVM installed successfully." -ForegroundColor Green
            Add-ToPath "C:\Program Files\LLVM\bin"
        } else {
            Write-Host "  Warning: LLVM installer exited with non-zero code: $($proc.ExitCode)" -ForegroundColor Red
        }
    } catch {
        Write-Host "  Warning: LLVM download/installation failed: $_" -ForegroundColor Red
    }
}

# ---------------------------------------------------------------------------
# 3. Check MinGW (ucrt64 sysroot headers/libs)
# ---------------------------------------------------------------------------
Write-Host "`n[3/4] Checking MinGW sysroot..." -ForegroundColor Yellow
$sysrootPath = "C:\msys64\ucrt64"
$hasSysroot = (Test-Path "$sysrootPath\include\stdint.h") -and (Test-Path "$sysrootPath\lib\libmingw32.a")

if ($hasSysroot) {
    Write-Host "  MinGW sysroot already present at: $sysrootPath" -ForegroundColor Green
} else {
    # Check fallback local path
    $localSysroot = Join-Path $InstallDir "toolchain\ucrt64"
    if ((Test-Path "$localSysroot\include\stdint.h") -and (Test-Path "$localSysroot\lib\libmingw32.a")) {
        Write-Host "  MinGW sysroot already present at: $localSysroot" -ForegroundColor Green
    } else {
        Write-Host "  MinGW sysroot not found. Downloading portable WinLibs MinGW..." -ForegroundColor Cyan
        # Download lightweight winlibs zip containing GCC/MinGW (includes all headers and libs)
        $mingwUrl = "https://github.com/brechtsanders/winlibs_mingw/releases/download/14.1.0posix-18.1.5-12.0.0-msvcrt-r1/winlibs-x86_64-posix-seh-gcc-14.1.0-mingw-w64msvcrt-12.0.0-r1.zip"
        $tempZip = Join-Path $env:TEMP "mingw.zip"
        
        try {
            Download-FileWithProgress $mingwUrl $tempZip
            Write-Host "  Extracting MinGW sysroot to $localSysroot..." -ForegroundColor Cyan
            # Extract to temp dir first since winlibs zip wraps everything in a 'mingw64' folder
            $tempExt = Join-Path $env:TEMP "mingw_ext"
            if (Test-Path $tempExt) { Remove-Item $tempExt -Recurse -Force }
            New-Item -ItemType Directory -Force -Path $tempExt | Out-Null
            
            Expand-Archive -Path $tempZip -DestinationPath $tempExt -Force
            Remove-Item $tempZip
            
            # Move extracted files to our target toolchain/ucrt64 directory
            if (Test-Path $localSysroot) { Remove-Item $localSysroot -Recurse -Force }
            Move-Item (Join-Path $tempExt "mingw64") $localSysroot -Force
            Remove-Item $tempExt -Recurse -Force
            
            Write-Host "  MinGW sysroot extracted to $localSysroot successfully." -ForegroundColor Green
        } catch {
            Write-Host "  Warning: MinGW sysroot download/extraction failed: $_" -ForegroundColor Red
        }
    }
}

# ---------------------------------------------------------------------------
# 4. Check MSVC C++ Build Tools (Universal CRT, libcmt.lib)
# ---------------------------------------------------------------------------
Write-Host "`n[4/4] Checking MSVC C++ Build Tools..." -ForegroundColor Yellow
$hasMsvc = $false
$vsPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC"
)
foreach ($p in $vsPaths) {
    if (Test-Path $p) {
        $subdirs = Get-ChildItem $p -Directory
        if ($subdirs.Count -gt 0) {
            $hasMsvc = $true
            break
        }
    }
}

if ($hasMsvc) {
    Write-Host "  MSVC C++ Build Tools detected." -ForegroundColor Green
} else {
    Write-Host "  MSVC Build Tools not found. Downloading Visual Studio Build Tools bootstrapper..." -ForegroundColor Cyan
    $vsUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
    $tempExe = Join-Path $env:TEMP "vs_buildtools.exe"
    
    try {
        Download-FileWithProgress $vsUrl $tempExe
        Write-Host "  Installing C++ Build Tools (silent, this may take a few minutes)..." -ForegroundColor Cyan
        
        # Run vs_buildtools silently with Desktop C++ Workload
        $proc = Start-Process -FilePath $tempExe -ArgumentList `
            "--quiet", "--wait", "--norestart", "--nocache", `
            "--add", "Microsoft.VisualStudio.Workload.VCTools", `
            "--includeRecommended" -Wait -PassThru -NoNewWindow
            
        Remove-Item $tempExe
        
        if ($proc.ExitCode -eq 0 -or $proc.ExitCode -eq 3010) {
            Write-Host "  MSVC C++ Build Tools installed successfully." -ForegroundColor Green
        } else {
            Write-Host "  Warning: MSVC Build Tools installer exited with code: $($proc.ExitCode)" -ForegroundColor Red
        }
    } catch {
        Write-Host "  Warning: MSVC Build Tools download/installation failed: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  All dependency checks and installations completed.     " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Please restart any open terminal windows to apply PATH changes." -ForegroundColor Yellow
Write-Host ""
