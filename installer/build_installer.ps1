# build_installer.ps1 - Build the IRIS Windows EXE installer (Inno Setup)
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1
#
# Produces: installer\dist\IRIS-<version>-windows-x64-setup.exe
#
# The script:
#   1. Builds the release binary (cargo build --release)
#   2. Stages all files into installer\_stage
#   3. Invokes ISCC (Inno Setup Compiler) to produce the single-EXE installer
#
# Bundled dependencies (no GCC -- clang + lld only):
#   - clang.exe + ld.lld.exe  (LLVM 17)
#   - MinGW ucrt64 sysroot    (C headers + static libs)

param(
    [string]$Version = "0.2.0",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host "IRIS Installer Builder v$Version" -ForegroundColor Cyan
Write-Host "Project root: $Root" -ForegroundColor Gray

# ---------------------------------------------------------------------------
# Step 1: Build release binary
# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "`n[1/9] Building release binary..." -ForegroundColor Yellow
    Push-Location $Root
    & cargo build --release
    if ($LASTEXITCODE -ne 0) { Write-Error "cargo build --release failed"; throw "Build failed" }
    Pop-Location
    Write-Host "  Build complete." -ForegroundColor Green
} else {
    Write-Host "`n[1/9] Skipped build (--SkipBuild)." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# Step 2: Prepare staging directory
# ---------------------------------------------------------------------------
Write-Host "[2/9] Preparing staging directory..." -ForegroundColor Yellow
$StageDir = Join-Path $Root "installer\_stage"
if (Test-Path $StageDir) { Remove-Item $StageDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

$InstallerDir = Join-Path $Root "installer"

# ---------------------------------------------------------------------------
# Step 3: Stage iris.exe
# ---------------------------------------------------------------------------
Write-Host "[3/9] Staging iris.exe..." -ForegroundColor Yellow
$IrisExe = Join-Path $Root "target\release\iris.exe"
if (-not (Test-Path $IrisExe)) { Write-Error "iris.exe not found at $IrisExe"; throw "iris.exe not found" }
Copy-Item $IrisExe $StageDir -Force
$irisSizeMB = [math]::Round((Get-Item $IrisExe).Length / 1048576, 1)
Write-Host "  iris.exe ($irisSizeMB MB)" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 4: Stage README + icon
# ---------------------------------------------------------------------------
Write-Host "[4/9] Staging docs and icon..." -ForegroundColor Yellow
Copy-Item (Join-Path $InstallerDir "README.md") $StageDir -Force
$Icon = Join-Path $Root "vscode-iris\icon.png"
if (Test-Path $Icon) { Copy-Item $Icon $StageDir -Force }
Write-Host "  README.md + icon.png" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 5: Stage VSCode extension
# ---------------------------------------------------------------------------
Write-Host "[5/9] Staging VSCode extension..." -ForegroundColor Yellow
$Vsix = Get-ChildItem (Join-Path $Root "vscode-iris\*.vsix") -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
if ($Vsix) {
    Copy-Item $Vsix.FullName $StageDir -Force
    Write-Host "  $($Vsix.Name)" -ForegroundColor Green
} else {
    Write-Host "  Warning: No .vsix found. Run 'cd vscode-iris && npm run package' first." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Step 6: Stage LLVM (clang + lld)
# ---------------------------------------------------------------------------
Write-Host "[6/9] Staging LLVM/clang + lld..." -ForegroundColor Yellow
$LlvmDst = New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "toolchain\llvm\bin")
$LlvmBinSrc = "C:\Program Files\LLVM\bin"
$llvmTotal = 0
foreach ($f in @('clang.exe', 'ld.lld.exe')) {
    $src = Join-Path $LlvmBinSrc $f
    if (Test-Path $src) {
        Copy-Item $src $LlvmDst.FullName -Force
        $sz = (Get-Item $src).Length
        $llvmTotal += $sz
        $szMB = [math]::Round($sz / 1048576, 1)
        Write-Host "  $f ($szMB MB)" -ForegroundColor Green
    } else {
        Write-Error "$f not found at $src"
        throw "$f not found"
    }
}
$llvmMB = [math]::Round($llvmTotal / 1048576, 1)
Write-Host "  LLVM total: $llvmMB MB" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 7: Stage MinGW sysroot (headers + static libs, NO executables)
# ---------------------------------------------------------------------------
Write-Host "[7/9] Staging MinGW sysroot..." -ForegroundColor Yellow
$Msys2Ucrt = "C:\msys64\ucrt64"
if (-not (Test-Path $Msys2Ucrt)) { Write-Error "MSYS2 ucrt64 not found at $Msys2Ucrt"; throw "MSYS2 ucrt64 not found" }

# 7a. GCC internal CRT support files (crtbegin.o, libgcc.a, etc.)
$GccVer     = "14.2.0"
$GccTriple  = "x86_64-w64-mingw32"
$GccLibSrc  = "$Msys2Ucrt\lib\gcc\$GccTriple\$GccVer"
$GccLibDst  = New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "toolchain\ucrt64\lib\gcc\$GccTriple\$GccVer")
$gccLibTotal = 0
Get-ChildItem $GccLibSrc -File | Where-Object { $_.Extension -in '.o','.a' } | ForEach-Object {
    Copy-Item $_.FullName $GccLibDst.FullName -Force
    $gccLibTotal += $_.Length
}
$gccLibMB = [math]::Round($gccLibTotal / 1048576, 1)
Write-Host "  GCC CRT libs: $gccLibMB MB" -ForegroundColor Green

# 7b. ucrt64\lib -- MinGW CRT libs (lib*.a, crt*.o)
$TcLib = New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "toolchain\ucrt64\lib")
$libTotal = 0
$EssentialLibs = @(
    'libmingw32.a', 'libmingwex.a', 'libmsvcrt.a', 'libucrt.a',
    'libpthread.a', 'libm.a', 'libkernel32.a', 'libuser32.a',
    'libadvapi32.a', 'libshell32.a', 'libws2_32.a', 'libole32.a',
    'liboleaut32.a', 'libuuid.a', 'libmoldname.a',
    'crt2.o', 'crtbegin.o', 'crtend.o', 'gcrt2.o', 'dllcrt2.o',
    'libntdll.a', 'libshlwapi.a', 'libversion.a',
    'libgdi32.a', 'libbcrypt.a', 'libuserenv.a', 'libiphlpapi.a'
)
foreach ($f in $EssentialLibs) {
    $src = Join-Path "$Msys2Ucrt\lib" $f
    if (Test-Path $src) {
        Copy-Item $src $TcLib.FullName -Force
        $libTotal += (Get-Item $src).Length
    }
}
$libMB = [math]::Round($libTotal / 1048576, 1)
Write-Host "  ucrt64\lib: $libMB MB" -ForegroundColor Green

# 7c. ucrt64\include -- C headers
$TcInc = New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "toolchain\ucrt64\include")
$incTotal = 0
Get-ChildItem "$Msys2Ucrt\include" -File -Filter '*.h' | ForEach-Object {
    Copy-Item $_.FullName $TcInc.FullName -Force
    $incTotal += $_.Length
}
@('sys', 'sec_api') | ForEach-Object {
    $subSrc = Join-Path "$Msys2Ucrt\include" $_
    if (Test-Path $subSrc) {
        $subDst = Join-Path $TcInc.FullName $_
        Copy-Item $subSrc $subDst -Recurse -Force
        Get-ChildItem $subSrc -Recurse -File | ForEach-Object { $incTotal += $_.Length }
    }
}
$incMB = [math]::Round($incTotal / 1048576, 1)
Write-Host "  ucrt64\include: $incMB MB" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Step 8: Report staging totals
# ---------------------------------------------------------------------------
Write-Host "[8/9] Staging complete." -ForegroundColor Yellow
$totalSize = 0
Get-ChildItem $StageDir -Recurse -File | ForEach-Object { $totalSize += $_.Length }
$fileCount = (Get-ChildItem $StageDir -Recurse -File).Count
$totalMB = [math]::Round($totalSize / 1048576, 1)
Write-Host "  $fileCount files, $totalMB MB uncompressed" -ForegroundColor Gray

# ---------------------------------------------------------------------------
# Step 9: Compile with Inno Setup
# ---------------------------------------------------------------------------
Write-Host "[9/9] Compiling installer with Inno Setup..." -ForegroundColor Yellow

$IsccPaths = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)
$Iscc = $null
foreach ($p in $IsccPaths) {
    if (Test-Path $p) { $Iscc = $p; break }
}
if (-not $Iscc) {
    Write-Error "ISCC.exe not found. Install Inno Setup 6 from https://jrsoftware.org/isdl.php"
    throw "ISCC.exe not found"
}

# Ensure output directory exists
$DistDir = Join-Path $InstallerDir "dist"
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

$IssFile = Join-Path $InstallerDir "iris.iss"
& $Iscc $IssFile
if ($LASTEXITCODE -ne 0) { Write-Error "Inno Setup compilation failed"; throw "Inno Setup compilation failed" }

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
$ExeName = "IRIS-$Version-windows-x64-setup.exe"
$ExePath = Join-Path $DistDir $ExeName
if (Test-Path $ExePath) {
    $exeSizeMB = [math]::Round((Get-Item $ExePath).Length / 1048576, 1)
    Write-Host "`nInstaller ready:" -ForegroundColor Cyan
    Write-Host "  $ExePath ($exeSizeMB MB)" -ForegroundColor White
} else {
    Write-Host "`nWarning: Expected output not found at $ExePath" -ForegroundColor Yellow
    Write-Host "  Check installer\dist\ for the generated .exe" -ForegroundColor Yellow
}

# Cleanup staging
Remove-Item $StageDir -Recurse -Force
Write-Host "Staging directory cleaned up." -ForegroundColor DarkGray
Write-Host ""
