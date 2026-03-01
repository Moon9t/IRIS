# build_installer.ps1 — Build the IRIS Windows installer package
# Run from the project root: powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1

param(
    [string]$Version = "0.1.0",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host "IRIS Installer Builder v$Version" -ForegroundColor Cyan
Write-Host "Project root: $Root" -ForegroundColor Gray

# Step 1: Build release binary
if (-not $SkipBuild) {
    Write-Host "`nBuilding release binary..." -ForegroundColor Yellow
    Push-Location $Root
    & cargo build --release
    if ($LASTEXITCODE -ne 0) { Write-Error "cargo build --release failed"; exit 1 }
    Pop-Location
    Write-Host "Build complete." -ForegroundColor Green
}

# Step 2: Prepare installer directory
$InstallerDir = Join-Path $Root "installer"
New-Item -ItemType Directory -Force -Path $InstallerDir | Out-Null

# Step 3: Copy iris.exe
$IrisExe = Join-Path $Root "target\release\iris.exe"
if (-not (Test-Path $IrisExe)) { Write-Error "iris.exe not found at $IrisExe"; exit 1 }
Copy-Item $IrisExe $InstallerDir -Force
Write-Host "Copied iris.exe ($([math]::Round((Get-Item $IrisExe).Length / 1MB, 2)) MB)" -ForegroundColor Green

# Step 4: Copy VSCode extension
$Vsix = Get-ChildItem (Join-Path $Root "vscode-iris\*.vsix") | Select-Object -First 1
if ($Vsix) {
    Copy-Item $Vsix.FullName $InstallerDir -Force
    Write-Host "Copied $($Vsix.Name)" -ForegroundColor Green
} else {
    Write-Host "Warning: No .vsix found. Run 'cd vscode-iris && npm run package' first." -ForegroundColor Yellow
}

# Step 5: Copy icon
$Icon = Join-Path $Root "vscode-iris\icon.png"
if (Test-Path $Icon) {
    Copy-Item $Icon $InstallerDir -Force
}

# Step 6: Create ZIP archive
$ZipName = "IRIS-$Version-windows-x64.zip"
$ZipPath = Join-Path $Root $ZipName
if (Test-Path $ZipPath) { Remove-Item $ZipPath }

$FilesToZip = @(
    (Join-Path $InstallerDir "install.ps1"),
    (Join-Path $InstallerDir "uninstall.ps1"),
    (Join-Path $InstallerDir "iris.exe"),
    (Join-Path $InstallerDir "README.md")
)
if ($Vsix) { $FilesToZip += (Join-Path $InstallerDir $Vsix.Name) }

Write-Host "`nCreating $ZipName..." -ForegroundColor Yellow
Compress-Archive -Path $FilesToZip -DestinationPath $ZipPath -Force
Write-Host "Created: $ZipPath ($([math]::Round((Get-Item $ZipPath).Length / 1MB, 2)) MB)" -ForegroundColor Green

Write-Host "`nInstaller package ready:" -ForegroundColor Cyan
Write-Host "  $ZipPath" -ForegroundColor White
Write-Host "`nTo install: unzip and run 'powershell -ExecutionPolicy Bypass -File install.ps1'" -ForegroundColor Gray
