# build_msi.ps1 — Build a Windows .msi installer for IRIS using WiX Toolset
# ──────────────────────────────────────────────────────────────────────────
# Usage:
#   powershell -ExecutionPolicy Bypass -File installer\windows\build_msi.ps1
#
# Produces: installer\dist\IRIS-<version>-windows-x64.msi
# Requires: WiX Toolset v4+ (dotnet tool install --global wix)
# ──────────────────────────────────────────────────────────────────────────

param(
    [string]$Version = "1.0.0-rc1",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $Root -or -not (Test-Path $Root)) {
    $Root = Split-Path $PSScriptRoot -Parent
    $Root = Split-Path $Root -Parent
}

Write-Host "IRIS MSI Builder v${Version}" -ForegroundColor Cyan
Write-Host "Project root: $Root" -ForegroundColor Gray

# ── Build ─────────────────────────────────────────────────────────────────
if (-not $SkipBuild) {
    Write-Host "`n[1/5] Building release binary..." -ForegroundColor Yellow
    Push-Location $Root
    & cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build --release failed" }
    Pop-Location
    Write-Host "  Build complete." -ForegroundColor Green
} else {
    Write-Host "`n[1/5] Skipped build." -ForegroundColor DarkGray
}

# ── Stage ─────────────────────────────────────────────────────────────────
Write-Host "[2/5] Staging files..." -ForegroundColor Yellow

$StageDir = Join-Path $Root "installer\dist\msi-stage"
if (Test-Path $StageDir) { Remove-Item $StageDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

$DistDir = Join-Path $Root "installer\dist"
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

$IrisExe = Join-Path $Root "target\release\iris.exe"
if (-not (Test-Path $IrisExe)) { throw "iris.exe not found at $IrisExe" }

Copy-Item $IrisExe $StageDir -Force
Copy-Item (Join-Path $Root "LICENSE") $StageDir -Force
Copy-Item (Join-Path $Root "README.md") $StageDir -Force

# Stdlib
$StdlibSrc = Join-Path $Root "stdlib"
if (Test-Path $StdlibSrc) {
    New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "stdlib") | Out-Null
    Copy-Item "$StdlibSrc\*" (Join-Path $StageDir "stdlib") -Recurse -Force
}

# Examples
$ExamplesSrc = Join-Path $Root "examples"
if (Test-Path $ExamplesSrc) {
    New-Item -ItemType Directory -Force -Path (Join-Path $StageDir "examples") | Out-Null
    Copy-Item "$ExamplesSrc\*" (Join-Path $StageDir "examples") -Recurse -Force
}

Write-Host "  Files staged." -ForegroundColor Green

# ── Generate WiX source ──────────────────────────────────────────────────
Write-Host "[3/5] Generating WiX source..." -ForegroundColor Yellow

$WxsPath = Join-Path $StageDir "iris.wxs"
$MsiFile = "IRIS-${Version}-windows-x64.msi"

@"
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">
  <Package
    Name="IRIS Language"
    Version="${Version}.0"
    Manufacturer="IRIS Language Project"
    UpgradeCode="A7B3C2D1-E4F5-4A6B-8C9D-0E1F2A3B4C5E"
    Scope="perUser">

    <MajorUpgrade
      DowngradeErrorMessage="A newer version of IRIS is already installed." />

    <MediaTemplate EmbedCab="yes" CompressionLevel="high" />

    <StandardDirectory Id="LocalAppDataFolder">
      <Directory Id="PROGRAMSDIR" Name="Programs">
        <Directory Id="INSTALLDIR" Name="IRIS">

          <Component Id="IrisExe" Guid="B1C2D3E4-F5A6-4B7C-8D9E-0F1A2B3C4D5E">
            <File Id="iris.exe" Source="iris.exe" KeyPath="yes" />
          </Component>

          <Component Id="License" Guid="C2D3E4F5-A6B7-4C8D-9E0F-1A2B3C4D5E6F">
            <File Id="LICENSE" Source="LICENSE" />
          </Component>

          <Component Id="Readme" Guid="D3E4F5A6-B7C8-4D9E-0F1A-2B3C4D5E6F7A">
            <File Id="README.md" Source="README.md" />
          </Component>

          <Component Id="PathEnv" Guid="E4F5A6B7-C8D9-4E0F-1A2B-3C4D5E6F7A8B">
            <Environment Id="PATH" Name="PATH" Value="[INSTALLDIR]"
                         Permanent="no" Part="last" Action="set" System="no" />
          </Component>

          <Directory Id="StdlibDir" Name="stdlib">
            <Component Id="StdlibFiles" Guid="F5A6B7C8-D9E0-4F1A-2B3C-4D5E6F7A8B9C">
              <File Id="io.iris" Source="stdlib\io.iris" />
              <File Id="file.iris" Source="stdlib\file.iris" />
            </Component>
          </Directory>
        </Directory>
      </Directory>
    </StandardDirectory>

    <Feature Id="Main" Title="IRIS Compiler" Level="1">
      <ComponentRef Id="IrisExe" />
      <ComponentRef Id="License" />
      <ComponentRef Id="Readme" />
      <ComponentRef Id="PathEnv" />
      <ComponentRef Id="StdlibFiles" />
    </Feature>

  </Package>
</Wix>
"@ | Set-Content -Path $WxsPath -Encoding UTF8

Write-Host "  WiX source generated." -ForegroundColor Green

# ── Build MSI ─────────────────────────────────────────────────────────────
Write-Host "[4/5] Building MSI..." -ForegroundColor Yellow

$wixCmd = Get-Command wix -ErrorAction SilentlyContinue
if (-not $wixCmd) {
    Write-Host "  WiX Toolset not found. Attempting to install..." -ForegroundColor Yellow
    & dotnet tool install --global wix 2>$null
    $wixCmd = Get-Command wix -ErrorAction SilentlyContinue
}

if ($wixCmd) {
    Push-Location $StageDir
    & wix build iris.wxs -o (Join-Path $DistDir $MsiFile)
    if ($LASTEXITCODE -ne 0) { throw "WiX build failed" }
    Pop-Location
    Write-Host "  MSI built successfully." -ForegroundColor Green
} else {
    Write-Host "  WiX not available. Skipping MSI build." -ForegroundColor Yellow
    Write-Host "  Install WiX: dotnet tool install --global wix" -ForegroundColor Gray
    Write-Host "  Or use the Inno Setup installer: build_installer.ps1" -ForegroundColor Gray
}

# ── Cleanup ───────────────────────────────────────────────────────────────
Write-Host "[5/5] Cleaning up..." -ForegroundColor Yellow
Remove-Item $StageDir -Recurse -Force

$MsiPath = Join-Path $DistDir $MsiFile
if (Test-Path $MsiPath) {
    $msiSizeMB = [math]::Round((Get-Item $MsiPath).Length / 1048576, 1)
    Write-Host "`nMSI ready:" -ForegroundColor Cyan
    Write-Host "  $MsiPath ($msiSizeMB MB)" -ForegroundColor White
    Write-Host ""
    Write-Host "  Install with:" -ForegroundColor Gray
    Write-Host "    msiexec /i $MsiFile" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "`nMSI output not found." -ForegroundColor Yellow
}
