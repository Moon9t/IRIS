# Download LibTorch (prebuilt) for Windows x64 and extract to installer/libtorch
# Usage: run from repository root in PowerShell

$dest = "$PSScriptRoot/../_stage/libtorch"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# Update version as required
$url = "https://download.pytorch.org/libtorch/cu117/libtorch-win-shared-with-deps-2.1.0%2Bcu117.zip"
$zip = "$env:TEMP\libtorch.zip"

Write-Host "Downloading LibTorch from $url..."
Invoke-WebRequest -Uri $url -OutFile $zip
Write-Host "Extracting to $dest..."
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip
Write-Host "Done. Set environment variable LIBTORCH_DIR to $dest to enable libtorch support."
