# Download ONNX Runtime for Windows (x64) and extract to installer/onnxruntime
# Usage: run from repository root in PowerShell

$dest = "$PSScriptRoot/../_stage/onnxruntime"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# URL may change per version; user should update as needed.
$url = "https://github.com/microsoft/onnxruntime/releases/download/v1.16.1/onnxruntime-win-x64-1.16.1.zip"
$zip = "$env:TEMP\onnxruntime.zip"

Write-Host "Downloading ONNX Runtime from $url..."
Invoke-WebRequest -Uri $url -OutFile $zip
Write-Host "Extracting to $dest..."
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip
Write-Host "Done. Set environment variable ONNXRUNTIME_DIR to $dest to enable ONNX support."
