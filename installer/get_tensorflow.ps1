# Download TensorFlow C library for Windows x64 and extract to installer/tensorflow
# Usage: run from repository root in PowerShell

$dest = "$PSScriptRoot/../_stage/tensorflow"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# Update version as required
$url = "https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-cpu-windows-x86_64-2.12.0.zip"
$zip = "$env:TEMP\tf.zip"

Write-Host "Downloading TensorFlow C library from $url..."
Invoke-WebRequest -Uri $url -OutFile $zip
Write-Host "Extracting to $dest..."
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip
Write-Host "Done. Set environment variable TENSORFLOW_DIR to $dest to enable TensorFlow support."
