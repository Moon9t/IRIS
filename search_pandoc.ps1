Write-Host "Searching for pandoc..."
$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if ($pandoc) {
    Write-Host "Found in PATH: $($pandoc.Source)"
    exit 0
}

# Search Registry
$reg = Get-ItemProperty -Path "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*", "HKLM:\Software\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*", "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*" -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -like "*pandoc*" }
if ($reg) {
    Write-Host "Found in Registry: $($reg.InstallLocation) -- $($reg.UninstallString)"
    exit 0
}

# Search standard AppData
$paths = @(
    "$env:LOCALAPPDATA\Programs\Pandoc",
    "$env:LOCALAPPDATA\Pandoc",
    "$env:APPDATA\Pandoc",
    "C:\Program Files\Pandoc",
    "C:\Program Files (x86)\Pandoc"
)
foreach ($p in $paths) {
    if (Test-Path "$p\pandoc.exe") {
        Write-Host "Found in path list: $p\pandoc.exe"
        exit 0
    }
}

Write-Host "Not found in common locations. Searching user profile..."
$profileSearch = Get-ChildItem -Path "$env:USERPROFILE" -Filter pandoc.exe -Recurse -ErrorAction SilentlyContinue -Force
if ($profileSearch) {
    if ($profileSearch -is [array]) {
        Write-Host "Found in profile: $($profileSearch[0].FullName)"
    } else {
        Write-Host "Found in profile: $($profileSearch.FullName)"
    }
    exit 0
}

Write-Host "Searching C:\..."
$cSearch = Get-ChildItem -Path "C:\" -Filter pandoc.exe -Recurse -ErrorAction SilentlyContinue -Force
if ($cSearch) {
    if ($cSearch -is [array]) {
        Write-Host "Found on C: $($cSearch[0].FullName)"
    } else {
        Write-Host "Found on C: $($cSearch.FullName)"
    }
    exit 0
}

Write-Host "Pandoc not found anywhere!"
