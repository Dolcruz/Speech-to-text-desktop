# Requires: Python venv activated and pyinstaller installed
# Usage:  .\build_windows.ps1
param(
    [switch]$OneFile
)

Write-Host "Packaging STTDesktop..."

# Ensure pyinstaller is installed
pip show pyinstaller > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller..."
    pip install pyinstaller
}

# Clean previous dist/build
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }

# Build windowed; default to onedir for faster startup.
$iconPath = Join-Path (Get-Location) "assets\app.ico"
$bundleArg = if ($OneFile) { "--onefile" } else { "--onedir" }

$pyInstallerArgs = @(
    "--noconfirm",
    "--noconsole",
    $bundleArg,
    "--name", "STTDesktop",
    "--add-data", "VERSION;.",
    "main.py"
)

if (Test-Path $iconPath) {
    $pyInstallerArgs += @("--icon", $iconPath)
}

pyinstaller @pyInstallerArgs

if ($OneFile) {
    Write-Host "Build finished. EXE at: dist\\STTDesktop.exe"
} else {
    Write-Host "Build finished. EXE at: dist\\STTDesktop\\STTDesktop.exe"
}
