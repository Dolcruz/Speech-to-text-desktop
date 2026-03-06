# Requires: Python available. Prefers .venv\Scripts\python.exe when present.
# Usage:  .\build_windows.ps1 [-OneFile]
param(
    [switch]$OneFile
)

$ErrorActionPreference = 'Stop'

Write-Host "Packaging STTDesktop..."

$projectRoot = Get-Location
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

Write-Host "Using Python: $pythonCmd"

& $pythonCmd -m pip show pyinstaller > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller..."
    & $pythonCmd -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller installation failed."
    }
}

if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }

$iconPath = Join-Path $projectRoot "assets\app.ico"
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

& $pythonCmd -m PyInstaller @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$artifactPath = if ($OneFile) { "dist\STTDesktop.exe" } else { "dist\STTDesktop\STTDesktop.exe" }
if (-not (Test-Path $artifactPath)) {
    throw "Build finished without expected artifact at $artifactPath"
}

Write-Host "Build finished. EXE at: $artifactPath"
