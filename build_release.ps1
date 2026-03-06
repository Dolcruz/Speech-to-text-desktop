# Build Script fuer Release
# Baut die .exe und bereitet sie fuer GitHub Release vor

$ErrorActionPreference = 'Stop'

Write-Host "Building STTDesktop Release..." -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path "VERSION")) {
    Write-Host "VERSION Datei nicht gefunden!" -ForegroundColor Red
    exit 1
}

$version = (Get-Content "VERSION" -Raw).Trim()
Write-Host "Building Version: $version" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".venv")) {
    Write-Host "Virtual Environment nicht gefunden!" -ForegroundColor Yellow
    Write-Host "Erstelle Virtual Environment..." -ForegroundColor Green
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual Environment konnte nicht erstellt werden."
    }
}

$pythonCmd = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonCmd)) {
    throw "Python im Virtual Environment nicht gefunden: $pythonCmd"
}

Write-Host "Using Python: $pythonCmd" -ForegroundColor Cyan

Write-Host "Installiere Dependencies..." -ForegroundColor Green
& $pythonCmd -m pip install -r requirements.txt --quiet --upgrade
if ($LASTEXITCODE -ne 0) {
    throw "Dependency-Installation fehlgeschlagen."
}

Write-Host "Installiere PyInstaller..." -ForegroundColor Green
& $pythonCmd -m pip install pyinstaller --quiet --upgrade
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller-Installation fehlgeschlagen."
}

Write-Host "Baue STTDesktop.exe (Version $version)..." -ForegroundColor Green
Write-Host ""
.\build_windows.ps1 -OneFile
if ($LASTEXITCODE -ne 0) {
    throw "Build fehlgeschlagen."
}

if (Test-Path "dist\STTDesktop.exe") {
    Write-Host ""
    Write-Host "Build erfolgreich!" -ForegroundColor Green
    Write-Host ""

    Write-Host "Teste ob VERSION inkludiert ist..." -ForegroundColor Cyan
    $fileSize = (Get-Item "dist\STTDesktop.exe").Length / 1MB
    Write-Host "Dateigroesse: $([math]::Round($fileSize, 2)) MB" -ForegroundColor White

    Write-Host ""
    Write-Host "Die fertige .exe befindet sich in:" -ForegroundColor Cyan
    Write-Host "dist\STTDesktop.exe" -ForegroundColor White
    Write-Host ""
    Write-Host "Naechste Schritte:" -ForegroundColor Yellow
    Write-Host "1. Teste die .exe: .\dist\STTDesktop.exe" -ForegroundColor White
    Write-Host "2. Pruefe ob Version $version angezeigt wird" -ForegroundColor White
    Write-Host "3. Wenn alles OK: GitHub Release erstellen" -ForegroundColor White
    Write-Host "https://github.com/Dolcruz/Speech-to-text-desktop/releases/new" -ForegroundColor Cyan
    Write-Host "4. Tag: v$version" -ForegroundColor White
    Write-Host "5. Lade dist\STTDesktop.exe hoch" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Build fehlgeschlagen!" -ForegroundColor Red
    Write-Host "Pruefe die Fehler oben." -ForegroundColor Yellow
    exit 1
}
