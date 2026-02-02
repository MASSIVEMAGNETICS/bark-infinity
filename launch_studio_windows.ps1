# Bark Infinity Studio - Windows 10 PowerShell Launcher
# Enterprise-Grade Voice Cloning and Audio Generation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bark Infinity Studio - Windows 10" -ForegroundColor Cyan
Write-Host "Enterprise Voice Cloning Platform" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://www.python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting Bark Infinity Studio..." -ForegroundColor Green
Write-Host ""

# Run the application
python bark_infinity_studio.py $args

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "An error occurred." -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
