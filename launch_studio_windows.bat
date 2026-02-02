@echo off
REM Bark Infinity Studio - Windows 10 Launcher
REM Enterprise-Grade Voice Cloning and Audio Generation

echo ========================================
echo Bark Infinity Studio - Windows 10
echo Enterprise Voice Cloning Platform
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Starting Bark Infinity Studio...
echo.

REM Run the application
python bark_infinity_studio.py %*

if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
)
