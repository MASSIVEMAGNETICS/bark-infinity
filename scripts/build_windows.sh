#!/bin/bash
# Build script for Windows executable using PyInstaller
# This script should be run on Windows or in a Windows environment

set -e

echo "=========================================="
echo "Bark Infinity - Windows Build Script"
echo "=========================================="

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.spec

# Install dependencies
echo "Installing dependencies..."
pip install -e .[build]

# Verify bark_webui.py exists
if [ ! -f "bark_webui.py" ]; then
    echo "Error: bark_webui.py not found in current directory"
    echo "Make sure you are running this script from the bark-infinity root directory"
    exit 1
fi

# Create spec file if it doesn't exist
if [ ! -f "bark-infinity.spec" ]; then
    echo "Generating PyInstaller spec file..."
    pyi-makespec --name bark-infinity-webui \
                 --onefile \
                 --windowed \
                 --add-data "bark_infinity/assets:bark_infinity/assets" \
                 --hidden-import=bark_infinity \
                 bark_webui.py
fi

# Build the executable
echo "Building Windows executable..."
pyinstaller bark-infinity.spec --clean --noconfirm

# Check if build was successful
if [ -f "dist/bark-infinity-webui.exe" ]; then
    echo "=========================================="
    echo "Build successful!"
    echo "Executable location: dist/bark-infinity-webui.exe"
    echo "=========================================="
    
    # Get file size
    ls -lh dist/bark-infinity-webui.exe
else
    echo "Build failed! Executable not found."
    exit 1
fi

echo ""
echo "To distribute:"
echo "1. Copy the dist/bark-infinity-webui.exe file"
echo "2. Users can run it directly without installing Python"
echo "3. First run will download models (~12GB) from Hugging Face"
echo ""
