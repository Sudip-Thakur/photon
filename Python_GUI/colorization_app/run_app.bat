@echo off
REM AI Colorization Studio - Windows Launch Script
REM This script helps launch the application with proper environment setup

title AI Colorization Studio

echo ========================================
echo   AI Colorization Studio
echo   Professional Image Colorization
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Check if we're in the correct directory
if not exist "main.py" (
    echo ERROR: main.py not found in current directory
    echo Please run this script from the colorization_app directory
    echo.
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

REM Check if virtual environment exists
if exist "venv\" (
    echo Found virtual environment, activating...
    call venv\Scripts\activate.bat
    if %errorlevel% neq 0 (
        echo WARNING: Failed to activate virtual environment
    ) else (
        echo Virtual environment activated
    )
    echo.
) else (
    echo No virtual environment found - using system Python
    echo Tip: Create a virtual environment with: python -m venv venv
    echo.
)

REM Check for required packages
echo Checking dependencies...
python -c "import PyQt5" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PyQt5 not found
    echo Installing required packages...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        echo Please install manually: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
) else (
    echo PyQt5 found
)

python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PyTorch not found
    echo Installing PyTorch...
    python -m pip install torch torchvision
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install PyTorch
        echo Please install manually: pip install torch torchvision
        echo.
        pause
        exit /b 1
    )
) else (
    echo PyTorch found
)

python -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: OpenCV not found
    echo Installing OpenCV...
    python -m pip install opencv-python
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install OpenCV
        echo Please install manually: pip install opencv-python
        echo.
        pause
        exit /b 1
    )
) else (
    echo OpenCV found
)

echo All dependencies satisfied!
echo.

REM Check for GPU support
echo Checking GPU support...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\" (
    echo CUDA toolkit detected
) else (
    echo No CUDA toolkit found - CPU mode will be used
)
echo.

REM Launch application
echo ========================================
echo   Starting AI Colorization Studio...
echo ========================================
echo.
echo Close this window to stop the application
echo Or press Ctrl+C to force quit
echo.

python main.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Application exited with error code: %errorlevel%
    echo ========================================
    echo.
    echo Common solutions:
    echo 1. Check that all dependencies are installed
    echo 2. Ensure you have sufficient system memory
    echo 3. Check that your camera is not in use by other applications
    echo 4. Try running in compatibility mode
    echo.
    echo For support, visit: https://github.com/aicolorization/studio/issues
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo Application closed successfully
    echo ========================================
    echo.
)

REM Deactivate virtual environment if it was activated
if defined VIRTUAL_ENV (
    echo Deactivating virtual environment...
    deactivate
)

echo.
echo Thank you for using AI Colorization Studio!
echo.
pause
