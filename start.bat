@echo off
chcp 65001 >nul
title Good-Badminton - Streamlit Web UI

echo ========================================
echo   Good-Badminton - Start
echo ========================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [1/4] Creating virtual environment...
    "D:\Program\Python.12.10\python.exe" -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment exists
)

echo [2/4] Activating venv...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv
    pause
    exit /b 1
)

echo [3/4] Checking dependencies...
python -c "import streamlit, torch, ultralytics, cv2, yaml" 2>nul
if errorlevel 1 (
    echo Dependencies missing, installing...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    if errorlevel 1 (
        echo ERROR: Dependency install failed
        pause
        exit /b 1
    )
) else (
    echo Dependencies OK, skip install
)

echo [4/4] Checking model weights...
if not exist "weights\yolo11s-ball.pt" (
    echo Downloading ball model from GitHub Releases...
    curl.exe -L -o "weights\yolo11s-ball.pt" "https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11s-ball.pt" --retry 3 --retry-delay 2
)
if not exist "weights\yolo11n-pose.pt" (
    echo Downloading pose model from GitHub Releases...
    curl.exe -L -o "weights\yolo11n-pose.pt" "https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11n-pose.pt" --retry 3 --retry-delay 2
)
if not exist "weights\yolo11n-pose.pt" (
    echo Pose model not found, falling back to yolo11n-pose.pt (will auto-download on first run)
)

echo.
echo ========================================
echo   Starting Streamlit...
echo   Browser: http://localhost:8501
echo ========================================
echo.

timeout /t 3 /nobreak >nul
start "" "http://localhost:8501"
python -m streamlit run app.py --server.headless true --server.port 8501 --server.maxUploadSize 4096

echo.
echo App stopped
pause
