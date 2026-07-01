# BirdieEye - PowerShell one-click launcher
$ErrorActionPreference = "Stop"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BirdieEye - Streamlit Web UI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# [1/5] Python check
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
$pyExe = $null
$py = Get-Command python -ErrorAction SilentlyContinue
if ($py) { $pyExe = "python" }
if (-not $pyExe) {
    Write-Host "ERROR: Python not found (tried 'python' and '$candidate')" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
& $pyExe --version

# [2/5] venv
if (!(Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
    & $pyExe -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create venv" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "[2/5] Virtual environment exists" -ForegroundColor Green
}

# [3/5] activate + install deps
Write-Host "[3/5] Checking dependencies..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -c "import streamlit, torch, ultralytics, cv2, yaml" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Dependencies missing, installing..." -ForegroundColor Yellow
    & .\.venv\Scripts\pip.exe install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
} else {
    Write-Host "  Dependencies OK" -ForegroundColor Green
}

# [4/5] model weights
Write-Host "[4/5] Checking model weights..." -ForegroundColor Yellow
if (!(Test-Path "weights\yolo11s-ball.pt")) {
    Write-Host "  downloading yolo11s-ball.pt..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "weights" -Force | Out-Null
    curl.exe -L -o "weights\yolo11s-ball.pt" "https://github.com/yo-WASSUP/BirdieEye/releases/download/v0.1.0/yolo11s-ball.pt" --retry 3 --retry-delay 2
}
if (!(Test-Path "weights\yolo11n-pose.pt")) {
    Write-Host "  downloading yolo11n-pose.pt..." -ForegroundColor Yellow
    curl.exe -L -o "weights\yolo11n-pose.pt" "https://github.com/yo-WASSUP/BirdieEye/releases/download/v0.1.0/yolo11n-pose.pt" --retry 3 --retry-delay 2
}

# [5/5] launch
Write-Host "[5/5] Launching Streamlit..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Browser: http://localhost:8501" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Start-Process "http://localhost:8501"
& .\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501 --server.maxUploadSize 4096

Read-Host "Press Enter to exit"
