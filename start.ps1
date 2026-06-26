# Good-Badminton - PowerShell one-click launcher
$ErrorActionPreference = "Stop"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Good-Badminton - Streamlit Web UI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# [1/5] Python check
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
$pyExe = $null
$py = Get-Command python -ErrorAction SilentlyContinue
if ($py) { $pyExe = "python" }
else {
    $candidate = "D:\Program\Python.12.10\python.exe"
    if (Test-Path $candidate) { $pyExe = $candidate }
}
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
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
$idx = "https://pypi.tuna.tsinghua.edu.cn/simple"
python -m pip install --upgrade pip -i $idx --trusted-host pypi.tuna.tsinghua.edu.cn
pip install -r requirements.txt -i $idx --trusted-host pypi.tuna.tsinghua.edu.cn

# [4/5] model weights
Write-Host "[4/5] Checking model weights..." -ForegroundColor Yellow
if (!(Test-Path "weights\yolo11s-ball.pt")) {
    Write-Host "  downloading yolo11s-ball.pt..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "weights" -Force | Out-Null
    curl.exe -L -o "weights\yolo11s-ball.pt" "https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11s-ball.pt" --retry 3 --retry-delay 2
}
if (!(Test-Path "weights\yolo11n-pose.pt")) {
    Write-Host "  downloading yolo11n-pose.pt..." -ForegroundColor Yellow
    curl.exe -L -o "weights\yolo11n-pose.pt" "https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11n-pose.pt" --retry 3 --retry-delay 2
}

# [5/5] launch
Write-Host "[5/5] Launching Streamlit..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Browser: http://localhost:8501" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Start-Process "http://localhost:8501"
streamlit run app.py --server.headless true --server.port 8501 --server.maxUploadSize 4096

Read-Host "Press Enter to exit"
