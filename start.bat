@echo off
chcp 65001 >nul
echo ========================================
echo   Good-Badminton - Streamlit Web UI
echo ========================================
echo.

cd /d "%~dp0"

if not exist ".venv" (
    echo [1/5] 创建虚拟环境...
    "D:\Program\Python.12.10\python.exe" -m venv .venv
    if errorlevel 1 (
        echo 错误: 创建虚拟环境失败
        pause
        exit /b 1
    )
) else (
    echo [1/5] 虚拟环境已存在
)

echo [2/5] 激活虚拟环境...
call .venv\Scripts\activate.bat

echo [3/5] 安装/更新依赖...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

echo [4/5] 检查 PyTorch GPU...
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo [5/5] 启动 Streamlit...
echo 浏览器打开: http://localhost:8501
start "" "http://localhost:8501"
streamlit run app.py --server.headless true --server.port 8501 --server.maxUploadSize 4096

pause
