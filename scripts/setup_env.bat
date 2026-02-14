@echo off
setlocal

echo Checking for NVIDIA GPU...

where nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected!
    echo Installing PyTorch 2.5.1 with CUDA 12.1 support...
    uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
) else (
    echo NVIDIA GPU not detected.
    echo Installing standard PyTorch 2.5.1 (CPU version)...
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
)

echo.
echo Installation complete.
echo converting to verify installation...
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

pause
