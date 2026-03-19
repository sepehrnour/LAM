@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_HOME%\bin;%PATH%
set TORCH_CUDA_ARCH_LIST=12.0
set XFORMERS_DISABLED=1

cd /d C:\Users\Sepehr\Desktop\Dev\lam\LAM
call lam_env\Scripts\activate.bat

python -u app_lam.py
pause
