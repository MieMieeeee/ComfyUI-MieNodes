@echo off
setlocal
cd /d "%~dp0.."
python "%~dp0sync_to_comfyui.py" %*
exit /b %ERRORLEVEL%
