@echo off
rem HarfScan OCR server.
rem Keeps the server running: if it stops or crashes it is started again
rem after 5 seconds. Close this window to stop the server.
rem install_autostart.bat makes Windows run this at every log-in.
title HarfScan OCR server (close this window to stop it)
cd /d "%~dp0"
set PYTHONUTF8=1

for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo   Server address for the app:%%a:8000
echo.

:loop
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
echo.
echo Server stopped - starting it again in 5 seconds (close this window to stop)...
timeout /t 5 /nobreak >nul
goto loop
