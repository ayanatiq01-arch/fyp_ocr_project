@echo off
rem Stops the HarfScan server from starting automatically at log-in.
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\HarfScan server.bat" 2>nul
echo HarfScan server will no longer start automatically.
