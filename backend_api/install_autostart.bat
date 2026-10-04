@echo off
rem Starts the HarfScan server automatically at every Windows log-in
rem (a small launcher in your Startup folder; no admin rights needed).
rem Undo: run uninstall_autostart.bat, or delete "HarfScan server.bat" from
rem   %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
(
  echo @echo off
  echo start "HarfScan OCR server" /min "%~dp0start_server.bat"
) > "%STARTUP%\HarfScan server.bat"
echo HarfScan server will now start automatically when you log in to Windows.
echo Launcher: "%STARTUP%\HarfScan server.bat"
