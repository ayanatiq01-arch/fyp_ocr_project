@echo off
rem Creates venv\ and installs the backend. Run from backend_api\.
rem Requires Python 3.11 (py launcher) and an internet connection.
setlocal
cd /d "%~dp0"

if not exist venv (
  echo Creating virtual environment...
  py -3.11 -m venv venv || goto :error
)

venv\Scripts\python -m pip install --upgrade pip || goto :error
venv\Scripts\python -m pip install -r requirements.txt || goto :error

rem opencv-python and opencv-contrib-python share the cv2 package; reinstall
rem contrib last so its (superset) files are the ones on disk.
venv\Scripts\python -m pip install --no-deps --force-reinstall opencv-contrib-python==4.10.0.84 || goto :error

rem Kraken (Arabic recognition) is installed separately: its click<8.3 pin
rem conflicts with huggingface-hub's in a single resolve (see requirements.txt).
venv\Scripts\python -m pip install kraken==7.1.1 || goto :error

rem Kraken model: "Printed Arabic Base Model Trained on the OpenITI Corpus"
rem (DOI 10.5281/zenodo.7050296, CC0 licence).
if not exist "kraken_models\arabic_best.mlmodel" (
  echo Downloading the Kraken Arabic model...
  if not exist kraken_models mkdir kraken_models
  curl -L -o "kraken_models\arabic_best.mlmodel" "https://zenodo.org/records/7050296/files/arabic_best.mlmodel?download=1" || goto :error
)

if not exist "UTRNet-High-Resolution-Urdu-Text-Recognition\saved_models\UTRNet-Large\best_norm_ED.pth" (
  echo.
  echo WARNING: UTRNet weights missing. Download UTRNet-Large from the link in
  echo UTRNet-High-Resolution-Urdu-Text-Recognition\README.md into
  echo UTRNet-High-Resolution-Urdu-Text-Recognition\saved_models\UTRNet-Large\best_norm_ED.pth
)

echo.
echo Setup complete. Start the server with:  venv\Scripts\python main.py
exit /b 0

:error
echo Setup failed.
exit /b 1
