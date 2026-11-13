@echo off
REM ============================================================
REM  MedRAG QA - start the GraphRAG + BGE-M3 medical Q&A website
REM
REM  Normal use: double-click run_web.vbs (no console window).
REM  Double-click THIS file only to see the server log for debugging;
REM  close the window (or press Ctrl+C) to stop the server.
REM
REM  run_web.vbs calls this file with --hidden: no browser launch
REM  (the .vbs opens it), no pause, output goes to 02_code\web_app.log.
REM ============================================================
setlocal

REM ---- settings (edit here if needed) ----
set "CONDA_ENV=tf_cpe"
set "PORT=5000"

set "ROOT=%~dp0"
set "HIDDEN="
if /i "%~1"=="--hidden" set "HIDDEN=1"
title MedRAG QA - http://127.0.0.1:%PORT%
cd /d "%ROOT%"

echo.
echo  ===============================================
echo    MedRAG QA  -  GraphRAG + BGE-M3 Medical QA
echo  ===============================================
echo.

REM ---- activate the conda environment (if Anaconda/Miniconda is installed) ----
set "ACTIVATE="
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
    if not defined ACTIVATE if exist "%%~D\Scripts\activate.bat" set "ACTIVATE=%%~D\Scripts\activate.bat"
)
if defined ACTIVATE (
    echo [1/3] Activating conda environment "%CONDA_ENV%" ...
    call "%ACTIVATE%" %CONDA_ENV%
    if errorlevel 1 (
        echo [!] Could not activate "%CONDA_ENV%". Falling back to the default Python.
    )
) else (
    echo [1/3] Conda not found - using the Python on PATH.
)

where python >nul 2>nul
if errorlevel 1 (
    echo [X] Python was not found. Install Python/Anaconda first.
    if defined HIDDEN (
        echo [X] Python was not found. Install Python/Anaconda first.> "%ROOT%02_code\web_app.log"
        exit /b 1
    )
    pause
    exit /b 1
)

REM ---- make sure the web dependency is installed ----
echo [2/3] Checking dependencies ...
python -c "import flask" >nul 2>nul
if errorlevel 1 (
    echo      Installing Flask ...
    python -m pip install flask
)
python -c "import faiss, sentence_transformers, networkx, pandas" >nul 2>nul
if errorlevel 1 (
    echo      Installing project requirements ...
    python -m pip install -r "%ROOT%02_code\requirements.txt"
)

if defined HIDDEN (
    python -u "%ROOT%02_code\web_app.py" --port %PORT% > "%ROOT%02_code\web_app.log" 2>&1
    exit /b
)

REM ---- open the browser shortly after the server starts ----
echo [3/3] Starting server on http://127.0.0.1:%PORT%
echo      The page shows a loading status while the model and index load
echo      (about 20-60 seconds; the first-ever run builds the index and takes ~20 minutes).
echo.
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start "" http://127.0.0.1:%PORT%"

python "%ROOT%02_code\web_app.py" --port %PORT%

echo.
echo Server stopped.
pause
endlocal
