@echo off
chcp 65001 >nul
setlocal ENABLEDELAYEDEXPANSION

REM ============================================================
REM  Apple Motion Lyrics — one-click launcher for Windows
REM  Double-click this file to install dependencies and open the UI.
REM ============================================================

cd /d "%~dp0"
set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%.venv"
set "PORT=4770"
set "HOST=127.0.0.1"
set "URL=http://%HOST%:%PORT%/static/index.html"

echo.
echo ============================================================
echo   Apple Motion Lyrics
echo ============================================================
echo.

REM ----- 1. Find Python -----
set "PYTHON="
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON=%%i"
)
if not defined PYTHON (
    where python >nul 2>&1
    if %ERRORLEVEL%==0 (
        for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON=%%i"
    )
)
if not defined PYTHON (
    echo [X] Python 3.10+ not found. Install it from https://www.python.org/downloads/
    echo     Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [i] Using Python: %PYTHON%
"%PYTHON%" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" || (
    echo [X] Python 3.10 or newer is required.
    pause
    exit /b 1
)

REM ----- 2. Create virtualenv if missing -----
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [i] Creating virtual environment in %VENV_DIR% ...
    "%PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [X] Failed to create virtualenv.
        pause
        exit /b 1
    )
)

set "PIP=%VENV_DIR%\Scripts\python.exe -m pip"

REM ----- 3. Upgrade pip -----
echo [i] Upgrading pip ...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet

REM ----- 4. Install base app deps -----
echo [i] Installing FastAPI, uvicorn, httpx, python-multipart ...
%PIP% install --quiet ^
    "fastapi>=0.110" ^
    "uvicorn[standard]>=0.27" ^
    "httpx[http2]>=0.27" ^
    "python-multipart>=0.0.9"

REM ----- 5. Install faster-whisper -----
echo [i] Installing faster-whisper (this may take a while on first run) ...
%PIP% install --quiet "faster-whisper>=1.0.0"

REM ----- 6. Install arena-api from the sibling folder with [auto] extras -----
set "ARENA_DIR=%APP_DIR%..\arenaaispofandpars"
if exist "%ARENA_DIR%\pyproject.toml" (
    echo [i] Installing arena-api from %ARENA_DIR% ...
    %PIP% install --quiet -e "%ARENA_DIR%[auto]"
    echo [i] Installing Playwright Chromium for invisible browser ...
    "%VENV_DIR%\Scripts\python.exe" -m playwright install chromium
) else (
    echo [!] arenaaispofandpars/ not found next to my-lyrics-app/.
    echo     Arena calls will be unavailable until you put it there and re-run.
)

REM ----- 7. Make sure uploads/ and state/ exist -----
if not exist "%APP_DIR%static\uploads" mkdir "%APP_DIR%static\uploads"
if not exist "%APP_DIR%static\state"   mkdir "%APP_DIR%static\state"

REM ----- 8. Start the server in this window and open the browser -----
echo.
echo ============================================================
echo   Starting server at %URL%
echo   (keep this window open; press Ctrl+C to stop)
echo ============================================================
echo.

REM Wait a couple seconds then open the browser
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start "" "%URL%""

REM Run uvicorn (foreground)
set "HOST=%HOST%"
set "PORT=%PORT%"
set "RELOAD=0"
"%VENV_DIR%\Scripts\python.exe" server.py

echo.
echo Server stopped.
pause
