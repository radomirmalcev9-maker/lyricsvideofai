@echo off
setlocal ENABLEDELAYEDEXPANSION
REM ============================================================
REM  Apple Motion Lyrics - one-click launcher for Windows
REM  Double-click this file to install deps and open the UI.
REM  Keep this file ASCII-only; cmd.exe chokes on UTF-8 BOM/emoji.
REM ============================================================

cd /d "%~dp0"
set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%.venv"
set "PORT=4770"
set "HOST=127.0.0.1"
set "URL=http://%HOST%:%PORT%/static/index.html"
set "PYTHON="

echo.
echo ============================================================
echo   Apple Motion Lyrics
echo ============================================================
echo.

REM ----- 1. Find Python (skip Windows Store stubs) -----
REM The "py" launcher is most reliable; fall back to PATH python.
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "CAND=%%i"
    if defined CAND call :check_python "!CAND!" && set "PYTHON=!CAND!"
)
if not defined PYTHON (
    where python >nul 2>&1
    if %ERRORLEVEL%==0 (
        for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "CAND=%%i"
        if defined CAND call :check_python "!CAND!" && set "PYTHON=!CAND!"
    )
)
set "CAND="
if not defined PYTHON (
    echo [X] Python 3.10+ not found (or only the Windows Store stub is present).
    echo.
    echo     Install real Python from  https://www.python.org/downloads/
    echo     and TICK "Add Python to PATH" during install.
    echo     Then re-run this file.
    echo.
    pause
    exit /b 1
)
echo [i] Using Python: %PYTHON%

REM ----- 2. Create virtualenv if missing -----
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [i] Creating virtual environment in %VENV_DIR% ...
    "%PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [X] Failed to create virtualenv. Try running this from Command Prompt,
        echo     or delete .venv\ and try again.
        pause
        exit /b 1
    )
)
set "VPY=%VENV_DIR%\Scripts\python.exe"
set "VPIP=%VPY% -m pip"

REM ----- 3. Upgrade pip -----
echo [i] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip --quiet

REM ----- 4. Install base app deps -----
echo [i] Installing FastAPI, uvicorn, httpx, python-multipart ...
"%VPIP%" install --quiet "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx[http2]>=0.27" "python-multipart>=0.0.9"
if errorlevel 1 (
    echo [X] Failed to install base dependencies. Check your network.
    pause
    exit /b 1
)

REM ----- 5. Install faster-whisper -----
echo [i] Installing faster-whisper (first run may download ~500 MB model) ...
"%VPIP%" install --quiet "faster-whisper>=1.0.0"
if errorlevel 1 (
    echo [X] faster-whisper install failed.
    pause
    exit /b 1
)

REM ----- 6. Install arena-api from the sibling folder with [auto] extras -----
set "ARENA_DIR=%APP_DIR%..\arenaaispofandpars"
if exist "%ARENA_DIR%\pyproject.toml" (
    echo [i] Installing arena-api from %ARENA_DIR% ...
    "%VPIP%" install --quiet -e "%ARENA_DIR%[auto]"
    if errorlevel 1 (
        echo [!] arena-api [auto] install failed; trying without optional browser deps ...
        "%VPIP%" install --quiet -e "%ARENA_DIR%"
    )
    echo [i] Installing Playwright Chromium for invisible browser ...
    "%VPY%" -m playwright install chromium
) else (
    echo [!] arenaaispofandpars/ not found next to my-lyrics-app/.
    echo     Arena calls will be unavailable until you put that folder there.
)

REM ----- 7. Make sure static subfolders exist -----
if not exist "%APP_DIR%static\uploads" mkdir "%APP_DIR%static\uploads"
if not exist "%APP_DIR%static\state"   mkdir "%APP_DIR%static\state"

REM ----- 8. Launch server and open browser -----
echo.
echo ============================================================
echo   Starting server at %URL%
echo   Keep this window open; press Ctrl+C to stop.
echo ============================================================
echo.

REM Fire off the browser after a short delay (don't fail if start misbehaves).
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start "" "%URL%""

set "HOST=%HOST%"
set "PORT=%PORT%"
set "RELOAD=0"
"%VPY%" server.py
set "EC=%ERRORLEVEL%"

echo.
echo Server stopped (exit code %EC%).
pause
exit /b %EC%

REM ============================================================
:check_python
REM Args: %1 = path to python.exe
REM Returns errorlevel 0 if it's a real working Python >= 3.10 (not Store stub).
set "_p=%~1"
REM Windows Store stubs live under WindowsApps and return exit code 9009
REM or fail on real imports; detect by path substring as a quick filter.
if "!_p!"=="" exit /b 1
echo !_p! | findstr /I /C:"WindowsApps\PythonSoftwareFoundation" >nul
if not errorlevel 1 (
    echo [!] Skipping Windows Store stub: !_p!
    exit /b 1
)
REM Actually import sys and check version.
"%_p%" -c "import sys, venv; sys.exit(0 if sys.version_info>=(3,10) else 2)" 2>nul
if errorlevel 2 exit /b 1
if errorlevel 1 exit /b 1
exit /b 0
