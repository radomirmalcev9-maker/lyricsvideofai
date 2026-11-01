@echo off
REM =====================================================================
REM  Apple Motion Lyrics - one-click launcher for Windows
REM  Double-click to install all dependencies and open the site.
REM
REM  Important: this file must stay ASCII-only - no BOM, no emoji.
REM  cmd.exe parses line-by-line; we avoid parentheses blocks and &&
REM  chaining because both trigger the "unexpected" syntax errors on
REM  non-English Windows locales.
REM =====================================================================

cd /d "%~dp0"
set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%.venv"
set "PORT=4770"
set "HOST=127.0.0.1"
set "URL=http://%HOST%:%PORT%/static/index.html"
set "PYTHON="
set "ERR="

echo.
echo ============================================================
echo   Apple Motion Lyrics
echo ============================================================
echo.

REM ---------------------------------------------------------------------
REM 1. Find a real Python (skip the Windows Store stub under WindowsApps).
REM ---------------------------------------------------------------------
set "CAND="
set "PY_CMD="

REM Prefer the "py" launcher - it never points at the Store stub.
where py >nul 2>nul
if %ERRORLEVEL%==0 set "PY_CMD=py -3"

if not defined PY_CMD goto :try_python
echo [i] Probing Python via "py -3" ...
for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "CAND=%%i"
if not defined CAND goto :no_python
call :test_candidate "%CAND%"
if defined PYTHON goto :found_python

:try_python
REM Fallback: look for a python.exe on PATH and check it is not the Store stub.
where python >nul 2>nul
if not %ERRORLEVEL%==0 goto :no_python
echo [i] Probing Python via "python" on PATH ...
for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "CAND=%%i"
if not defined CAND goto :no_python
REM Reject if the path contains the Windows Store Python stub directory.
echo %CAND% | findstr /I /C:"WindowsApps\PythonSoftwareFoundation" >nul
if %ERRORLEVEL%==0 goto :no_python
call :test_candidate "%CAND%"
if defined PYTHON goto :found_python

:no_python
echo [X] Could not find a working Python 3.10+.
echo.
echo     Install real Python from  https://www.python.org/downloads/
echo     and tick "Add Python to PATH" during setup.
echo     Then double-click this file again.
echo.
goto :die

:found_python
echo [i] Using Python: %PYTHON%

REM ---------------------------------------------------------------------
REM 2. Create virtualenv if it does not exist.
REM ---------------------------------------------------------------------
if exist "%VENV_DIR%\Scripts\python.exe" goto :have_venv
echo [i] Creating virtual environment in %VENV_DIR% ...
"%PYTHON%" -m venv "%VENV_DIR%"
if exist "%VENV_DIR%\Scripts\python.exe" goto :have_venv
echo [X] Failed to create virtualenv. Delete .venv\ and try again.
goto :die

:have_venv
set "VPY=%VENV_DIR%\Scripts\python.exe"
set "VPIP=%VPY% -m pip"

REM ---------------------------------------------------------------------
REM 3. Upgrade pip.
REM ---------------------------------------------------------------------
echo [i] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip --quiet
if %ERRORLEVEL%==0 goto :pip_ok
echo [X] pip upgrade failed (continuing anyway).
:pip_ok

REM ---------------------------------------------------------------------
REM 4. Base dependencies.
REM ---------------------------------------------------------------------
echo [i] Installing FastAPI, uvicorn, httpx, python-multipart ...
"%VPIP%" install --quiet "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx[http2]>=0.27" "python-multipart>=0.0.9"
if %ERRORLEVEL%==0 goto :base_ok
set "ERR=base dependencies"
goto :install_fail
:base_ok

REM ---------------------------------------------------------------------
REM 5. faster-whisper.
REM ---------------------------------------------------------------------
echo [i] Installing faster-whisper (first run downloads ~500 MB) ...
"%VPIP%" install --quiet "faster-whisper>=1.0.0"
if %ERRORLEVEL%==0 goto :whisper_ok
set "ERR=faster-whisper"
goto :install_fail
:whisper_ok

REM ---------------------------------------------------------------------
REM 6. arena-api (editable from sibling folder) + Playwright Chromium.
REM ---------------------------------------------------------------------
set "ARENA_DIR=%APP_DIR%..\arenaaispofandpars"
if not exist "%ARENA_DIR%\pyproject.toml" goto :arena_skip
echo [i] Installing arena-api from %ARENA_DIR% ...
"%VPIP%" install --quiet -e "%ARENA_DIR%[auto]"
if %ERRORLEVEL%==0 goto :arena_full_ok
echo [!] arena-api [auto] install failed; trying without browser extras ...
"%VPIP%" install --quiet -e "%ARENA_DIR%"
:arena_full_ok
echo [i] Installing Playwright Chromium for invisible browser ...
"%VPY%" -m playwright install chromium
goto :arena_done
:arena_skip
echo [!] arenaaispofandpars/ folder not found next to my-lyrics-app/.
echo     Arena AI calls will be unavailable until you put it there.
:arena_done

REM ---------------------------------------------------------------------
REM 7. Ensure static subfolders exist.
REM ---------------------------------------------------------------------
if not exist "%APP_DIR%static\uploads" mkdir "%APP_DIR%static\uploads"
if not exist "%APP_DIR%static\state"   mkdir "%APP_DIR%static\state"

REM ---------------------------------------------------------------------
REM 8. Start server and open browser.
REM ---------------------------------------------------------------------
echo.
echo ============================================================
echo   Starting server at %URL%
echo   Keep this window open; press Ctrl+C to stop.
echo ============================================================
echo.

REM Open the browser after a short delay, then run the server in foreground.
REM Use a one-line temp .cmd to avoid nested-quoting hell inside cmd /c.
echo @echo off> "%TEMP%\aml_open_url.cmd"
echo timeout /t 4 /nobreak ^>nul>> "%TEMP%\aml_open_url.cmd"
echo start "" "%URL%">> "%TEMP%\aml_open_url.cmd"
start "" /b "%TEMP%\aml_open_url.cmd"
set "HOST=%HOST%"
set "PORT=%PORT%"
set "RELOAD=0"
"%VPY%" server.py
echo.
echo Server stopped.
goto :end

REM ---------------------------------------------------------------------
REM Helpers
REM ---------------------------------------------------------------------

:install_fail
echo [X] Failed to install %ERR%. Check network, then re-run.
goto :die

:die
echo.
echo Something went wrong. The window stays open so you can read the error above.
echo Press any key to close.
pause
exit /b 1

:end
pause
exit /b 0

REM - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
REM :test_candidate  pathToPythonExe
REM   Sets PYTHON to the path if it runs and is Python 3.10+.
REM   Leaves PYTHON empty on failure.
:test_candidate
set "_p=%~1"
if "%_p%"=="" exit /b 0
REM Reject Windows Store stubs explicitly.
echo %_p% | findstr /I /C:"WindowsApps\PythonSoftwareFoundation" >nul
if not %ERRORLEVEL%==0 goto :tc_not_store_stub
echo [!] Skipping Windows Store stub: %_p%
exit /b 0
:tc_not_store_stub
REM Try to actually import sys and check version.
"%_p%" -c "import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 3)" 2>nul
if %ERRORLEVEL%==0 goto :tc_version_ok
echo [!] Rejected (not a usable Python >=3.10): %_p%
exit /b 0
:tc_version_ok
set "PYTHON=%_p%"
exit /b 0
