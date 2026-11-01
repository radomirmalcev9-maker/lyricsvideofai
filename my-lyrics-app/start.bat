@echo off
REM =====================================================================
REM  Apple Motion Lyrics - one-click launcher for Windows
REM  Double-click to install all dependencies and open the site.
REM
REM  This file MUST stay ASCII-only (no BOM, no emoji, no non-ASCII chars).
REM  cmd.exe trips over parenthesized if-blocks and && chaining on Russian
REM  Windows, so we use a linear goto-based flow.
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
REM 1. Find a real Python 3.10+.
REM
REM    Strategy:
REM      a) "py -3" launcher (most reliable when real Python is installed)
REM      b) "python" on PATH, but REJECT any path under WindowsApps (Store stub)
REM      c) Direct scan of common install folders:
REM           C:\Python310, C:\Python311, C:\Python312, C:\Python313
REM           %LOCALAPPDATA%\Programs\Python\Python310\python.exe (and 11/12/13)
REM           %ProgramFiles%\Python31x\python.exe
REM           %ProgramFiles(x86)%\Python31x\python.exe
REM ---------------------------------------------------------------------

:try_py_launcher
set "CAND="
where py >nul 2>nul
if not %ERRORLEVEL%==0 goto :try_python_path
echo [i] Checking "py -3" launcher ...
for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "CAND=%%i"
if defined CAND call :test_candidate "%CAND%"
if defined PYTHON goto :found_python

:try_python_path
set "CAND="
where python >nul 2>nul
if not %ERRORLEVEL%==0 goto :scan_common_paths
echo [i] Checking "python" on PATH ...
for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "CAND=%%i"
if defined CAND call :test_candidate "%CAND%"
if defined PYTHON goto :found_python

:scan_common_paths
echo [i] Scanning common Python install folders ...

REM --- C:\Python310 .. C:\Python314 (all-users system install) ---
call :try_path "C:\Python314\python.exe"
if defined PYTHON goto :found_python
call :try_path "C:\Python313\python.exe"
if defined PYTHON goto :found_python
call :try_path "C:\Python312\python.exe"
if defined PYTHON goto :found_python
call :try_path "C:\Python311\python.exe"
if defined PYTHON goto :found_python
call :try_path "C:\Python310\python.exe"
if defined PYTHON goto :found_python

REM --- %LOCALAPPDATA%\Programs\Python\Python3xx\python.exe (default per-user install) ---
if not defined LOCALAPPDATA goto :scan_programfiles
call :try_path "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if defined PYTHON goto :found_python
call :try_path "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if defined PYTHON goto :found_python
call :try_path "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PYTHON goto :found_python
call :try_path "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if defined PYTHON goto :found_python
call :try_path "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if defined PYTHON goto :found_python

REM --- %ProgramFiles% ---
:scan_programfiles
if not defined ProgramFiles goto :scan_programfiles_x86
call :try_path "%ProgramFiles%\Python314\python.exe"
if defined PYTHON goto :found_python
call :try_path "%ProgramFiles%\Python313\python.exe"
if defined PYTHON goto :found_python
call :try_path "%ProgramFiles%\Python312\python.exe"
if defined PYTHON goto :found_python
call :try_path "%ProgramFiles%\Python311\python.exe"
if defined PYTHON goto :found_python
call :try_path "%ProgramFiles%\Python310\python.exe"
if defined PYTHON goto :found_python

:scan_programfiles_x86
if not defined ProgramFiles(x86) goto :scan_conda
call :try_path "%ProgramFiles(x86)%\Python314\python.exe"
if defined PYTHON goto :found_python
call :try_path "%ProgramFiles(x86)%\Python313\python.exe"
if defined PYTHON goto :found_python
call :try_path "%ProgramFiles(x86)%\Python312\python.exe"
if defined PYTHON goto :found_python
call :try_path "%ProgramFiles(x86)%\Python311\python.exe"
if defined PYTHON goto :found_python
call :try_path "%ProgramFiles(x86)%\Python310\python.exe"
if defined PYTHON goto :found_python

REM --- Anaconda / Miniconda per-user installs ---
:scan_conda
if not defined LOCALAPPDATA goto :scan_done
call :try_path "%LOCALAPPDATA%\anaconda3\python.exe"
if defined PYTHON goto :found_python
call :try_path "%LOCALAPPDATA%\miniconda3\python.exe"
if defined PYTHON goto :found_python
:scan_done

REM - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
REM :try_path  path
REM   If the file exists, test it as a Python candidate via :test_candidate.
:try_path
if not exist "%~1" exit /b 0
call :test_candidate "%~1"
exit /b 0

:no_python
echo [X] No working Python 3.10+ was found on this PC.
echo.
echo     Download the installer from:
echo         https://www.python.org/downloads/
echo     Run it and IMPORTANT: tick "Add Python to PATH" at the bottom.
echo     Then double-click this start.bat again.
echo.
echo     Opening the download page in your browser now ...
start "" "https://www.python.org/downloads/"
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
echo [X] Failed to create virtualenv. Delete the .venv\ folder (if any) and try again.
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
echo [!] pip upgrade failed (continuing anyway).
:pip_ok

REM ---------------------------------------------------------------------
REM 4. Base dependencies.
REM ---------------------------------------------------------------------
echo [i] Installing FastAPI, uvicorn, httpx, python-multipart ...
"%VPIP%" install --quiet "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx[http2]>=0.27" "python-multipart>=0.0.9"
if %ERRORLEVEL%==0 goto :base_ok
set "ERR=base dependencies (fastapi, uvicorn, httpx, python-multipart)"
goto :install_fail
:base_ok

REM ---------------------------------------------------------------------
REM 5. faster-whisper.
REM ---------------------------------------------------------------------
echo [i] Installing faster-whisper (first run downloads ~500 MB model) ...
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
echo [i] Installing Playwright Chromium (the invisible browser) ...
"%VPY%" -m playwright install chromium
goto :arena_done
:arena_skip
echo [!] arenaaispofandpars\ folder not found next to my-lyrics-app\.
echo     Arena AI calls will be unavailable until you put that folder there.
:arena_done

REM ---------------------------------------------------------------------
REM 7. Ensure static subfolders exist.
REM ---------------------------------------------------------------------
if not exist "%APP_DIR%static\uploads" mkdir "%APP_DIR%static\uploads"
if not exist "%APP_DIR%static\state"   mkdir "%APP_DIR%static\state"

REM ---------------------------------------------------------------------
REM 8. Launch server and open browser.
REM ---------------------------------------------------------------------
echo.
echo ============================================================
echo   Starting server at %URL%
echo   Keep this window open; press Ctrl+C to stop.
echo ============================================================
echo.

REM Write a tiny helper .cmd to open the browser after 4 seconds.
REM Avoids cmd /c "command1 && command2" quoting problems.
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
echo.
echo [X] Failed to install %ERR%.
echo     Check your Internet connection and re-run this file.
goto :die

:die
echo.
echo Press any key to close this window.
pause
exit /b 1

:end
echo.
echo Press any key to close this window.
pause
exit /b 0

REM - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
REM :test_candidate  pathToPythonExe
REM   Sets PYTHON to the path if it is a real runnable Python >= 3.10
REM   and is NOT the Windows Store stub.  Leaves PYTHON empty otherwise.
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
echo [!] Rejected (not a usable Python ^>=3.10): %_p%
exit /b 0
:tc_version_ok
set "PYTHON=%_p%"
exit /b 0
