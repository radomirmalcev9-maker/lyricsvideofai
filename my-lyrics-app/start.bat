@echo off
REM =====================================================================
REM  Apple Motion Lyrics - one-click launcher for Windows
REM
REM  Rules for this file:
REM    * ASCII-only (no BOM, no emoji, no Cyrillic in this script itself).
REM    * Linear goto-based flow. No parenthesized if/for blocks and no
REM      &&/|| chaining - both break cmd.exe on Russian Windows when
REM      quoted paths contain spaces/parentheses.
REM    * Always quote paths with "..." but DO NOT quote the entire
REM      "executable arg1 arg2" string - cmd.exe would treat it as the
REM      executable name.
REM =====================================================================

REM Switch console to UTF-8 so Cyrillic usernames/paths don't get mangled.
chcp 65001 >nul

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
REM ---------------------------------------------------------------------

:try_py_launcher
set "CAND="
where py >nul 2>nul
if not %ERRORLEVEL%==0 goto :try_python_path
echo [i] Checking "py -3" launcher ...
for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "CAND=%%i"
if defined CAND call :test_candidate "%%i"
if defined PYTHON goto :found_python

:try_python_path
set "CAND="
where python >nul 2>nul
if not %ERRORLEVEL%==0 goto :scan_common_paths
echo [i] Checking "python" on PATH ...
for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "CAND=%%i"
if defined CAND call :test_candidate "%%i"
if defined PYTHON goto :found_python

:scan_common_paths
echo [i] Scanning common Python install folders ...

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

:scan_conda
if not defined LOCALAPPDATA goto :scan_done
call :try_path "%LOCALAPPDATA%\anaconda3\python.exe"
if defined PYTHON goto :found_python
call :try_path "%LOCALAPPDATA%\miniconda3\python.exe"
if defined PYTHON goto :found_python
:scan_done

:no_python
echo [X] No working Python 3.10+ was found on this PC.
echo.
echo     Download it from:  https://www.python.org/downloads/
echo     During install, TICK "Add Python to PATH" at the bottom.
echo     Then double-click this file again.
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
echo [X] Failed to create virtualenv. Delete the .venv\ folder (if any)
echo     and double-click this file again.
goto :die

:have_venv
set "VPY=%VENV_DIR%\Scripts\python.exe"

REM ---------------------------------------------------------------------
REM 3. Upgrade pip.
REM    NOTE: call pip as  "%VPY%" -m pip ...   (quotes ONLY around exe,
REM    never around the whole command, otherwise cmd treats spaces
REM    and "-m pip" as part of the executable path).
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
"%VPY%" -m pip install --quiet "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx[http2]>=0.27" "python-multipart>=0.0.9"
if %ERRORLEVEL%==0 goto :base_ok
set "ERR=base dependencies (fastapi, uvicorn, httpx, python-multipart)"
goto :install_fail
:base_ok

REM ---------------------------------------------------------------------
REM 5. faster-whisper.
REM ---------------------------------------------------------------------
echo [i] Installing faster-whisper (first run downloads ~500 MB) ...
"%VPY%" -m pip install --quiet "faster-whisper>=1.0.0"
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
"%VPY%" -m pip install --quiet -e "%ARENA_DIR%[auto]"
if %ERRORLEVEL%==0 goto :arena_full_ok
echo [!] arena-api [auto] install failed; trying without browser extras ...
"%VPY%" -m pip install --quiet -e "%ARENA_DIR%"
:arena_full_ok
echo [i] Installing Playwright Chromium (invisible browser) ...
"%VPY%" -m playwright install chromium
goto :arena_done
:arena_skip
echo [!] arenaaispofandpars\ folder not found next to my-lyrics-app\.
echo     Arena AI calls will be unavailable until you add it.
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

REM Open the browser via a tiny helper .cmd (avoids nested quoting).
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
echo     Check your Internet connection and double-click this file again.
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
REM :try_path  path  -> if exists, pass it to :test_candidate.
:try_path
if not exist "%~1" exit /b 0
call :test_candidate "%~1"
exit /b 0

REM - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
REM :test_candidate  path  -> sets PYTHON if it's real Python ^>=3.10.
:test_candidate
set "_p=%~1"
if "%_p%"=="" exit /b 0
REM Reject Windows Store stubs.
echo %_p% | findstr /I /C:"WindowsApps\PythonSoftwareFoundation" >nul
if not %ERRORLEVEL%==0 goto :tc_not_store_stub
echo [!] Skipping Windows Store stub: %_p%
exit /b 0
:tc_not_store_stub
REM Actually try to run and check version.
"%_p%" -c "import sys;raise SystemExit(0 if sys.version_info>=(3,10) else 3)" 2>nul
if %ERRORLEVEL%==0 goto :tc_version_ok
echo [!] Rejected (not a usable Python ^>=3.10): %_p%
exit /b 0
:tc_version_ok
set "PYTHON=%_p%"
exit /b 0
