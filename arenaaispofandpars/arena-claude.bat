@echo off
rem ===========================================================================
rem  arena-claude.bat -- Claude Code on Arena models, one file.
rem
rem  What it does:
rem    1. picks the *installed build that really has the shim*: it tries
rem       arena-api, then `py -m arena_api`, then `python -m arena_api`, and
rem       keeps the first one that reports "anthropic shim: yes" and --session
rem       support (a stale install in PATH is the usual reason "it does not
rem       work");
rem    2. starts it hidden:  --claude --port N (log in %TEMP%);
rem    3. waits for /healthz, and fails immediately with the log tail if the
rem       shim process dies (no 60-second silence);
rem    4. exports ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY / ANTHROPIC_MODEL and
rem       runs claude, forwarding every argument;
rem    5. stops the shim it started when Claude Code exits.
rem
rem  Usage:
rem    arena-claude.bat                 normal start (port 8082)
rem    arena-claude.bat --resume ...    forwarded to claude as is
rem    arena-claude.bat doctor          package / session / engine / SSE check
rem    arena-claude.bat version         which build answers and what it can do
rem
rem  Settings (environment variables, set them before running):
rem    ARENA_CLAUDE_PORT=8082           shim port
rem    ARENA_CLAUDE_MODEL=arena-auto    model Arena should use
rem    ARENA_CLAUDE_SESSION=firefox     session source: firefox, firefox:<profile>,
rem                                     storage, chrome, auto
rem    ARENA_CLAUDE_ENGINE=warm         warm (daemon), http, echo
rem    ARENA_CLAUDE_VERBOSE=1           log every request in the shim log
rem    (chats are ALWAYS archived with this launcher: the sidebar stays clean)
rem    ARENA_CLAUDE_FRESH_PROFILE=1      delete the browser profile first; use it
rem                                     when Chrome refuses to start on it
rem
rem  The output of this file is deliberately ASCII: cmd reads a script in the
rem  console codepage, so Cyrillic here turns into garbage.  Russian
rem  diagnostics come from Python (arena-api --claude --doctor), which writes
rem  through the console API and is unaffected by the codepage.
rem ===========================================================================

setlocal EnableExtensions
chcp 65001 >nul 2>&1
title Arena + Claude Code

set "SHIM_PORT=%ARENA_CLAUDE_PORT%"
if "%SHIM_PORT%"=="" set "SHIM_PORT=8082"
set "MODEL=%ARENA_CLAUDE_MODEL%"
if "%MODEL%"=="" set "MODEL=arena-auto"
set "SESSION=%ARENA_CLAUDE_SESSION%"
if "%SESSION%"=="" set "SESSION=firefox"
set "ENGINE=%ARENA_CLAUDE_ENGINE%"
if "%ENGINE%"=="" set "ENGINE=warm"
set "EXTRA="
if "%ARENA_CLAUDE_VERBOSE%"=="1" set "EXTRA=--verbose"
rem Archive is not optional here: every answered chat is filed away, so the
rem Arena sidebar does not collect hundreds of one-shot conversations.
set "ARENA_CLAUDE_ARCHIVE=1"
set "EXTRA=%EXTRA% --archive"
set "ARCHIVE_NOTE=always (after the whole answer is received)"
set "PROFILE_NOTE=reused (stale locks are healed)"
if "%ARENA_CLAUDE_FRESH_PROFILE%"=="1" set "EXTRA=%EXTRA% --fresh-profile"
if "%ARENA_CLAUDE_FRESH_PROFILE%"=="1" set "PROFILE_NOTE=recreated (--fresh-profile)"

set "SHIMPID="
set "STARTED_SHIM="
set "BUILD_OK="
set "LAUNCH_EXE="
set "LAUNCH_ARGS="
set "DOCTOR_ARGS="
set "REPO_DIR=%~dp0"
set "BASE=%TEMP%\arena-claude-%SHIM_PORT%"
set "LOG=%BASE%.log"
set "ERRLOG=%BASE%.err.log"
set "VERSION_FILE=%BASE%.version.txt"

rem Python must write UTF-8 into the log whatever the console codepage is.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

call :pick_build
if not defined LAUNCH_EXE goto no_package

if /i "%~1"=="doctor" goto doctor
if /i "%~1"=="--doctor" goto doctor
if /i "%~1"=="version" goto print_version
if /i "%~1"=="--version" goto print_version

if not "%BUILD_OK%"=="1" goto stale_build

call :probe
if "%PORT_OPEN%"=="1" goto run_claude

echo [arena] starting shim on port %SHIM_PORT%  (engine=%ENGINE% session=%SESSION% model=%MODEL%)
echo [arena] using %LAUNCH_EXE% %DOCTOR_ARGS%
if exist "%LOG%" del "%LOG%" >nul 2>&1
if exist "%ERRLOG%" del "%ERRLOG%" >nul 2>&1

for /f "usebackq delims=" %%p in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $p = Start-Process -FilePath '%LAUNCH_EXE%' -ArgumentList %PS_ARGS% -PassThru -WindowStyle Hidden -RedirectStandardOutput '%LOG%' -RedirectStandardError '%ERRLOG%'; $p.Id" 2^>nul`) do set "SHIMPID=%%p"

if not defined SHIMPID goto no_start
set "STARTED_SHIM=1"
echo [arena] shim pid %SHIMPID%, log %LOG%

powershell -NoProfile -ExecutionPolicy Bypass -Command "$u = 'http://127.0.0.1:%SHIM_PORT%/healthz'; for ($i = 0; $i -lt 300; $i++) { try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 -Uri $u; if ($r.StatusCode -eq 200) { exit 0 } } catch {} if (-not (Get-Process -Id %SHIMPID% -ErrorAction SilentlyContinue)) { exit 2 } Start-Sleep -Milliseconds 80 }; exit 1" >nul 2>&1
if "%ERRORLEVEL%"=="0" goto shim_ready
if "%ERRORLEVEL%"=="2" goto shim_died
goto no_response

:shim_died
echo.
echo [arena] ERROR: the shim exited right after start.
call :show_log
echo.
echo [arena] running diagnostics...
call :run_doctor
call :stop_shim
call :cleanup
endlocal & exit /b 1

:no_response
echo.
echo [arena] ERROR: no answer on http://127.0.0.1:%SHIM_PORT%/healthz in 30 s.
call :show_log
echo.
echo [arena] running diagnostics...
call :run_doctor
call :stop_shim
call :cleanup
endlocal & exit /b 1

:no_start
echo.
echo [arena] ERROR: PowerShell could not start the shim.
echo [arena] Run it yourself to see the reason:
echo         %LAUNCH_EXE% %DOCTOR_ARGS% --claude --port %SHIM_PORT%
call :cleanup
endlocal & exit /b 1

:stale_build
echo.
echo [arena] ERROR: no build here can serve --claude.
echo [arena] Two usual causes:
echo [arena]   * arena-api.exe is left over from an install whose Python no
echo [arena]     longer has the package  (ModuleNotFoundError: arena_api), or
echo [arena]   * the installed build predates --claude / --session.
echo.
if exist "%VERSION_FILE%" powershell -NoProfile -Command "Get-Content -Encoding UTF8 '%VERSION_FILE%'"
echo [arena] What that output says and what to do:
findstr /C:"No module named" "%VERSION_FILE%" >nul 2>&1
if not errorlevel 1 (
    echo         the module is missing in that interpreter. Reinstall it there:
    echo                 py -m pip install -e "."
    echo         or just delete the broken script:
    echo                 del "%LOCALAPPDATA%\Programs\Python\Python311\Scripts\arena-api.exe"
)
findstr /C:"unrecognized arguments" "%VERSION_FILE%" >nul 2>&1
if not errorlevel 1 (
    echo         that build is too old. Reinstall from the clone:
    echo                 py -m pip install -U -e ".[auto]"
)
echo [arena] Meanwhile this file always works from the repository root:
echo         py -m arena_api --claude --port %SHIM_PORT%
call :cleanup
endlocal & exit /b 1

:shim_ready
echo [arena] shim is up.

:run_claude
echo.
echo   ANTHROPIC_BASE_URL = http://127.0.0.1:%SHIM_PORT%
echo   model              = %MODEL%
echo   session            = %SESSION%
echo   engine             = %ENGINE%
echo   archive chat       = %ARCHIVE_NOTE%
echo   browser profile    = %PROFILE_NOTE%
echo.
if not defined STARTED_SHIM echo [arena] port %SHIM_PORT% was already served - reusing it, not stopping it.

where claude >nul 2>&1
if errorlevel 1 goto no_claude

set "ANTHROPIC_BASE_URL=http://127.0.0.1:%SHIM_PORT%"
set "ANTHROPIC_API_KEY=arena"
set "ANTHROPIC_MODEL=%MODEL%"
set "ANTHROPIC_SMALL_FAST_MODEL=%MODEL%"
rem Telemetry and update checks bypass the shim; switching them off keeps
rem Claude Code from stalling on api.anthropic.com.
set "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"

claude %*
set "CLAUDE_EXIT=%ERRORLEVEL%"

echo.
echo [arena] Claude Code exited with code %CLAUDE_EXIT%.
call :stop_shim
call :cleanup
endlocal & exit /b %CLAUDE_EXIT%

:no_claude
echo [arena] ERROR: the claude command is not in PATH.
echo [arena] Install Claude Code:  npm install -g @anthropic-ai/claude-code
echo.
echo [arena] The shim is running anyway: http://127.0.0.1:%SHIM_PORT%
echo [arena] In another window:
echo           set ANTHROPIC_BASE_URL=http://127.0.0.1:%SHIM_PORT%
echo           set ANTHROPIC_API_KEY=arena
echo           claude
pause
call :stop_shim
call :cleanup
endlocal & exit /b 1

rem ===========================================================================
rem  Helpers
rem ===========================================================================

:doctor
echo [arena] diagnostics: package, session, answer engine, loopback protocol
echo.
call :run_doctor
call :cleanup
endlocal & exit /b %ERRORLEVEL%

:print_version
"%LAUNCH_EXE%" %DOCTOR_ARGS% --version
call :cleanup
endlocal & exit /b %ERRORLEVEL%

:run_doctor
"%LAUNCH_EXE%" %DOCTOR_ARGS% --claude --doctor --engine %ENGINE% --session "%SESSION%" --model "%MODEL%"
goto :eof

rem --- find the build that can actually serve -------------------------------
:pick_build
set "BUILD_OK="
rem 1. The console script (what pip installs).
call :consider arena-api exe
if "%BUILD_OK%"=="1" goto :build_args
rem 2./3. Run the package straight out of this clone: the script lives in the
rem    repository root, so `-m arena_api` needs no installation at all.  This
rem    also rescues the common breakage "arena-api.exe exists but its Python
rem    no longer has the module".
call :consider py module
if "%BUILD_OK%"=="1" goto :build_args
call :consider python module
if "%BUILD_OK%"=="1" goto :build_args
goto :eof

:consider
rem %~1 = executable to look for, %~2 = exe | module
set "CAND="
for /f "delims=" %%e in ('where %~1 2^>nul') do if not defined CAND set "CAND=%%e"
if not defined CAND goto :eof
set "LAUNCH_EXE=%CAND%"
set "DOCTOR_ARGS="
if /i "%~2"=="module" (
    set "DOCTOR_ARGS=-m arena_api"
    rem Make the clone importable from any working directory.
    set "PYTHONPATH=%REPO_DIR%;%PYTHONPATH%"
)
if exist "%VERSION_FILE%" del "%VERSION_FILE%" >nul 2>&1
"%LAUNCH_EXE%" %DOCTOR_ARGS% --version > "%VERSION_FILE%" 2>&1
findstr /C:"anthropic shim: yes" "%VERSION_FILE%" >nul 2>&1
if errorlevel 1 goto :eof
findstr /C:"--session" "%VERSION_FILE%" >nul 2>&1
if errorlevel 1 goto :eof
set "BUILD_OK=1"
goto :eof

:build_args
set "LAUNCH_ARGS=--claude --port %SHIM_PORT% --engine %ENGINE% --session "%SESSION%" --model "%MODEL%" %EXTRA%"
rem PowerShell takes the child arguments as an array of single-quoted strings,
rem so a profile path with spaces survives the hand-off.
set "PS_ARGS="
if defined DOCTOR_ARGS set "PS_ARGS='-m','arena_api',"
set "PS_ARGS=%PS_ARGS%'--claude','--port','%SHIM_PORT%','--engine','%ENGINE%','--session','%SESSION%','--model','%MODEL%'"
if defined EXTRA set "PS_ARGS=%PS_ARGS%,'%EXTRA%'"
goto :eof

rem --- /healthz probe: curl when present, PowerShell otherwise ---------------
:probe
set "PORT_OPEN=0"
where curl >nul 2>&1
if errorlevel 1 goto probe_ps
curl -s -f -m 2 -o NUL "http://127.0.0.1:%SHIM_PORT%/healthz" >nul 2>&1
if not errorlevel 1 set "PORT_OPEN=1"
goto :eof

:probe_ps
for /f "usebackq delims=" %%r in (`powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%SHIM_PORT%/healthz').StatusCode } catch { '' }" 2^>nul`) do (
    if not "%%r"=="" set "PORT_OPEN=1"
)
goto :eof

rem --- is our shim process still alive? --------------------------------------
:alive
set "ALIVE=0"
if not defined SHIMPID goto :eof
tasklist /FI "PID eq %SHIMPID%" 2>nul | findstr /R /C:" %SHIMPID% " >nul 2>&1
if not errorlevel 1 set "ALIVE=1"
goto :eof

rem --- print the tail of the shim log ---------------------------------------
:show_log
if not exist "%LOG%" goto show_err_log
echo ------- shim stdout (tail) -------
powershell -NoProfile -Command "Get-Content -Encoding UTF8 -Tail 20 '%LOG%'"
:show_err_log
if not exist "%ERRLOG%" goto :eof
echo ------- shim stderr (tail) -------
powershell -NoProfile -Command "Get-Content -Encoding UTF8 -Tail 20 '%ERRLOG%'"
goto :eof

rem --- stop the shim we started (never somebody else's) ----------------------
:stop_shim
if not defined STARTED_SHIM goto :eof
if not defined SHIMPID goto :eof
taskkill /F /T /PID %SHIMPID% >nul 2>&1
set "STARTED_SHIM="
goto :eof

:cleanup
if exist "%VERSION_FILE%" del "%VERSION_FILE%" >nul 2>&1
goto :eof

:no_package
echo.
echo [arena] ERROR: neither arena-api nor python was found.
echo [arena] Install the package from the repository clone:
echo         pip install -U -e ".[auto]"
echo [arena] Then check:  arena-api --version
call :cleanup
endlocal & exit /b 1
