@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Always run from this script's directory
cd /d "%~dp0"
set "PROJECT_DIR=%CD%"

REM ---- Python interpreter detection ----
set "PY_EXE="
if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    set "PY_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
) else if exist "%PROJECT_DIR%\..\.venv\Scripts\python.exe" (
    set "PY_EXE=%PROJECT_DIR%\..\.venv\Scripts\python.exe"
) else (
    for %%I in (python.exe) do set "PY_EXE=%%~$PATH:I"
)

if "%PY_EXE%"=="" (
    echo [ERROR] Python interpreter not found.
    echo Install Python 3.11+ and/or create a virtual env at:
    echo   %PROJECT_DIR%\.venv
    echo or
    echo   %PROJECT_DIR%\..\.venv
    pause
    exit /b 1
)

echo [INFO] Using Python: %PY_EXE%

REM ---- Start MQTT broker (Mosquitto via Docker Compose) ----
echo [INFO] Starting Mosquitto broker (docker compose up -d)...
docker compose up -d
if errorlevel 1 (
    echo [WARN] Could not start Docker Compose broker.
    echo [WARN] If Mosquitto is already running locally on localhost:1883, continuing...
)

REM Small wait for broker startup
for /L %%A in (1,1,2) do (
    >nul ping -n 2 127.0.0.1
)

echo [INFO] Checking broker reachability on localhost:1883...
"%PY_EXE%" -c "import socket,sys; s=socket.socket(); s.settimeout(1.5); rc=s.connect_ex(('127.0.0.1',1883)); s.close(); sys.exit(0 if rc==0 else 1)"
if errorlevel 1 (
    echo [ERROR] MQTT broker is not reachable on localhost:1883.
    echo [ERROR] Start Docker Desktop then run this script again,
    echo [ERROR] or install/start native Mosquitto service on Windows.
    echo [HINT] Docker route: open Docker Desktop, wait until it says running, then execute:
    echo [HINT]   docker compose up -d
    pause
    exit /b 1
)

REM ---- Launch services in order ----
echo [INFO] Launching Edge Preprocessor...
start "Edge Preprocessor" /D "%PROJECT_DIR%" cmd /k ""%PY_EXE%" "%PROJECT_DIR%\src\edge_preprocessor.py" --debug"

for /L %%A in (1,1,2) do (
    >nul ping -n 2 127.0.0.1
)

echo [INFO] Launching Viewer (text)...
start "Viewer" /D "%PROJECT_DIR%" cmd /k ""%PY_EXE%" "%PROJECT_DIR%\src\viewer.py""

for /L %%A in (1,1,2) do (
    >nul ping -n 2 127.0.0.1
)

echo [INFO] Launching Visualizer (live graphs)...
start "Visualizer" /D "%PROJECT_DIR%" cmd /k ""%PY_EXE%" "%PROJECT_DIR%\src\visualizer.py""

for /L %%A in (1,1,2) do (
    >nul ping -n 2 127.0.0.1
)

echo [INFO] Launching Replayer (synthetic ECG)...
start "Replayer" /D "%PROJECT_DIR%" cmd /k ""%PY_EXE%" "%PROJECT_DIR%\src\replayer.py" --mode synthetic --fs 250 --chunk-ms 250 --duration-sec 120 --hr-bpm 72"

echo.
echo [DONE] Started all components in separate windows:
echo [INFO]   - Edge Preprocessor (processes ECG + publishes features)
echo [INFO]   - Viewer (text output: HR/SQI/notes)
echo [INFO]   - Visualizer (live graphs: raw/filtered/peaks)
echo [INFO]   - Replayer (publishes synthetic ECG chunks)
echo.
echo [INFO] Close windows or press Ctrl+C in each to stop services.

echo [TIP] To stop broker too, run: docker compose down
exit /b 0
