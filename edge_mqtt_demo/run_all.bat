@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM Multi-Sensor Edge Pipeline - Full Demo
REM ============================================================================
REM Starts all components: MQTT broker, edge preprocessor, viewer, replayer
REM Supports: ECG, PPG, SpO2, Temperature, IMU sensors
REM ============================================================================

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
echo.
echo [INFO] Launching Edge Preprocessor (multi-sensor)...
start "Edge Preprocessor" /D "%PROJECT_DIR%" cmd /k ""%PY_EXE%" "%PROJECT_DIR%\src\edge_preprocessor.py" --debug"

for /L %%A in (1,1,2) do (
    >nul ping -n 2 127.0.0.1
)

echo [INFO] Launching Monitor (single-window dashboard with page navigation)...
start "Monitor" /D "%PROJECT_DIR%" cmd /k ""%PY_EXE%" "%PROJECT_DIR%\src\visualizer.py""

for /L %%A in (1,1,3) do (
    >nul ping -n 2 127.0.0.1
)

echo [INFO] Launching Replayer (all sensors: ECG, PPG, SpO2, Temp, IMU)...
start "Replayer" /D "%PROJECT_DIR%" cmd /k ""%PY_EXE%" "%PROJECT_DIR%\src\replayer.py" --mode synthetic --duration-sec 300 --hr-bpm 72"

echo.
echo ============================================================================
echo [DONE] Started all components in separate windows:
echo ============================================================================
echo.
echo   REPLAYER (Sensor Simulator)
echo     - Publishes synthetic data for: ECG, PPG, SpO2, Temperature, IMU
echo     - Topics: sim/patient1/ecg, sim/patient1/ppg, sim/patient1/spo2, etc.
echo.
echo   EDGE PREPROCESSOR (Raspberry Pi Simulator)
echo     - Subscribes to all sensor topics
echo     - Processes 5-second windows
echo     - Publishes: edge/patient1/features, edge/patient1/events
echo.
echo   MONITOR (Single-Window Dashboard — 3 pages)
echo     - Page 1: Raw inputs (ECG, PPG, SpO2, Temperature, IMU)
echo     - Page 2: Preprocessed signals (filtered + peaks, trends, motion)
echo     - Page 3: AI features, ML predictions, clinical rules, decision
echo     - Click buttons at bottom to switch pages
echo.
echo ============================================================================
echo [INFO] Close windows or press Ctrl+C in each to stop services.
echo [TIP] To stop broker too, run: docker compose down
echo ============================================================================
exit /b 0
