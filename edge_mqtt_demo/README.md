# edge_mqtt_demo

Hardware-free digital twin for your smart medical mat pipeline:

- **Publisher (ESP32 simulator)**: replays or synthesizes ECG and publishes MQTT chunks.
- **Edge AI Gateway (Raspberry simulator)**: subscribes, windows, preprocesses ECG, detects R-peaks, estimates HR, computes SQI, publishes features.
- **Viewer (server/dashboard simulator)**: subscribes to features and prints/plots.
  - **Text viewer**: logs HR/SQI/notes to console
  - **Visual viewer**: live graphs showing raw/filtered ECG + R-peaks

This is intentionally minimal but plug-compatible with future ESP32 + Raspberry deployment.

---

## Architecture and future mapping

### Future target behavior
- ESP32 acquires ECG (and later PPG/SpO2/Temp/IMU), publishes MQTT chunks with stable schema/topic.
- Raspberry Pi runs `edge_preprocessor.py` as Edge AI Gateway.
- Gateway outputs compact features/events (`edge_summary` later) to server/dashboard/LLM-RAG.

### What we simulate now
- Laptop runs all roles:
  1) `replayer.py` instead of ESP32,
  2) `edge_preprocessor.py` instead of Raspberry,
  3) `viewer.py` + `visualizer.py` instead of cloud/dashboard.
- Mosquitto runs locally as broker.

### Why this matters for later integration
- Keeps **message contracts** fixed from day one.
- Validates **timestamps/chunking/windowing** behavior before hardware arrives.
- Minimizes integration risk to “replace publisher + move edge service to Raspberry + update env vars”.

---

## Project structure

```
edge_mqtt_demo/
  data/
    README.md
  src/
    common.py
    replayer.py
    edge_preprocessor.py
    viewer.py
    visualizer.py
  requirements.txt
  docker-compose.yml
  run_all.bat
  run_visualizer.bat
  README.md
```

---

## MQTT contracts (stable for future hardware)

### Input topic
- `sim/patient1/ecg`

### Output topic
- `edge/patient1/features`

### Input JSON schema
```json
{
  "patient_id": "patient1",
  "t0_ms": 1700000000000,
  "fs_hz": 250,
  "samples": [0.01, 0.02, -0.01]
}
```

### Output JSON schema
```json
{
  "patient_id": "patient1",
  "window_start_ms": 1700000000000,
  "window_sec": 5,
  "hr_bpm": 72.4,
  "sqi": 0.91,
  "notes": ["low_sqi"]
}
```

---

## Step 0 — Prerequisites + project creation + Mosquitto setup

### Future target behavior
- On deployment, broker may run on Raspberry or LAN server; services connect via host/port config only.

### What we simulate now
- Local Mosquitto broker on your laptop (`localhost:1883`).

### Why this step matters for later integration
- MQTT endpoint abstraction is the same in simulation and production.

### 0.1 Install required software (Windows)
1. Install **Python 3.11+** from python.org.
2. Install **Docker Desktop** (recommended) for Mosquitto via Compose.
   - Alternative: install native Mosquitto.
3. Open PowerShell in project root:
   - `cd "c:\Users\amria\OneDrive\Bureau\raspberry pi models\edge_mqtt_demo"`

### 0.2 Create Python virtual environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Expected output:
- `Successfully installed paho-mqtt ... numpy ... scipy ... matplotlib ...`

Debug checklist:
- `python --version` shows `3.11+`.
- `pip list` contains `paho-mqtt`, `numpy`, `scipy`.

### 0.3 Start Mosquitto broker
```powershell
docker compose up -d
```

Expected output:
- Container `edge-mqtt-demo-broker` running.

Debug checklist:
- `docker ps` includes `edge-mqtt-demo-broker` and `0.0.0.0:1883->1883/tcp`.
- If port conflict, stop other MQTT service and rerun.

### 0.4 What is Docker + Mosquitto?
**In plain terms:**
- Mosquitto is a message broker (like a mailbox server).
- The three Python services (replayer, edge, viewer) send/receive messages through it on `localhost:1883`.
- Docker is just a way to run Mosquitto without installing it on your machine.

**If you have Docker Desktop:**
1. Open Docker Desktop, wait until it says "Engine running".
2. Run: `docker compose up -d`
3. This starts a Mosquitto container on port 1883.
4. Then run: `.\run_all.bat`

**If you don't have/want Docker:**
1. Install native Mosquitto from https://mosquitto.org/download/#windows (choose Windows MSI).
2. Install and run it as a Windows service (starts automatically).
3. It will listen on `localhost:1883` automatically.
4. Then run: `.\run_all.bat`

**Why viewer was empty before:**
- Services tried to connect to broker on `localhost:1883`, but broker wasn't running.
- Now services wait and retry (instead of crashing), so once broker is up, they auto-connect.

---

## Step 1 — Run Edge AI Gateway service (Raspberry simulator)

### Future target behavior
- Same code runs on Raspberry Pi, subscribing to sensor topics and publishing edge features.

### What we simulate now
- On laptop, `edge_preprocessor.py` subscribes to `sim/patient1/ecg` and publishes to `edge/patient1/features`.

### Why this step matters for later integration
- Locks in edge behavior (windowing, preprocessing, feature contract) before hardware coupling.

Open **Terminal 1**:
```powershell
cd "c:\Users\amria\OneDrive\Bureau\raspberry pi models\edge_mqtt_demo"
.\.venv\Scripts\Activate.ps1
python .\src\edge_preprocessor.py --debug
```

Expected output:
- `[edge] connected to MQTT localhost:1883`
- `[edge] subscribed to sim/patient1/ecg`

Debug checklist:
- If connection fails, verify Mosquitto is running.
- If schema errors appear, inspect publisher payload fields.

---

## Step 2 — Run ECG replayer (ESP32 simulator)

### Future target behavior
- ESP32 firmware will publish same topic/schema/chunk cadence.

### What we simulate now
- `replayer.py` publishes chunked ECG (synthetic or file mode) every 200–500 ms.

### Why this step matters for later integration
- Ensures timing and payload assumptions are validated against edge pipeline now.

Open **Terminal 2** (synthetic mode, real-time):
```powershell
cd "c:\Users\amria\OneDrive\Bureau\raspberry pi models\edge_mqtt_demo"
.\.venv\Scripts\Activate.ps1
python .\src\replayer.py --mode synthetic --fs 250 --chunk-ms 250 --duration-sec 120 --hr-bpm 72
```

Expected output:
- `[replayer] connected to MQTT localhost:1883`
- `[replayer] topic=sim/patient1/ecg chunk=... samples=62 t0=...`

Debug checklist:
- `samples` should be around `62` for `250 ms @ 250 Hz`.
- If no edge output, check topic spelling and broker host/port.

### Optional file replay mode
- Put ECG values into a CSV/TXT file (see [data/README.md](data/README.md)).

```powershell
python .\src\replayer.py --mode file --file .\data\your_ecg.csv --fs 250 --chunk-ms 250 --loop
```

---

## Step 3 — Run viewer (dashboard simulator)

### Future target behavior
- Server/dashboard consumes edge features and events continuously.

### What we simulate now
- `viewer.py` subscribes to edge features and logs HR/SQI/notes.

### Why this step matters for later integration
- Confirms downstream contract and observability path for monitoring/alerting.

Open **Terminal 3**:
```powershell
cd "c:\Users\amria\OneDrive\Bureau\raspberry pi models\edge_mqtt_demo"
.\.venv\Scripts\Activate.ps1
python .\src\viewer.py
```

Expected output:
- `[viewer] connected to MQTT localhost:1883`
- `[viewer] subscribed to edge/patient1/features`
- Repeating lines like: `hr=71.8 sqi=0.92 notes=[]`

Optional live HR plot:
```powershell
python .\src\viewer.py --plot
```

Debug checklist:
- If viewer silent, ensure edge service prints published windows.
- If `hr=None`, check whether at least one full 5-second window was accumulated.

---

## Step 4 — Visualize signal processing (see before/after)

### Future target behavior
- Debug preprocessing on real sensor data to verify quality.

### What we simulate now
- Live visualization of raw vs processed ECG with R-peaks highlighted.

### Why this step matters for later integration
- Validates that filtering removes noise without distorting morphology.
- Shows exactly where R-peaks are detected.
- Instant feedback on HR calculation quality.

Open **Terminal 4** (or replace Terminal 3 with this if you want visualization instead of text viewer):
```powershell
cd "c:\Users\amria\OneDrive\Bureau\raspberry pi models\edge_mqtt_demo"
.\.venv\Scripts\Activate.ps1
python .\src\visualizer.py
```

Expected output:
- Three live-updating plots:
  1. **Raw ECG**: shows baseline drift + noise + powerline interference
  2. **Filtered ECG**: clean signal after detrend + bandpass + notch
  3. **Filtered + R-peaks**: red dots mark detected peaks + HR displayed

Debug checklist:
- Raw signal should show visible drift and 50 Hz noise.
- Filtered signal should be clean (P-QRS-T waves clear).
- R-peaks should align with highest points in QRS complex.
- HR displayed should match replayer `--hr-bpm` setting (±5%).

---

## Step 5 — Validate robustness behavior (important before hardware)

### Future target behavior
- Real devices can disconnect, jitter, resend, or deliver delayed packets.

### What we simulate now
- Replayer injects gaps and out-of-order chunks.

### Why this step matters for later integration
- Verifies edge gateway resilience under realistic transport imperfections.

Use in Terminal 2:
```powershell
python .\src\replayer.py --mode synthetic --chunk-ms 250 --inject-gap-every 25 --inject-oop-every 40 --duration-sec 180
```

Expected edge notes (Terminal 1):
- `gap_filled_zeros`
- `overlap_trimmed`
- `old_or_duplicate_chunk_dropped`
- `low_sqi` when stream quality degrades

---

## Processing details (minimum viable ECG pipeline)

### Windowing
- 5-second fixed windows (`window_sec=5`).
- If insufficient data for meaningful HR: `hr_bpm=null`, note includes `insufficient_data`.

### Preprocessing
- Baseline removal: linear detrend.
- Bandpass: Butterworth `0.5–40 Hz`.
- Optional notch: `50 Hz` (`--disable-notch` to disable).

### R-peak + HR
- Peak detector with refractory period (`~250 ms`) and prominence threshold.
- HR estimated from RR intervals in plausible range (`0.3s..2.0s`).

### SQI heuristic (0..1)
- Penalizes flatline, high high-frequency noise, and implausible peak density.
- Emits notes like `low_sqi` when quality is poor.

---

## Why these technology choices

- **MQTT + Mosquitto**: lightweight pub/sub, ideal for ESP32 and edge gateways.
- **Python 3.11+**: fast iteration and strong signal-processing ecosystem.
- **NumPy/SciPy**: reliable baseline filtering and peak detection primitives.
- **Message schema in JSON**: easy debugging now, replaceable with protobuf later if needed.
- **Three-process split**: mirrors future production boundaries (device/edge/backend).

---

## Plug-compatibility checklist for hardware arrival

1. Keep topic names + JSON keys unchanged.
2. ESP32 publisher sends same fields:
   - `patient_id`, `t0_ms`, `fs_hz`, `samples`.
3. Maintain chunk duration (200–500 ms).
4. On Raspberry Pi, run the same `edge_preprocessor.py` with broker host changed.
5. Keep output contract stable for dashboard/LLM-RAG.

---

## Commands quick reference

From project root with venv activated:

```powershell
# Terminal 1 (edge gateway)
python .\src\edge_preprocessor.py --debug

# Terminal 2 (text viewer)
python .\src\viewer.py

# Terminal 3 (visual signal viewer)
python .\src\visualizer.py

# Terminal 4 (publisher)
python .\src\replayer.py --mode synthetic --fs 250 --chunk-ms 250 --duration-sec 120 --hr-bpm 72
```

Stop with `Ctrl+C` in each terminal.

---

## One-click start on Windows (.bat)

If you want to launch everything in the correct order automatically:

```powershell
cd "c:\Users\amria\OneDrive\Bureau\raspberry pi models\edge_mqtt_demo"
.\run_all.bat
```

What it does:
- Starts Mosquitto with `docker compose up -d`.
- Launches **4 windows** in sequence:
  1. **Edge Preprocessor** — processes ECG, publishes features
  2. **Viewer (text)** — logs HR/SQI/notes to console
  3. **Visualizer (graphs)** — live plots of raw/filtered ECG + R-peaks
  4. **Replayer** — publishes synthetic ECG chunks (250 ms each)

Notes:
- It auto-detects Python in `edge_mqtt_demo\\.venv` or parent `.venv`.
- To stop the broker after testing: `docker compose down`.
- All 4 windows run simultaneously so you can see text logs AND live graphs.

### Alternative: Visual signal analysis only

If you want **just the visualizer** (no edge preprocessor or text viewer):

```powershell
.\run_visualizer.bat
```

What it does:
- Starts Mosquitto broker.
- Launches `visualizer.py` (shows 3 live-updating plots).
- Launches `replayer.py` (publishes ECG data).

Expected output: A matplotlib window with 3 subplots updating every 500ms showing:
1. Raw ECG with baseline drift + noise
2. Filtered ECG (clean)
3. Filtered ECG + detected R-peaks + calculated HR

---

## Next extension points (after this MVP)

- Add PPG/SpO2/Temp topics and shared timestamp alignment.
- Add edge local persistence (SQLite) for offline buffering.
- Publish compact `edge_summary` JSON for LLM/RAG ingestion.
- Add unit tests for windowing, SQI, and R-peak edge cases.
