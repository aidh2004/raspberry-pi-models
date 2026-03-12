# edge_mqtt_demo — Smart Medical Mat Edge AI Gateway

Hardware-free digital twin for a smart medical mat pipeline with **full Edge AI capabilities**:

- **Publisher (ESP32 simulator)**: replays or synthesizes multi-sensor data (ECG, PPG, SpO2, Temperature, IMU) and publishes MQTT chunks.
- **Edge AI Gateway (Raspberry Pi simulator)**: subscribes, windows, preprocesses signals, extracts clinical features, evaluates clinical rules, runs ML inference (Logistic Regression, Random Forest, XGBoost), and fuses results into severity/color/action decisions.
- **Viewer (server/dashboard simulator)**: subscribes to features and prints/plots.
  - **Text viewer**: logs features, events, and clinical decisions to console.
  - **Visual viewer**: live graphs showing raw/filtered ECG + R-peaks.

Plug-compatible with future ESP32 + Raspberry Pi deployment.

---

## Edge AI Pipeline Overview

The gateway implements a **6-step pipeline** for each 5-second sensor window:

```
Sensor Data (MQTT)
       │
       ▼
┌─────────────────────┐
│ 1. Feature Extraction│  18 signal-processing formulas
│    (feature_extraction│  HR, HRV, SpO2, Temp, Motion,
│     .py)             │  Respiration, SQI
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. Clinical Rules   │  7 threshold-based rules
│    (clinical_rules   │  (tachycardia, hypoxemia, fever…)
│     .py)             │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. ML Inference     │  3 models: LogReg, RF, XGBoost
│    (ml_inference.py) │  → risk, class, deterioration
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. Decision Fusion  │  Rules + ML → severity/color/action
│    (decision_engine  │
│     .py)             │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. Publish          │  MQTT features + events
│ 6. Local Alert      │  Console color-coded severity
└─────────────────────┘
```

### ML Models

| Model | Task | Output |
|-------|------|--------|
| **Logistic Regression** | Binary risk scoring | `risk_score` (0–1) |
| **Random Forest** (50 trees) | 6-class event classification | `event_class` (normal, tachycardia, bradycardia, hypoxemia, fever, apnea_risk) |
| **XGBoost** (50 trees) | Deterioration prediction | `deterioration_prob` (0–1) |

### Clinical Rules

| Rule | Threshold | Severity |
|------|-----------|----------|
| SpO2 < 90% | Critical hypoxemia | CRITICAL |
| HR > 120 bpm | Tachycardia | HIGH |
| HR < 45 bpm | Bradycardia | HIGH |
| Temp > 38°C | Fever | MODERATE |
| Desat index >= 5/hr | Apnea suspicion | MODERATE–HIGH |
| SQI < 0.4 | Low signal quality | LOW–MODERATE |
| Motion score > 0.5 | Motion artifact | MODERATE–HIGH |

### Decision Color Map

| Severity | Color | Action |
|----------|-------|--------|
| LOW | Green | surveillance_standard |
| MODERATE | Yellow | verification_demandee |
| HIGH | Orange | intervention_recommandee |
| CRITICAL | Red | alerte_immediate |

---

## Architecture and Future Mapping

### Future target behavior
- ESP32 acquires ECG, PPG, SpO2, Temp, IMU and publishes MQTT chunks with stable schema/topic.
- Raspberry Pi runs `edge_preprocessor.py` as Edge AI Gateway with ML inference.
- Gateway outputs features, clinical events, and severity decisions to server/dashboard/LLM-RAG.

### What we simulate now
- Laptop runs all roles:
  1. `replayer.py` instead of ESP32.
  2. `edge_preprocessor.py` instead of Raspberry Pi (with full AI pipeline).
  3. `viewer.py` + `visualizer.py` instead of cloud/dashboard.
- Mosquitto runs locally as broker.
- ML models trained on synthetic clinical data (6 scenarios, 2000 samples).

### Why this matters for later integration
- Keeps **message contracts** fixed from day one.
- Validates **feature extraction, clinical rules, and ML inference** before hardware arrives.
- Minimizes integration risk to "replace publisher + move edge service to Raspberry + update env vars".
- Same `.joblib` model files deploy directly to Raspberry Pi.

---

## Project Structure

```
edge_mqtt_demo/
├── src/
│   ├── common.py                # Shared types, configs, message contracts
│   ├── feature_extraction.py    # 18 signal formulas (HR, HRV, SpO2, etc.)
│   ├── clinical_rules.py        # 7 clinical threshold rules
│   ├── ml_inference.py          # 3 ML models (LogReg, RF, XGBoost)
│   ├── decision_engine.py       # Rules + ML fusion → severity/color/action
│   ├── edge_preprocessor.py     # Main Edge AI Gateway (MQTT pipeline)
│   ├── replayer.py              # ESP32 simulator (multi-sensor publisher)
│   ├── viewer.py                # Text dashboard (console)
│   └── visualizer.py            # Visual dashboard (matplotlib plots)
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
├── data/
│   └── README.md
├── test_pipeline.py             # Integration test (6 clinical scenarios)
├── requirements.txt
├── docker-compose.yml
├── run_all.bat
├── run_visualizer.bat
├── README.md
└── DOCUMENTATION.md
```

---

## MQTT Contracts

### Input Topics (sensor data)

| Topic | Sensor | Rate |
|-------|--------|------|
| `sim/patient1/ecg` | ECG | 250 Hz chunks |
| `sim/patient1/ppg` | PPG | 100 Hz chunks |
| `sim/patient1/imu` | Accelerometer | 50 Hz triplets |
| `sim/patient1/spo2` | SpO2 | 1 Hz samples |
| `sim/patient1/temp` | Temperature | 0.2 Hz samples |

### Output Topics

| Topic | Content |
|-------|---------|
| `edge/patient1/features` | Extracted features + ML results + decision |
| `edge/patient1/events` | Clinical events (alerts, warnings) |

### Output JSON Schema (features)
```json
{
  "patient_id": "patient1",
  "window_start_ms": 1700000000000,
  "window_sec": 5,
  "ecg": {
    "hr_mean": 72.4, "hr_min": 70.1, "hr_max": 74.8,
    "hrv_sdnn_ms": 45.2, "hrv_rmssd_ms": 38.1,
    "qrs_duration_ms": 92.0, "abnormal_beats": 0, "sqi": 0.91
  },
  "ppg": { "pulse_rate": 72.1, "amplitude": 0.45, "ptt_ms": 150.0, "sqi": 0.88 },
  "spo2": { "mean": 97.2, "min": 96.0, "desaturation_count": 0 },
  "temp": { "mean": 36.8, "variation": 0.1, "fever": false },
  "motion": { "mvt_count": 0, "immobility": true, "agitation_index": 0.0, "score": 0.0 },
  "respiration": { "rate_bpm": 15.2, "amplitude": 0.12 },
  "rules": { "triggered": [], "severity": "low" },
  "ml": {
    "risk_score": 0.06, "event_class": "normal", "deterioration_prob": 0.03
  },
  "decision": { "severity": "low", "color": "green", "action": "surveillance_standard" }
}
```

---

## Quick Start

### Step 0 — Prerequisites

1. Install **Python 3.11+** from python.org.
2. Install **Docker Desktop** (for Mosquitto) or native Mosquitto.
3. Open PowerShell in project root.

### Create virtual environment
```powershell
cd "c:\Users\amria\OneDrive\Bureau\raspberry pi models\edge_mqtt_demo"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies installed: `paho-mqtt`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `xgboost`, `joblib`.

### Start Mosquitto broker
```powershell
docker compose up -d
```

---

### Step 1 — Run Edge AI Gateway

Open **Terminal 1**:
```powershell
cd "c:\Users\amria\OneDrive\Bureau\raspberry pi models\edge_mqtt_demo"
.\.venv\Scripts\Activate.ps1
python .\src\edge_preprocessor.py --debug
```

Expected output:
- `[edge] connected to MQTT localhost:1883`
- `[edge] subscribed to sim/patient1/ecg`
- On first run: models auto-train and save to `models/` directory.

---

### Step 2 — Run ECG Replayer (ESP32 simulator)

Open **Terminal 2**:
```powershell
python .\src\replayer.py --mode synthetic --fs 250 --chunk-ms 250 --duration-sec 120 --hr-bpm 72
```

---

### Step 3 — Run Viewer (dashboard)

Open **Terminal 3**:
```powershell
python .\src\viewer.py
```

Optional live HR plot:
```powershell
python .\src\viewer.py --plot
```

---

### Step 4 — Visualize Signal Processing

Open **Terminal 4**:
```powershell
python .\src\visualizer.py
```

Shows three live-updating plots:
1. **Raw ECG**: baseline drift + noise + powerline interference
2. **Filtered ECG**: clean signal after detrend + bandpass + notch
3. **Filtered + R-peaks**: red dots on detected peaks + HR display

---

### One-Click Start

```powershell
.\run_all.bat
```

Starts broker + edge gateway + viewer + visualizer + replayer in separate windows.

---

## Testing

### Integration Test (no MQTT needed)

```powershell
python test_pipeline.py
```

Runs 6 clinical scenarios end-to-end:

| Scenario | Expected Rules | Expected ML Class | Expected Decision |
|----------|---------------|-------------------|-------------------|
| Normal patient | none | normal | GREEN |
| Tachycardia (HR=130) | tachycardia_high | tachycardia | RED |
| Bradycardia (HR=40) | bradycardia | bradycardia | YELLOW–RED |
| Hypoxemia (SpO2=85%) | spo2_critical | hypoxemia | RED |
| Fever (39°C) | fever | fever | ORANGE |
| Multi-alarm | spo2_critical + tachy + fever | hypoxemia | RED |

### Robustness Test (gaps + out-of-order)

```powershell
python .\src\replayer.py --mode synthetic --chunk-ms 250 --inject-gap-every 25 --inject-oop-every 40 --duration-sec 180
```

---

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.11+** | Core language (signal processing, ML, MQTT) |
| **NumPy / SciPy** | Signal filtering, peak detection, detrending |
| **scikit-learn** | Logistic Regression, Random Forest classifiers |
| **XGBoost** | Gradient-boosted deterioration classifier |
| **joblib** | Model serialization (.joblib files) |
| **paho-mqtt** | MQTT client for pub/sub communication |
| **Mosquitto** | MQTT broker (Docker or native) |
| **Matplotlib** | Real-time signal visualization |

---

## Plug-Compatibility Checklist for Hardware

1. Keep topic names + JSON keys unchanged.
2. ESP32 publishes same fields: `patient_id`, `t0_ms`, `fs_hz`, `samples`.
3. On Raspberry Pi, run same `edge_preprocessor.py` with `--broker-host <pi_ip>`.
4. Copy `models/` directory to Raspberry Pi — same `.joblib` files work directly.
5. Install same `requirements.txt` on Raspberry Pi.

---

## Next Extension Points

- Replace synthetic training data with real patient datasets.
- Add edge local persistence (SQLite) for offline buffering.
- Publish compact `edge_summary` JSON for LLM/RAG ingestion.
- Add TLS/authentication to MQTT for production deployment.
- Update viewer/visualizer to display ML results and decision colors.
