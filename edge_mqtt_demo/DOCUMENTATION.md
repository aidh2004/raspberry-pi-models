# Edge AI Gateway — Technical Documentation

> Smart Medical Mat: Edge Intelligence for Continuous Patient Monitoring
>
> This document covers the complete AI pipeline — from raw sensor signals to clinical
> decisions — designed to serve as the reference for the AI chapter of the project report.

---

## Table of Contents

1. [Introduction & Objectives](#1-introduction--objectives)
2. [System Architecture](#2-system-architecture)
3. [Development Phases](#3-development-phases)
4. [Technology Stack](#4-technology-stack)
5. [Communication Layer — MQTT Protocol](#5-communication-layer--mqtt-protocol)
6. [Data Acquisition — Sensor Simulation](#6-data-acquisition--sensor-simulation)
7. [Edge AI Pipeline — 6-Step Processing](#7-edge-ai-pipeline--6-step-processing)
   - 7.1 [Signal Preprocessing & Feature Extraction](#71-signal-preprocessing--feature-extraction)
   - 7.2 [Clinical Rules Engine](#72-clinical-rules-engine)
   - 7.3 [Machine Learning Inference](#73-machine-learning-inference)
   - 7.4 [Decision Fusion Engine](#74-decision-fusion-engine)
   - 7.5 [Publishing & Alerting](#75-publishing--alerting)
8. [Real-Time Visualization Dashboard](#8-real-time-visualization-dashboard)
9. [Data Logging (CSV)](#9-data-logging-csv)
10. [Testing & Validation Results](#10-testing--validation-results)
11. [Data Flow — Complete Example](#11-data-flow--complete-example)
12. [Project Files & Module Map](#12-project-files--module-map)
13. [Deployment — From Simulation to Real Hardware](#13-deployment--from-simulation-to-real-hardware)

---

## 1. Introduction & Objectives

### 1.1 What is this project?

This project implements a **complete Edge AI Gateway** for a smart medical mat system. It
processes multi-sensor biomedical data in real time, extracts clinical features, applies
medical rules, runs machine learning inference, and produces severity-graded clinical
decisions — all at the edge (Raspberry Pi), without requiring cloud connectivity.

### 1.2 Project goals

| Goal | Description |
|------|-------------|
| **Validate the AI pipeline** | Prove that feature extraction, clinical rules, ML models, and decision fusion work correctly end-to-end |
| **Simulate before hardware** | Build a hardware-free digital twin that runs entirely on a development PC |
| **Plug-compatible design** | Fixed MQTT topics and JSON schemas — when real sensors arrive, only the data source changes |
| **Edge-first architecture** | All AI processing runs locally on constrained hardware (Raspberry Pi), no cloud dependency |
| **Clinical-grade output** | Produce severity levels (LOW → CRITICAL) with color-coded actions for medical staff |

### 1.3 Key results achieved

- **18 signal-processing formulas** implemented (ECG, PPG, SpO2, Temperature, IMU, Respiration)
- **7 clinical rules** based on medical thresholds
- **3 machine learning models** trained and validated (Logistic Regression, Random Forest, XGBoost)
- **Decision fusion engine** combining rules + ML with priority logic
- **Single-window real-time dashboard** with 3 navigable pages
- **CSV data logging** (33 columns per 5-second window)
- **6 clinical scenarios** tested and validated

---

## 2. System Architecture

### 2.1 Overall architecture diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DEVELOPMENT PC (Simulation)                         │
│                                                                              │
│  ┌──────────────┐                                      ┌──────────────────┐ │
│  │  replayer.py  │    sim/patient1/{ecg,ppg,spo2,      │  visualizer.py   │ │
│  │  (ESP32 sim.) │    temp,imu}                        │  (Dashboard)     │ │
│  │               │─────────┐                           │                  │ │
│  │  Publishes:   │         │                           │  3 pages:        │ │
│  │  - ECG 250 Hz │         ▼                           │  1. Raw Inputs   │ │
│  │  - PPG 100 Hz │   ┌───────────┐  edge/patient1/    │  2. Processed    │ │
│  │  - IMU  50 Hz │   │ Mosquitto │  features + events  │  3. AI Features  │ │
│  │  - SpO2  1 Hz │   │  Broker   │────────────────────▶│                  │ │
│  │  - Temp 0.2Hz │   │ :1883     │                     └──────────────────┘ │
│  └──────────────┘   └─────┬─────┘                                           │
│                           │ sim/patient1/*                                   │
│                           ▼                                                  │
│              ┌─────────────────────────────────────┐                        │
│              │    Edge Preprocessor (Raspi sim.)    │                        │
│              │                                     │                        │
│              │  Step 1. Feature Extraction (18)     │                        │
│              │  Step 2. Clinical Rules (7)          │                        │
│              │  Step 3. ML Inference (3 models)     │                        │
│              │  Step 4. Decision Fusion             │                        │
│              │  Step 5. MQTT Publish                │                        │
│              │  Step 6. Local Alert + CSV Log       │                        │
│              │                                     │                        │
│              │  Models:                            │                        │
│              │  ├── logistic_regression.joblib      │                        │
│              │  ├── random_forest.joblib            │                        │
│              │  └── xgboost.joblib                  │                        │
│              └─────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component roles

| Component | Simulates | Role | Key Technologies |
|-----------|-----------|------|-----------------|
| **replayer.py** | ESP32 sensor node | Generates & publishes synthetic multi-sensor data via MQTT | NumPy, paho-mqtt |
| **edge_preprocessor.py** | Raspberry Pi Gateway | Full 6-step AI pipeline: extract → rules → ML → decide → publish → log | SciPy, scikit-learn, XGBoost |
| **visualizer.py** | Cloud dashboard | Single-window real-time display with 3 switchable pages | Matplotlib, paho-mqtt |
| **Mosquitto broker** | Network infrastructure | Routes MQTT messages between all components | Eclipse Mosquitto (Docker) |

### 2.3 Design principles

1. **Plug-compatible**: Fixed MQTT topics and JSON schemas from day one — swap the data source and the pipeline works unchanged.
2. **Stateless services**: Each component can restart independently without data loss.
3. **Portable models**: ML models saved as `.joblib` files — train on laptop, deploy on Raspberry Pi.
4. **Edge-first**: No internet required; all processing happens locally.

---

## 3. Development Phases

This section describes the phases followed to build the complete AI pipeline, useful for
structuring the project report.

### Phase 1 — Communication Infrastructure

**Objective**: Establish the MQTT messaging backbone.

- Set up Eclipse Mosquitto broker via Docker (`docker-compose.yml`)
- Defined MQTT topic hierarchy: `sim/{patient_id}/{sensor}` for raw data, `edge/{patient_id}/{features|events}` for processed output
- Defined JSON message schemas for each sensor type (chunk messages for high-rate sensors, sample messages for low-rate sensors)
- Implemented shared configuration module (`common.py`) with enums, validation functions, and MQTT settings

### Phase 2 — Sensor Simulation (Data Acquisition Layer)

**Objective**: Generate realistic multi-sensor biomedical data.

- Built `replayer.py` as ESP32 digital twin
- Implemented synthetic signal generation for 5 sensor types:
  - **ECG** (250 Hz): P-QRS-T morphology with baseline wander, 50 Hz powerline interference, and Gaussian noise
  - **PPG** (100 Hz): Systolic peak + dicrotic notch with baseline variation
  - **IMU** (50 Hz): 3-axis accelerometer with gravity vector and optional motion bursts
  - **SpO2** (1 Hz): Oxygen saturation with optional desaturation events
  - **Temperature** (0.2 Hz): Random walk around 37°C baseline
- Added robustness testing features: gap injection, out-of-order packets, speed control

### Phase 3 — Signal Processing & Feature Extraction

**Objective**: Extract clinically meaningful features from raw signals.

- Implemented `feature_extraction.py` with **18 signal-processing formulas**
- ECG processing chain: detrend → Butterworth bandpass (0.5–40 Hz) → 50 Hz notch filter → R-peak detection → HR/HRV/QRS computation
- PPG processing: bandpass (0.5–8 Hz) → peak detection → pulse rate + PTT
- SpO2/Temperature/IMU: statistical features + clinical thresholds
- Respiration rate derived from ECG envelope modulation via FFT
- Implemented signal quality index (SQI) for both ECG and PPG

### Phase 4 — Clinical Rules Engine

**Objective**: Apply medical knowledge as deterministic threshold rules.

- Implemented `clinical_rules.py` with **7 clinical rules** based on medical standards
- Rules cover: hypoxemia, tachycardia, bradycardia, fever, apnea suspicion, signal quality, motion artifacts
- Each rule produces a severity level (LOW → CRITICAL) and event details
- All rules evaluate independently (no short-circuit) — multiple alarms can fire simultaneously

### Phase 5 — Machine Learning Models

**Objective**: Add predictive intelligence beyond fixed thresholds.

- Implemented `ml_inference.py` with **3 complementary ML models**:
  - **Logistic Regression** → binary risk score (0–1)
  - **Random Forest** (50 trees) → 6-class event classification
  - **XGBoost** (50 trees) → deterioration probability (0–1)
- Built a **19-feature vector** from all sensor domains
- Generated **2000 synthetic training samples** across 6 clinical scenarios
- Models auto-train on first run and save as `.joblib` files
- Training accuracies: LR = 94.6%, RF = 97.8%, XGB = 96.4%

### Phase 6 — Decision Fusion

**Objective**: Combine rules and ML into a single clinical decision.

- Implemented `decision_engine.py` with priority-based fusion logic
- Rules severity serves as baseline; ML can only **escalate**, never downgrade
- Output: severity level + color code + recommended action
- Mapping: LOW/Green → routine | MODERATE/Yellow → verify | HIGH/Orange → intervene | CRITICAL/Red → immediate alert

### Phase 7 — Integration & Data Pipeline

**Objective**: Wire all AI modules into the edge gateway.

- `edge_preprocessor.py` orchestrates the complete 6-step pipeline
- Buffering system handles multi-rate sensor data (5-second non-overlapping windows)
- CSV logging captures all 33 feature columns per window for offline analysis
- Results published via MQTT and printed to console with color-coded severity

### Phase 8 — Visualization & Optimization

**Objective**: Build a real-time monitoring dashboard.

- Implemented `visualizer.py` as a **single-window dashboard with 3 switchable pages**
- Optimized for low resource usage: only the active page redraws each tick (600 ms interval)
- Single MQTT connection instead of multiple, minimal GPU/memory footprint
- Page navigation via clickable buttons at the bottom of the window

---

## 4. Technology Stack

### 4.1 Dependencies

| Library | Version | Role in the project |
|---------|---------|-------------------|
| **Python** | 3.11+ | Core language — runs identically on laptop and Raspberry Pi |
| **NumPy** | ≥ 1.24 | Numerical arrays for signal buffers, feature computation (mean, std, FFT) |
| **SciPy** | ≥ 1.10 | Signal processing: Butterworth filters, notch filter, peak detection, detrend |
| **scikit-learn** | ≥ 1.3 | ML models: Logistic Regression, Random Forest, label encoding, train/test split |
| **XGBoost** | ≥ 2.0 | Gradient-boosted trees for deterioration prediction |
| **joblib** | ≥ 1.3 | Model serialization (`.joblib` files) — portable across platforms |
| **paho-mqtt** | ≥ 1.6.1 | MQTT client with auto-reconnect, QoS=1, publish/subscribe callbacks |
| **Matplotlib** | ≥ 3.7 | Real-time dashboard with `FuncAnimation`, `GridSpec`, `Button` widgets |
| **Mosquitto** | 2.x | MQTT broker — runs via Docker (dev) or native install (production) |
| **Docker** | — | Container for Mosquitto broker during development (one-command setup) |

### 4.2 Why these choices?

- **SciPy** was chosen over custom filters because it provides clinically validated Butterworth and IIR filter implementations, critical for biomedical signal processing.
- **scikit-learn + XGBoost** provide a range of model complexity (linear → ensemble → boosted) suitable for an edge device with limited RAM.
- **MQTT** is the standard IoT messaging protocol — lightweight, supports publish/subscribe, works on constrained networks, and is the default for medical IoT.
- **Docker** is used only for the MQTT broker during development — it provides instant, reproducible setup with `docker-compose up -d`. In production on Raspberry Pi, Mosquitto is installed natively (`sudo apt install mosquitto`).

---

## 5. Communication Layer — MQTT Protocol

### 5.1 What is MQTT?

MQTT (Message Queuing Telemetry Transport) is a lightweight publish/subscribe messaging 
protocol designed for IoT devices. It uses a central **broker** to route messages between
**publishers** (sensors) and **subscribers** (processors, dashboards).

### 5.2 Why MQTT for this project?

| Advantage | Explanation |
|-----------|-------------|
| **Lightweight** | Minimal overhead — runs on ESP32 (2 KB RAM for client) and Raspberry Pi |
| **Publish/Subscribe** | Decouples producers from consumers — add or remove components without changing others |
| **Topic-based routing** | Hierarchical topics (`sim/patient1/ecg`) provide clean message organization |
| **QoS levels** | Guarantees message delivery (QoS=1 used in this project) |
| **Standard protocol** | Supported by all IoT platforms (AWS IoT, Azure, HiveMQ, Mosquitto) |

### 5.3 Broker setup

The MQTT broker (Eclipse Mosquitto) runs via Docker during development:

```yaml
# docker-compose.yml
services:
  mosquitto:
    image: eclipse-mosquitto:2
    container_name: edge-mqtt-demo-broker
    ports:
      - "1883:1883"    # MQTT protocol
      - "9001:9001"    # WebSocket (optional)
    restart: unless-stopped
```

**Start**: `docker-compose up -d`
**Stop**: `docker-compose down`

### 5.4 Topic hierarchy

```
sim/{patient_id}/ecg       ← Raw ECG chunks (250 Hz)
sim/{patient_id}/ppg       ← Raw PPG chunks (100 Hz)
sim/{patient_id}/imu       ← Raw IMU triplets (50 Hz)
sim/{patient_id}/spo2      ← SpO2 samples (1 Hz)
sim/{patient_id}/temp      ← Temperature samples (0.2 Hz)

edge/{patient_id}/features ← Processed features (every 5 seconds)
edge/{patient_id}/events   ← Clinical events (on trigger)
```

### 5.5 Message formats

**High-rate sensor chunk** (ECG, PPG):
```json
{
  "patient_id": "patient1",
  "t_ms": 1000,
  "fs_hz": 250,
  "samples": [0.12, -0.05, 0.87, ...]
}
```

**Low-rate sensor sample** (SpO2, Temp):
```json
{
  "patient_id": "patient1",
  "t_ms": 5000,
  "value": 97.2
}
```

**IMU chunk** (3-axis accelerometer):
```json
{
  "patient_id": "patient1",
  "t_ms": 1000,
  "fs_hz": 50,
  "samples": [[0.01, -0.02, 9.81], ...]
}
```

### 5.6 Docker vs native MQTT

| Environment | Broker Setup | Why |
|-------------|-------------|-----|
| **Development (PC)** | Docker: `docker-compose up -d` | One command, isolated, easy cleanup |
| **Production (Raspberry Pi)** | Native: `sudo apt install mosquitto` | Lower overhead, no Docker layer on ARM |
| **Cloud deployment** | Managed service (AWS IoT, HiveMQ) | No broker to maintain |

---

## 6. Data Acquisition — Sensor Simulation

### 6.1 Simulated sensors

The `replayer.py` module acts as a digital twin of the ESP32 sensor node, generating
realistic synthetic signals for 5 sensor types:

| Sensor | Sampling Rate | Signal Characteristics |
|--------|--------------|----------------------|
| **ECG** | 250 Hz | P-QRS-T morphology, baseline wander, 50 Hz powerline noise, Gaussian noise |
| **PPG** (Photoplethysmography) | 100 Hz | Systolic peak, dicrotic notch, diastolic component, baseline variation |
| **IMU** (Accelerometer 3-axis) | 50 Hz | Gravity on Z-axis (~1g), optional motion bursts, Gaussian noise |
| **SpO2** (Pulse Oximetry) | 1 Hz | Baseline ~98%, optional desaturation drop events with recovery |
| **Temperature** | 0.2 Hz | Random walk around 37°C, configurable drift and noise |

### 6.2 Synthetic ECG generation

```
For each heartbeat at interval 60/HR_bpm:
  R-peak   : Gaussian pulse (amplitude=1.0, width=0.01s)
  P-wave   : Gaussian pulse (amplitude=0.15, 160ms before R)
  T-wave   : Gaussian pulse (amplitude=0.3, 200ms after R)

Added artifacts:
  + Baseline drift  : low-frequency sinusoidal wander
  + Powerline noise : 50 Hz sinusoidal interference
  + Random noise    : Gaussian (configurable std)
```

### 6.3 Publishing behavior

- Each sensor publishes at its native chunk rate (ECG: every 250 ms, SpO2: every 1 s)
- Real-time pacing with optional speed multiplier (`--speed 2.0`)
- Separate threads per sensor stream
- Optional gap injection and out-of-order packet simulation for robustness testing

---

## 7. Edge AI Pipeline — 6-Step Processing

The core of the AI system. Every **5 seconds**, the Edge Preprocessor collects a window of
multi-sensor data and processes it through 6 sequential steps:

```
 Raw sensor data (MQTT)
         │
         ▼
 ┌───────────────────────┐
 │ Step 1: Feature       │  18 signal-processing formulas
 │ Extraction            │  ECG, PPG, SpO2, Temp, IMU, Respiration
 └──────────┬────────────┘
            ▼
 ┌───────────────────────┐
 │ Step 2: Clinical      │  7 threshold-based rules
 │ Rules                 │  Medical standards → severity
 └──────────┬────────────┘
            ▼
 ┌───────────────────────┐
 │ Step 3: ML            │  3 models: LogReg + RF + XGBoost
 │ Inference             │  19-feature vector → predictions
 └──────────┬────────────┘
            ▼
 ┌───────────────────────┐
 │ Step 4: Decision      │  Rules + ML → final severity
 │ Fusion                │  Priority: rules baseline, ML escalates
 └──────────┬────────────┘
            ▼
 ┌───────────────────────┐
 │ Step 5: Publish       │  MQTT features + events
 │ Step 6: Alert + Log   │  Console output + CSV file
 └───────────────────────┘
```

### 7.1 Signal Preprocessing & Feature Extraction

**Module**: `feature_extraction.py` — Implements **18 formulas** from the medical specifications.

#### 7.1.1 ECG Processing Chain

```
Raw ECG (250 Hz, 1250 samples per window)
    │
    ├── scipy.signal.detrend()           → Remove baseline drift
    ├── Butterworth bandpass (0.5–40 Hz, order 3)  → Remove out-of-band noise
    ├── IIR notch filter (50 Hz, Q=30)   → Remove powerline interference
    │
    ▼
Filtered ECG
    │
    ├── scipy.signal.find_peaks()        → Detect R-peaks
    │     (adaptive prominence = 35% of std, min distance = 250 ms)
    │
    ▼
R-peak positions
    │
    ├── RR intervals → HR mean/min/max (bpm)
    ├── RR intervals → HRV SDNN = std(RR) × 1000 (ms)
    ├── RR intervals → HRV RMSSD = sqrt(mean(diff(RR)²)) × 1000 (ms)
    ├── Peak width   → QRS duration (ms), clamped 40–200 ms
    ├── RR outliers  → Abnormal beat count (> 1.5σ from mean)
    └── Quality      → ECG SQI (0–1), penalizes flatline/noise/bad peak density
```

#### 7.1.2 PPG Processing

```
Raw PPG (100 Hz, 500 samples per window)
    │
    ├── Butterworth bandpass (0.5–8 Hz)  → Clean signal
    ├── Peak detection (0.4s refractory) → PPG peaks
    │
    ▼
    ├── Pulse rate = 60 / peak_interval (bpm)
    ├── Amplitude = mean(peak − trough) (mV)
    ├── PTT = median(ECG_R → next_PPG_peak) delay (ms)
    └── PPG SQI (0–1)
```

#### 7.1.3 Other Sensor Features

| Sensor | Features Extracted | Method |
|--------|-------------------|--------|
| **SpO2** | Mean SpO2 (%), Min SpO2 (%), Desaturation count | Statistical analysis, 3% drop detection |
| **Temperature** | Mean (°C), Variation (max−min), Fever flag (>38°C) | Statistical thresholds |
| **IMU** | Movement count, Immobility flag, Agitation index, Motion score (0–1) | Magnitude threshold, composite scoring |
| **Respiration** | Rate (breaths/min), Amplitude | ECG envelope modulation via FFT, filtered to 6–30 breaths/min |

#### 7.1.4 Complete feature summary

| # | Feature | Source | Unit | Range |
|---|---------|--------|------|-------|
| 1 | HR mean | ECG | bpm | 30–200 |
| 2 | HR min | ECG | bpm | 30–200 |
| 3 | HR max | ECG | bpm | 30–200 |
| 4 | HRV SDNN | ECG | ms | 0–300 |
| 5 | HRV RMSSD | ECG | ms | 0–300 |
| 6 | RR mean | ECG | ms | 300–2000 |
| 7 | QRS duration | ECG | ms | 40–200 |
| 8 | ECG SQI | ECG | — | 0–1 |
| 9 | Pulse rate | PPG | bpm | 30–200 |
| 10 | PPG amplitude | PPG | mV | 0–5 |
| 11 | PPG SQI | PPG | — | 0–1 |
| 12 | SpO2 mean | SpO2 | % | 70–100 |
| 13 | SpO2 min | SpO2 | % | 70–100 |
| 14 | Desaturation count | SpO2 | count | 0–50 |
| 15 | Temperature mean | Temp | °C | 34–42 |
| 16 | Temperature variation | Temp | °C | 0–5 |
| 17 | Motion score | IMU | — | 0–1 |
| 18 | Agitation index | IMU | — | 0–10 |
| 19 | Respiration rate | ECG-derived | breaths/min | 6–30 |

---

### 7.2 Clinical Rules Engine

**Module**: `clinical_rules.py` — Implements **7 deterministic rules** based on medical standards.

| # | Rule | Condition | Severity | Event Name |
|---|------|-----------|----------|-----------|
| 1 | **Critical hypoxemia** | SpO2 < 90% | CRITICAL | spo2_critical |
| 2 | **Tachycardia** | HR > 120 bpm | HIGH | tachycardia_high |
|   |                 | HR > 100 bpm | MODERATE | tachycardia_moderate |
| 3 | **Bradycardia** | HR < 45 bpm | HIGH | bradycardia_high |
|   |                 | HR < 60 bpm | MODERATE | bradycardia_moderate |
| 4 | **Fever** | Temp > 38°C | MODERATE | fever |
| 5 | **Apnea suspicion** | Desat index ≥ 15/hr | HIGH | apnea_suspicion |
|   |                     | Desat index ≥ 5/hr | MODERATE | apnea_suspicion |
| 6 | **Low signal quality** | Average SQI < 0.4 | LOW–MODERATE | low_signal_quality |
| 7 | **Motion artifact** | Motion score > 0.5 | MODERATE–HIGH | motion_artifact |

**Behavior**:
- All 7 rules are evaluated independently (no short-circuit)
- Multiple rules can trigger simultaneously (e.g., tachycardia + fever)
- The maximum severity across all triggered rules becomes the rules-based severity
- Each triggered rule generates an event with `patient_id`, `timestamp`, `event_type`, `severity`, and `details`

---

### 7.3 Machine Learning Inference

**Module**: `ml_inference.py` — Implements **3 complementary ML models** via class `MLInferenceEngine`.

#### 7.3.1 Why 3 models?

Each model answers a different clinical question:

| Model | Algorithm | Clinical Question | Output |
|-------|-----------|------------------|--------|
| **Logistic Regression** | LogisticRegression | Is this patient at risk? | `risk_score` (0–1) |
| **Random Forest** | 50 trees, max_depth=8 | What type of event is occurring? | `event_class` (6 categories) |
| **XGBoost** | 50 trees, max_depth=5 | Is the patient deteriorating? | `deterioration_prob` (0–1) |

#### 7.3.2 Input: 19-feature vector

All 3 models receive the same standardized feature vector:

```
[ hr_mean, hr_min, hr_max, hrv_sdnn, hrv_rmssd, rr_mean_ms,
  qrs_duration_ms, ecg_sqi,
  pulse_rate, ppg_amplitude, ppg_sqi,
  spo2_mean, spo2_min, desaturation_count,
  temp_mean, temp_variation,
  motion_score, agitation_index,
  respiration_rate ]
```

Missing or `None` values are replaced with safe defaults (0.0 for most, 1.0 for SQI fields).

#### 7.3.3 Training data — 6 clinical scenarios

Models are trained on **2000 synthetic samples** with realistic clinical distributions:

| Scenario | HR (bpm) | SpO2 (%) | Temp (°C) | Class Label |
|----------|----------|----------|-----------|-------------|
| **Normal** | 60–80 | 95–100 | 36.0–37.5 | normal |
| **Tachycardia** | 120–180 | 93–98 | 36.5–37.5 | tachycardia |
| **Bradycardia** | 30–50 | 93–98 | 36.0–37.0 | bradycardia |
| **Hypoxemia** | 80–110 | 70–89 | 36.5–37.5 | hypoxemia |
| **Fever** | 85–120 | 93–98 | 38.5–41.0 | fever |
| **Apnea risk** | 60–90 | 85–93 | 36.0–37.5 | apnea_risk |

#### 7.3.4 Training results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | **94.6%** | Binary: low-risk vs high-risk |
| Random Forest | **97.8%** | 6-class classification |
| XGBoost | **96.4%** | Binary: stable vs deteriorating |

#### 7.3.5 Model persistence

- Models are saved as `.joblib` files in the `models/` directory
- On first run, if no `.joblib` files exist, models train automatically and save
- Same `.joblib` files work on any platform (laptop → Raspberry Pi)
- To retrain: delete `.joblib` files and restart the edge preprocessor

---

### 7.4 Decision Fusion Engine

**Module**: `decision_engine.py` — Combines rules and ML into a **single clinical decision**.

#### 7.4.1 Fusion logic

```
1. Start with rules_severity (from clinical rules engine)

2. Check ML escalation:
   if risk_score > 0.9       → escalate to CRITICAL
   if risk_score > 0.7       → escalate to HIGH
   if risk_score > 0.5       → escalate to MODERATE
   if deterioration_prob > 0.8 → escalate to HIGH
   if deterioration_prob > 0.6 → escalate to MODERATE

3. Final severity = max(rules_severity, ml_severity)

KEY PRINCIPLE: ML can only ESCALATE, never downgrade.
Rules-based severity is always respected as the minimum.
```

#### 7.4.2 Output mapping

| Severity Level | Color Code | Recommended Action | Clinical Meaning |
|---------------|------------|-------------------|-----------------|
| **LOW** | 🟢 Green | `surveillance_standard` | Patient stable, routine monitoring |
| **MODERATE** | 🟡 Yellow | `verification_demandee` | Abnormality detected, manual verification needed |
| **HIGH** | 🟠 Orange | `intervention_recommandee` | Clinical intervention recommended |
| **CRITICAL** | 🔴 Red | `alerte_immediate` | Immediate medical attention required |

---

### 7.5 Publishing & Alerting

After decision fusion, the gateway:

1. **Publishes features** to MQTT topic `edge/{patient_id}/features` — JSON containing all extracted features, rules results, ML predictions, and the final decision
2. **Publishes events** to MQTT topic `edge/{patient_id}/events` — one message per triggered clinical rule
3. **Prints to console** with color-coded severity (green/yellow/orange/red)
4. **Logs to CSV** file in `data/logs/` (33 columns per row, one row per 5-second window)

---

## 8. Real-Time Visualization Dashboard

**Module**: `visualizer.py` — Single-window dashboard with 3 switchable pages.

### 8.1 Architecture

- **Single matplotlib figure** with `GridSpec(3, 2)` layout (6 panels per page)
- **3 navigation buttons** at the bottom of the window for page switching
- **Only the active page redraws** each animation tick (600 ms interval) — saves GPU/CPU
- **Single MQTT client** subscribes to `edge/{patient_id}/features` and `edge/{patient_id}/events`
- Thread-safe data buffers (`Buffers` class) shared between MQTT callback and animation loop

### 8.2 Pages

| Page | Name | Content (6 panels) |
|------|------|-------------------|
| **Page 1** | Raw Inputs | ECG raw waveform, PPG raw waveform, SpO2 readings, Temperature readings, IMU 3-axis plot, IMU magnitude |
| **Page 2** | Processed | Filtered ECG + R-peaks + HR, Filtered PPG + peaks + pulse rate, SpO2 trend + desaturation, Temperature + fever threshold, Motion score, Respiration rate |
| **Page 3** | AI Features | Vital signs banner (HR, SpO2, Temp, Resp), HRV trends (SDNN/RMSSD), SQI metrics, ML predictions (risk/deterioration/class), Decision output + clinical rules |

### 8.3 Performance optimizations

| Optimization | Effect |
|-------------|--------|
| Single window (vs 3 separate windows) | ~75% less GPU/memory usage |
| Only active page redraws | Idle pages consume zero rendering time |
| 600 ms update interval | Smooth enough for monitoring, light on CPU |
| Single MQTT connection | Reduced network and threading overhead |

---

## 9. Data Logging (CSV)

The `CsvFeatureLogger` class in `edge_preprocessor.py` records **33 columns** per 5-second
window to `data/logs/features_YYYYMMDD_HHMMSS.csv`:

**Column categories**:
- Patient metadata (4): timestamp, patient_id, window_start, window_end
- ECG features (8): hr_mean, hr_min, hr_max, hrv_sdnn, hrv_rmssd, rr_mean, qrs_duration, ecg_sqi
- PPG features (4): pulse_rate, ppg_amplitude, ptt, ppg_sqi
- SpO2 features (3): spo2_mean, spo2_min, desaturation_count
- Temperature (3): temp_mean, temp_variation, fever_flag
- Motion (4): motion_score, movement_count, immobility, agitation_index
- Respiration (2): resp_rate, resp_amplitude
- Rules (2): triggered_rules, rules_severity
- ML predictions (3): risk_score, event_class, deterioration_prob
- Decision (3): severity, color, action

---

## 10. Testing & Validation Results

### 10.1 Integration test (offline, no MQTT needed)

```powershell
python test_pipeline.py
```

Tests the complete AI pipeline directly in Python without requiring the MQTT broker.

**6 clinical scenarios validated:**

| # | Scenario | HR | SpO2 | Temp | Rules Triggered | ML Class | Decision |
|---|----------|-----|------|------|----------------|----------|----------|
| 1 | Normal patient | 75 bpm | 97% | 36.8°C | none | normal | 🟢 GREEN |
| 2 | Tachycardia | 130 bpm | 96% | 37°C | tachycardia_high | tachycardia | 🔴 RED |
| 3 | Bradycardia | 40 bpm | 96% | 36.5°C | bradycardia | bradycardia | 🟡–🔴 |
| 4 | Hypoxemia | 90 bpm | 85% | 37°C | spo2_critical | hypoxemia | 🔴 RED |
| 5 | Fever | 95 bpm | 96.5% | 39°C | fever | fever | 🟠 ORANGE |
| 6 | Multi-alarm | 135 bpm | 84% | 39.4°C | spo2_critical + tachy + fever | hypoxemia | 🔴 RED |

**Example test output:**
```
SCENARIO: Normal patient
  ECG:  HR_mean=75.0 HRV_SDNN=0.0 SQI=0.7
  SpO2: mean=97.2 min=96.0
  Rules: severity=low triggered=[]
  ML: risk=0.061 deterioration=0.027 class=normal
  DECISION: GREEN → surveillance_standard

SCENARIO: Hypoxemia (SpO2=85%)
  ECG:  HR_mean=90.05 HRV_SDNN=2.14 SQI=0.7
  SpO2: mean=85.2 min=83.0
  Rules: severity=critical triggered=['spo2_critical']
  ML: risk=1.0 deterioration=0.992 class=hypoxemia
  DECISION: RED → alerte_immediate
```

### 10.2 End-to-End MQTT test

```powershell
# Start broker
docker-compose up -d

# Terminal 1: Edge AI Gateway
python src\edge_preprocessor.py --debug

# Terminal 2: Sensor simulator
python src\replayer.py --mode synthetic --hr-bpm 72 --duration-sec 120

# Terminal 3: Dashboard
python src\visualizer.py
```

**Expected results**:
- Features published every 5 seconds
- HR ≈ 72 bpm (±5%)
- Decision = GREEN for normal conditions

### 10.3 Robustness testing

```powershell
# Test with gaps and out-of-order packets
python src\replayer.py --mode synthetic --inject-gap-every 25 --inject-oop-every 40

# Test different clinical conditions
python src\replayer.py --hr-bpm 130   # Tachycardia → expect RED
python src\replayer.py --hr-bpm 40    # Bradycardia → expect ORANGE/RED
```

---

## 11. Data Flow — Complete Example

### 11.1 Normal operation timeline

```
T=0s     System starts
         ├── Mosquitto broker listening on localhost:1883
         ├── Edge Preprocessor connects, loads 3 ML models
         ├── Visualizer connects to features/events topics
         └── Replayer begins generating sensor data

T=0.25s  First ECG chunk published (62 samples at 250 Hz)
         └── Edge buffers: 62/1250 = 5% of window filled

T=1s     Multiple sensors active
         ├── ECG: 250 samples buffered
         ├── PPG: 100 samples buffered
         ├── SpO2: 1 sample received
         └── Temp: 0 samples yet (0.2 Hz rate)

T=5s     ★ First 5-second window complete — AI pipeline runs:
         │
         ├── Feature Extraction
         │   ├── ECG (1250 samples): HR=72.0, HRV_SDNN=45ms, QRS=92ms, SQI=0.91
         │   ├── PPG (500 samples): pulse_rate=72.1, PTT=150ms
         │   ├── SpO2 (5 samples): mean=97.2, min=96.0
         │   ├── Temp (1 sample): mean=36.8, fever=false
         │   ├── IMU (250 triplets): motion_score=0.0, immobility=true
         │   └── Respiration: 15 breaths/min (from ECG envelope)
         │
         ├── Clinical Rules → no rules triggered, severity=LOW
         ├── ML Inference → risk=0.06, class=normal, deterioration=0.03
         ├── Decision Fusion → LOW → 🟢 GREEN → surveillance_standard
         │
         └── Output: MQTT publish + console + CSV log

T=10s    Second window processed...
T=15s    Third window processed...
```

### 11.2 Alarm scenario — Hypoxemia detected

```
T=30s    SpO2 drops to 85%
         │
         ├── Feature Extraction
         │   └── spo2_mean=85.2, spo2_min=83.0
         │
         ├── Clinical Rules
         │   └── Rule 1 fires: SpO2 < 90% → CRITICAL (spo2_critical)
         │
         ├── ML Inference
         │   ├── risk_score = 1.0
         │   ├── event_class = "hypoxemia"
         │   └── deterioration_prob = 0.99
         │
         ├── Decision Fusion
         │   ├── Rules: CRITICAL (baseline)
         │   ├── ML: risk > 0.9 → CRITICAL (no further escalation needed)
         │   └── Final: CRITICAL → 🔴 RED → alerte_immediate
         │
         └── Output:
             ├── MQTT: features + event published
             ├── Console: RED alerte_immediate [spo2_critical]
             └── CSV: row logged with all values
```

---

## 12. Project Files & Module Map

### 12.1 File structure

```
edge_mqtt_demo/
├── docker-compose.yml          ← Mosquitto MQTT broker configuration
├── requirements.txt            ← Python dependencies (7 packages)
├── run_all.bat                 ← One-click launcher (broker + 3 services)
├── DOCUMENTATION.md            ← This document
├── README.md                   ← Project overview
│
├── src/
│   ├── common.py               ← Shared types, MQTT config, validation
│   ├── feature_extraction.py   ← 18 signal-processing formulas
│   ├── clinical_rules.py       ← 7 clinical threshold rules
│   ├── ml_inference.py         ← 3 ML models (LR + RF + XGB)
│   ├── decision_engine.py      ← Rules + ML fusion logic
│   ├── edge_preprocessor.py    ← Main Edge AI Gateway + CSV logger
│   ├── replayer.py             ← ESP32 sensor simulator
│   ├── visualizer.py           ← Single-window 3-page dashboard
│   └── viewer.py               ← Legacy console/plot viewer (optional)
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
│
└── data/
    └── logs/                   ← CSV feature logs (auto-created)
```

### 12.2 Module dependency graph

```
common.py ◄──────────────── All modules depend on this
    ▲
    │
    ├── feature_extraction.py ◄─┐
    ├── clinical_rules.py     ◄─┤
    ├── ml_inference.py       ◄─┼── edge_preprocessor.py
    └── decision_engine.py    ◄─┘
                                          │
                                          ▼ MQTT
replayer.py ──── MQTT ────► edge_preprocessor.py ──── MQTT ────► visualizer.py
(sensor data)              (AI processing)                      (dashboard)
```

### 12.3 Running the system

**One-click launch** (Windows):
```powershell
run_all.bat
```

This starts 4 components in order:
1. Mosquitto broker (Docker)
2. Edge Preprocessor (separate window, `--debug` mode)
3. Visualizer dashboard (separate window)
4. Replayer (separate window, 5 minutes of synthetic data at 72 bpm)

**Manual launch** (any OS):
```powershell
docker-compose up -d                                          # 1. Broker
python src/edge_preprocessor.py --debug                       # 2. Gateway
python src/visualizer.py                                      # 3. Dashboard
python src/replayer.py --mode synthetic --duration-sec 300     # 4. Sensors
```

---

## 13. Deployment — From Simulation to Real Hardware

### 13.1 What changes in production?

```
SIMULATION (current):
  replayer.py ──► MQTT (Docker) ──► edge_preprocessor.py ──► visualizer.py
  [PC]             [PC]             [PC]                     [PC]

PRODUCTION (target):
  Real sensors ──► MQTT (native) ──► edge_preprocessor.py ──► Web dashboard
  [ESP32]          [Raspberry Pi]    [Raspberry Pi]           [Server/Cloud]
```

| Component | Simulation | Production |
|-----------|-----------|------------|
| Data source | `replayer.py` (synthetic) | Real sensors (MAX30102, AD8232, MPU6050) on ESP32 |
| MQTT broker | Mosquitto via Docker | Mosquitto native (`sudo apt install mosquitto`) |
| AI Gateway | `edge_preprocessor.py` on PC | Same code on Raspberry Pi |
| ML models | `.joblib` trained on synthetic data | Same files, or retrained on real patient data |
| Dashboard | `visualizer.py` (matplotlib) | Web dashboard or cloud platform |

### 13.2 Deploying to Raspberry Pi

```bash
# Transfer project to Pi
scp -r edge_mqtt_demo pi@192.168.1.100:/home/pi/

# On the Pi
ssh pi@192.168.1.100
cd /home/pi/edge_mqtt_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install native MQTT broker (no Docker needed)
sudo apt install mosquitto mosquitto-clients

# Launch Edge AI Gateway
python src/edge_preprocessor.py --broker-host 192.168.1.100
```

### 13.3 Retraining models with real data

When real patient data becomes available:
1. Replace `_generate_synthetic_data()` in `ml_inference.py` with real dataset loading
2. Delete existing `.joblib` files in `models/`
3. Restart `edge_preprocessor.py` — models retrain automatically and save new `.joblib` files

### 13.4 Production security (future)

- Add TLS encryption to MQTT (certificates in Mosquitto config)
- Add username/password authentication to MQTT broker
- Add `systemd` service for auto-start at boot on Raspberry Pi
