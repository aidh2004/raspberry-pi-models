# Multi-Sensor Extension Documentation

## Overview

This document describes the multi-sensor extension to the edge_mqtt_demo pipeline. The system now supports:

- **ECG** (Electrocardiogram) - 250 Hz, high-rate chunk stream
- **PPG** (Photoplethysmogram) - 100 Hz, high-rate chunk stream
- **IMU** (Accelerometer) - 50 Hz, high-rate triplet chunk stream
- **SpO2** (Oxygen Saturation) - 1 Hz, low-rate sample stream
- **Temperature** - 0.2 Hz, low-rate sample stream

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-SENSOR PIPELINE                                │
│                                                                              │
│   ESP32 / Replayer                 Raspberry Pi / Edge              Cloud   │
│   ────────────────                 ─────────────────────           ─────── │
│                                                                              │
│   ┌─────────────┐                  ┌─────────────────┐                      │
│   │  ECG Sensor │──┐               │                 │    ┌───────────────┐ │
│   │  (250 Hz)   │  │               │                 │    │               │ │
│   └─────────────┘  │               │                 │    │   Viewer.py   │ │
│                    │               │                 │──► │   Dashboard   │ │
│   ┌─────────────┐  │               │  Edge           │    │               │ │
│   │  PPG Sensor │──┼──► MQTT ──►   │  Preprocessor  │    │  - HR (ECG)   │ │
│   │  (100 Hz)   │  │               │                 │    │  - HR (PPG)   │ │
│   └─────────────┘  │               │  - Windowing    │    │  - SpO2       │ │
│                    │               │  - Filtering    │    │  - Temp       │ │
│   ┌─────────────┐  │               │  - Features     │    │  - Motion     │ │
│   │  IMU Sensor │──┤               │  - Events       │    │  - Events     │ │
│   │  (50 Hz)    │  │               │                 │    │               │ │
│   └─────────────┘  │               └────────┬────────┘    └───────────────┘ │
│                    │                        │                               │
│   ┌─────────────┐  │                        │                               │
│   │  SpO2       │──┤               ┌────────┴────────┐                      │
│   │  (1 Hz)     │  │               │   Output Topics │                      │
│   └─────────────┘  │               │                 │                      │
│                    │               │ edge/patient1/  │                      │
│   ┌─────────────┐  │               │   features      │                      │
│   │  Temp       │──┘               │   events        │                      │
│   │  (0.2 Hz)   │                  │                 │                      │
│   └─────────────┘                  └─────────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MQTT Topics

### Input Topics (from ESP32/Simulator)

| Topic | Sensor | Rate | Message Type |
|-------|--------|------|--------------|
| `sim/patient1/ecg` | ECG | 250 Hz | Chunk |
| `sim/patient1/ppg` | PPG | 100 Hz | Chunk |
| `sim/patient1/imu` | IMU | 50 Hz | Triplet Chunk |
| `sim/patient1/spo2` | SpO2 | 1 Hz | Sample |
| `sim/patient1/temp` | Temperature | 0.2 Hz | Sample |

### Output Topics (from Edge Gateway)

| Topic | Description | Rate |
|-------|-------------|------|
| `edge/patient1/features` | Unified sensor features | Every 5 seconds |
| `edge/patient1/events` | Alerts and events | On condition |

---

## Message Schemas

### Chunk Message (ECG, PPG)

High-rate sensor data sent in chunks (typically 250ms worth).

```json
{
  "patient_id": "patient1",
  "t0_ms": 1704067200000,
  "fs_hz": 250,
  "samples": [0.123, 0.456, 0.789, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | string | Patient/device identifier |
| `t0_ms` | int | Start timestamp (epoch milliseconds) |
| `fs_hz` | int | Sampling frequency in Hz |
| `samples` | float[] | Array of sample values |

### IMU Chunk Message

IMU data with [ax, ay, az] triplets.

```json
{
  "patient_id": "patient1",
  "t0_ms": 1704067200000,
  "fs_hz": 50,
  "samples": [[0.01, 0.02, 1.0], [0.01, 0.01, 1.0], ...]
}
```

### Sample Message (SpO2, Temperature)

Low-rate sensor data sent as individual samples.

```json
{
  "patient_id": "patient1",
  "t_ms": 1704067200000,
  "value": 98.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | string | Patient/device identifier |
| `t_ms` | int | Timestamp (epoch milliseconds) |
| `value` | float | Sensor reading |

### Features Message

Unified features output, published every 5 seconds.

```json
{
  "patient_id": "patient1",
  "window_start_ms": 1704067200000,
  "window_sec": 5,
  "ecg": {
    "hr_bpm": 72.5,
    "sqi": 0.95,
    "notes": []
  },
  "ppg": {
    "hr_bpm": 73.0,
    "sqi": 0.90,
    "notes": []
  },
  "spo2": {
    "mean": 98.2,
    "min": 97.5,
    "drops": 0
  },
  "temp_c": {
    "mean": 37.1
  },
  "motion": {
    "score": 0.05,
    "notes": []
  }
}
```

### Events Message

Alerts and events, published on conditions.

```json
{
  "patient_id": "patient1",
  "t_ms": 1704067200000,
  "type": "spo2_drop",
  "severity": "moderate",
  "details": {
    "min_value": 91.5,
    "duration_sec": 12
  }
}
```

| Event Type | Trigger |
|------------|---------|
| `spo2_drop` | SpO2 < 93% for ≥10 seconds |
| `tachycardia` | HR > 100 bpm |
| `bradycardia` | HR < 50 bpm |
| `low_quality` | Average SQI < 0.4 |
| `motion_artifact` | Motion score > 0.5 |
| `sensor_detached` | No data received (future) |

| Severity | Description |
|----------|-------------|
| `low` | Minor concern, monitoring |
| `moderate` | Attention needed |
| `high` | Urgent, immediate attention |

---

## Feature Extraction Deep Dive

This section explains exactly how each feature is computed from raw sensor data.
All processing happens in `edge_preprocessor.py` on the Raspberry Pi (or simulation).

### Overview: From Raw Data to Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION PIPELINE                          │
│                                                                             │
│   Raw Sensor Data          Processing Steps              Output Features    │
│   ───────────────          ────────────────              ───────────────    │
│                                                                             │
│   ECG (1250 samples) ──► Filter ──► R-Peaks ──► HR, SQI                    │
│   PPG (500 samples)  ──► Filter ──► Peaks   ──► HR, SQI                    │
│   IMU (250 triplets) ──► Magnitude ──► Variance ──► Motion Score           │
│   SpO2 (~5 samples)  ──► Statistics ──► Threshold ──► Mean, Min, Drops     │
│   Temp (~1 sample)   ──► Average ──► Sanity Check ──► Mean                 │
│                                                                             │
│   All features combined into single JSON message every 5 seconds            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. ECG Feature Extraction

**Input**: 1250 raw samples (250 Hz × 5 seconds)

**Step 1: Signal Preprocessing**

```
Raw ECG Signal
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ DETRENDING                                                                  │
│ ──────────                                                                  │
│ Problem: ECG has slow baseline wander from breathing, movement              │
│ Solution: Fit a line to the signal and subtract it                          │
│                                                                             │
│ Before: Signal drifts up and down slowly                                    │
│ After:  Signal centered around zero                                         │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ BANDPASS FILTER (0.5 - 40 Hz)                                               │
│ ─────────────────────────────                                               │
│ Why 0.5 Hz low cutoff?  Removes baseline drift (breathing ~0.2 Hz)          │
│ Why 40 Hz high cutoff?  Keeps QRS complex, removes muscle noise             │
│                                                                             │
│ Implementation: 3rd order Butterworth filter (smooth response)              │
│ Code: butter(3, [0.5/nyq, 40/nyq], btype="band")                           │
│       filtfilt(b, a, signal)  # Zero-phase filtering                        │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ NOTCH FILTER (50 Hz)                                                        │
│ ────────────────────                                                        │
│ Problem: Powerline interference at 50 Hz (or 60 Hz in USA)                  │
│ Solution: Narrow notch filter removes only 50 Hz                            │
│                                                                             │
│ Implementation: IIR notch filter with Q=30 (narrow)                         │
│ Code: iirnotch(w0=50/nyq, Q=30)                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Step 2: R-Peak Detection**

```
Filtered ECG Signal
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ R-PEAK DETECTION ALGORITHM                                                  │
│ ──────────────────────────                                                  │
│                                                                             │
│ The R-peak is the tallest spike in each heartbeat (QRS complex)             │
│                                                                             │
│        R                     R                     R                        │
│        ▲                     ▲                     ▲                        │
│       /│\                   /│\                   /│\                       │
│      / │ \                 / │ \                 / │ \                      │
│   P /  │  \ S    T      P /  │  \ S    T      P /  │  \ S                   │
│    ▲   │   ▼    ▲        ▲   │   ▼    ▲        ▲   │   ▼                    │
│ ──/────┼────\──/──\────/────┼────\──/──\────/────┼────\───                  │
│   Q    │     \/                   │                   │                     │
│        │                          │                   │                     │
│   ◄────┼──────────────────────────┼───────────────────┤                     │
│        │      R-R interval        │   R-R interval    │                     │
│                                                                             │
│ Algorithm:                                                                  │
│ 1. Calculate signal standard deviation (STD)                                │
│ 2. Set threshold = 0.35 × STD (peaks must be prominent)                     │
│ 3. Set minimum distance = 0.25 seconds (max 240 bpm)                        │
│ 4. Find all local maxima above threshold                                    │
│ 5. Keep only peaks separated by minimum distance                            │
│                                                                             │
│ Code: find_peaks(filtered, distance=min_distance, prominence=threshold)     │
└────────────────────────────────────────────────────────────────────────────┘
```

**Step 3: Heart Rate Calculation**

```
R-Peak Indices: [125, 333, 541, 749, 957]  (sample numbers)
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ HEART RATE CALCULATION                                                      │
│ ──────────────────────                                                      │
│                                                                             │
│ Step 1: Calculate R-R intervals (difference between consecutive peaks)      │
│         intervals_samples = diff([125, 333, 541, 749, 957])                 │
│                          = [208, 208, 208, 208]                             │
│                                                                             │
│ Step 2: Convert to seconds (divide by sampling rate)                        │
│         intervals_sec = [208, 208, 208, 208] / 250 Hz                       │
│                      = [0.832, 0.832, 0.832, 0.832] seconds                 │
│                                                                             │
│ Step 3: Filter out invalid intervals                                        │
│         Keep only intervals between 0.3s and 2.0s                           │
│         (corresponds to 30-200 bpm - physiological range)                   │
│                                                                             │
│ Step 4: Calculate HR from mean interval                                     │
│         mean_interval = 0.832 seconds                                       │
│         HR = 60 / mean_interval = 60 / 0.832 = 72.1 bpm                    │
│                                                                             │
│ Formula: HR (bpm) = 60 / mean(R-R intervals in seconds)                     │
└────────────────────────────────────────────────────────────────────────────┘
```

**Step 4: Signal Quality Index (SQI)**

```
Raw + Filtered Signal + R-Peaks
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ SIGNAL QUALITY INDEX (SQI) CALCULATION                                      │
│ ──────────────────────────────────────                                      │
│                                                                             │
│ SQI measures how reliable the ECG signal is (0 = bad, 1 = excellent)        │
│                                                                             │
│ Start with: score = 1.0                                                     │
│                                                                             │
│ Check 1: FLAT SIGNAL                                                        │
│          IF standard_deviation(raw) < 0.0001:                               │
│             score -= 0.8  (sensor disconnected or dead)                     │
│                                                                             │
│ Check 2: EXCESSIVE FLAT SEGMENTS                                            │
│          Count samples where consecutive difference < 0.00001               │
│          IF more than 20% of signal is flat:                                │
│             score -= 0.4  (poor contact or saturation)                      │
│                                                                             │
│ Check 3: HIGH NOISE                                                         │
│          noise = raw - filtered  (high-frequency component)                 │
│          noise_ratio = std(noise) / std(filtered)                           │
│          IF noise_ratio > 1.0:                                              │
│             score -= 0.3  (very noisy)                                      │
│          ELSE IF noise_ratio > 0.7:                                         │
│             score -= 0.15 (moderately noisy)                                │
│                                                                             │
│ Check 4: ABNORMAL PEAK COUNT                                                │
│          Expected peaks in 5s: 3-15 (for 40-180 bpm)                        │
│          IF peaks < 3 OR peaks > 15:                                        │
│             score -= 0.2  (unrealistic HR)                                  │
│                                                                             │
│ Final: SQI = max(0.0, min(1.0, score))                                      │
│                                                                             │
│ Examples:                                                                   │
│   Clean signal, 6 peaks: SQI = 1.0 - 0 - 0 - 0 - 0 = 1.0                   │
│   Noisy signal, 6 peaks: SQI = 1.0 - 0 - 0 - 0.3 - 0 = 0.7                 │
│   Flat signal:           SQI = 1.0 - 0.8 - 0.4 - 0 - 0.2 = 0.0             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. PPG Feature Extraction

**Input**: 500 raw samples (100 Hz × 5 seconds)

```
Raw PPG Signal (photoplethysmogram from pulse oximeter)
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ PPG SIGNAL CHARACTERISTICS                                                  │
│ ──────────────────────────                                                  │
│                                                                             │
│ PPG measures blood volume changes using light absorption                    │
│ Each heartbeat creates a characteristic waveform:                           │
│                                                                             │
│     Systolic Peak                                                           │
│          ▲                                                                  │
│         /│\                                                                 │
│        / │ \    Dicrotic                                                    │
│       /  │  \    Notch                                                      │
│      /   │   \    ▼                                                         │
│     /    │    \__/\                                                         │
│    /     │         \____ Diastolic                                          │
│ __/      │                                                                  │
│          │                                                                  │
│     ◄────┼─────────────────►                                                │
│          │   One heartbeat                                                  │
│                                                                             │
│ We detect systolic peaks (highest point) similar to ECG R-peaks             │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ BANDPASS FILTER (0.5 - 8 Hz)                                                │
│ ───────────────────────────                                                 │
│                                                                             │
│ Why narrower than ECG (0.5-40 Hz)?                                          │
│ - PPG waveform is slower, smoother than ECG                                 │
│ - No sharp QRS complex to preserve                                          │
│ - 8 Hz upper limit still captures waveform shape                            │
│                                                                             │
│ Implementation: 2nd order Butterworth (gentler than ECG)                    │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ PEAK DETECTION & HR CALCULATION                                             │
│ ───────────────────────────────                                             │
│                                                                             │
│ Same algorithm as ECG R-peak detection                                      │
│ - Find local maxima (systolic peaks)                                        │
│ - Calculate intervals between peaks                                         │
│ - HR = 60 / mean(peak intervals)                                            │
│                                                                             │
│ PPG HR should closely match ECG HR (cross-validation)                       │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ PPG SIGNAL QUALITY INDEX                                                    │
│ ────────────────────────                                                    │
│                                                                             │
│ Similar to ECG SQI but with different considerations:                       │
│                                                                             │
│ Check 1: Flat signal → score -= 0.8                                         │
│                                                                             │
│ Check 2: Peak interval regularity (PPG-specific)                            │
│          intervals = diff(peak_indices)                                     │
│          coefficient_of_variation = std(intervals) / mean(intervals)        │
│          IF CV > 0.3:                                                       │
│             score -= 0.3  (irregular peaks, possible motion artifact)       │
│                                                                             │
│ Check 3: Noise ratio → score -= 0.3 if noisy                                │
│                                                                             │
│ PPG is MORE sensitive to motion than ECG (finger movement)                  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. IMU / Motion Score Extraction

**Input**: 250 triplets (50 Hz × 5 seconds), each triplet is [ax, ay, az] in g-force

```
IMU Raw Data: [[0.01, 0.02, 1.0], [0.02, 0.01, 0.99], ...]
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ UNDERSTANDING IMU DATA                                                      │
│ ──────────────────────                                                      │
│                                                                             │
│ Accelerometer measures acceleration in 3 axes:                              │
│   ax = left/right acceleration                                              │
│   ay = forward/backward acceleration                                        │
│   az = up/down acceleration (includes gravity!)                             │
│                                                                             │
│ When device is lying flat and still:                                        │
│   ax ≈ 0, ay ≈ 0, az ≈ 1.0 (1g = gravity)                                  │
│                                                                             │
│ When patient moves:                                                         │
│   Values deviate from this baseline                                         │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: COMPUTE MAGNITUDE                                                   │
│ ─────────────────────────                                                   │
│                                                                             │
│ For each sample, compute total acceleration magnitude:                      │
│                                                                             │
│   magnitude = √(ax² + ay² + az²)                                           │
│                                                                             │
│ Example:                                                                    │
│   Sample: [0.01, 0.02, 1.0]                                                │
│   magnitude = √(0.0001 + 0.0004 + 1.0) = √1.0005 ≈ 1.0003                  │
│                                                                             │
│ Still patient: all magnitudes ≈ 1.0 (just gravity)                          │
│ Moving patient: magnitudes vary around 1.0                                  │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: REMOVE GRAVITY BASELINE                                             │
│ ───────────────────────────────                                             │
│                                                                             │
│   detrended = magnitude - 1.0                                               │
│                                                                             │
│ Still patient:  detrended values ≈ 0                                        │
│ Moving patient: detrended values fluctuate around 0                         │
│                                                                             │
│ Example (5 samples):                                                        │
│   magnitudes = [1.0003, 1.0001, 1.5, 0.8, 1.0002]                           │
│   detrended  = [0.0003, 0.0001, 0.5, -0.2, 0.0002]                          │
│                  ▲▲▲ still ▲▲▲   ▲▲ moving ▲▲                              │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: COMPUTE MOTION SCORE                                                │
│ ────────────────────────────                                                │
│                                                                             │
│   variance = var(detrended)  # How much values spread from mean            │
│   motion_score = variance / 0.5  # Normalize to 0-1 scale                  │
│   motion_score = clip(motion_score, 0.0, 1.0)                              │
│                                                                             │
│ Thresholds (empirically determined):                                        │
│   variance < 0.01  →  score ≈ 0.02  →  "Patient is still"                  │
│   variance = 0.25  →  score = 0.5   →  "Moderate motion"                   │
│   variance ≥ 0.5   →  score = 1.0   →  "High motion"                       │
│                                                                             │
│ Example:                                                                    │
│   Still patient:  variance = 0.001, score = 0.002                          │
│   Walking:        variance = 0.3,   score = 0.6                            │
│   Seizure/fall:   variance = 0.8,   score = 1.0                            │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ MOTION AFFECTS OTHER SENSORS                                                │
│ ────────────────────────────                                                │
│                                                                             │
│ When motion_score > 0.5:                                                    │
│   - Add "motion_artifact" note to ECG features                              │
│   - Add "motion_artifact" note to PPG features                              │
│   - Reduce ECG SQI by 0.2                                                   │
│   - Reduce PPG SQI by 0.3 (more sensitive to motion)                        │
│   - Generate "motion_artifact" event                                        │
│                                                                             │
│ This warns users that ECG/PPG readings may be unreliable during motion      │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 4. SpO2 Feature Extraction

**Input**: ~5 samples (1 Hz × 5 seconds), each sample is a percentage value

```
SpO2 Samples: [98.2, 97.8, 98.5, 98.1, 97.9]
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ UNDERSTANDING SpO2                                                          │
│ ──────────────────                                                          │
│                                                                             │
│ SpO2 = Peripheral Oxygen Saturation                                         │
│ Measures % of hemoglobin carrying oxygen                                    │
│                                                                             │
│ Normal values:                                                              │
│   95-100%: Normal                                                           │
│   90-94%:  Low (hypoxemia), needs attention                                │
│   <90%:    Severe hypoxemia, dangerous                                      │
│                                                                             │
│ In our system, SpO2 comes from MAX30102 sensor (on ESP32)                   │
│ Sensor publishes 1 reading per second                                       │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: WINDOW STATISTICS                                                   │
│ ─────────────────────────                                                   │
│                                                                             │
│ Collect all SpO2 samples within the 5-second window                         │
│                                                                             │
│   samples = [98.2, 97.8, 98.5, 98.1, 97.9]                                 │
│   mean = average(samples) = 98.1%                                           │
│   min  = minimum(samples) = 97.8%                                           │
│                                                                             │
│ Output:                                                                     │
│   "spo2": {                                                                 │
│     "mean": 98.1,                                                           │
│     "min": 97.8,                                                            │
│     "drops": 0                                                              │
│   }                                                                         │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: DROP DETECTION (Persistent State)                                   │
│ ─────────────────────────────────────────                                   │
│                                                                             │
│ Clinical definition of concerning SpO2 drop:                                │
│   Value < 93% sustained for >= 10 seconds                                   │
│                                                                             │
│ Algorithm (tracks state across windows):                                    │
│                                                                             │
│   IF any sample < 93%:                                                      │
│     below_time += number_of_low_samples  (accumulate)                       │
│     min_value = min(current_min, lowest_sample)                             │
│                                                                             │
│     IF below_time >= 10 seconds AND not already reported:                   │
│       Generate EVENT:                                                       │
│         type: "spo2_drop"                                                   │
│         severity: based on how low it dropped                               │
│           - min < 85%: "high"                                               │
│           - min < 90%: "moderate"                                           │
│           - else:      "low"                                                │
│         details: { min_value, duration_sec }                                │
│       drops_count += 1                                                      │
│       mark as reported (don't repeat)                                       │
│                                                                             │
│   ELSE (all samples >= 93%):                                                │
│     Reset tracking (patient recovered)                                      │
│     below_time = 0                                                          │
│     reported = false                                                        │
│                                                                             │
│ Example timeline:                                                           │
│   Window 1: [98, 97, 96] → all ≥93, no drop                                │
│   Window 2: [92, 91, 90] → below 93! below_time=3s                         │
│   Window 3: [89, 88, 87] → still low, below_time=6s                        │
│   Window 4: [86, 85, 84] → still low, below_time=9s                        │
│   Window 5: [83, 82, 81] → below_time=12s ≥10 → GENERATE EVENT!            │
│   Window 6: [95, 96, 97] → recovered, reset tracking                       │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 5. Temperature Feature Extraction

**Input**: ~1 sample (0.2 Hz × 5 seconds), value in Celsius

```
Temperature Samples: [37.1]
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ TEMPERATURE PROCESSING                                                      │
│ ──────────────────────                                                      │
│                                                                             │
│ Body temperature measured by sensor (e.g., MLX90614 infrared)               │
│ Normal range: 36.1°C - 37.2°C (97°F - 99°F)                                │
│                                                                             │
│ Processing:                                                                 │
│   1. Collect all temp samples in window                                     │
│   2. Calculate mean                                                         │
│   3. Sanity check: reject if outside 30-45°C                               │
│      (sensor error, disconnected, or impossible value)                      │
│                                                                             │
│ Output:                                                                     │
│   "temp_c": {                                                               │
│     "mean": 37.1                                                            │
│   }                                                                         │
│                                                                             │
│ If sanity check fails:                                                      │
│   "temp_c": {                                                               │
│     "mean": null,                                                           │
│     "notes": ["out_of_range"]                                               │
│   }                                                                         │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 6. Event Generation

Events are generated when specific conditions are detected:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ EVENT TYPES AND TRIGGERS                                                    │
│ ────────────────────────                                                    │
│                                                                             │
│ ┌─────────────────┬───────────────────────────────────────────────────────┐│
│ │ Event Type      │ Trigger Condition                                     ││
│ ├─────────────────┼───────────────────────────────────────────────────────┤│
│ │ tachycardia     │ HR > 100 bpm                                          ││
│ │                 │ Severity: low (100-120), moderate (120-150), high(>150)│
│ ├─────────────────┼───────────────────────────────────────────────────────┤│
│ │ bradycardia     │ HR < 50 bpm                                           ││
│ │                 │ Severity: low (45-50), moderate (40-45), high (<40)   ││
│ ├─────────────────┼───────────────────────────────────────────────────────┤│
│ │ spo2_drop       │ SpO2 < 93% for >= 10 seconds                          ││
│ │                 │ Severity: low (90-93%), moderate (85-90%), high (<85%)││
│ ├─────────────────┼───────────────────────────────────────────────────────┤│
│ │ low_quality     │ Average SQI (ECG + PPG) < 0.4                         ││
│ │                 │ Severity: low (0.2-0.4), moderate (<0.2)              ││
│ ├─────────────────┼───────────────────────────────────────────────────────┤│
│ │ motion_artifact │ Motion score > 0.5                                    ││
│ │                 │ Severity: moderate (0.5-0.8), high (>0.8)             ││
│ ├─────────────────┼───────────────────────────────────────────────────────┤│
│ │ sensor_detached │ No data received (future implementation)             ││
│ └─────────────────┴───────────────────────────────────────────────────────┘│
│                                                                             │
│ Events are published to: edge/patient1/events                               │
│ Format: { patient_id, t_ms, type, severity, details }                       │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### Summary: Feature Extraction Table

| Sensor | Samples/Window | Processing Steps | Output Features |
|--------|----------------|------------------|-----------------|
| **ECG** | 1250 (250Hz×5s) | Detrend → Bandpass 0.5-40Hz → Notch 50Hz → R-peak detection → RR intervals | `hr_bpm`, `sqi` |
| **PPG** | 500 (100Hz×5s) | Detrend → Bandpass 0.5-8Hz → Peak detection → intervals | `hr_bpm`, `sqi` |
| **IMU** | 250 (50Hz×5s) | Magnitude → Remove gravity → Variance | `motion_score` |
| **SpO2** | ~5 (1Hz×5s) | Statistics + threshold tracking | `mean`, `min`, `drops` |
| **Temp** | ~1 (0.2Hz×5s) | Average + sanity check | `mean` |

### Code Reference

All feature extraction code is in `src/edge_preprocessor.py`:

| Function | Lines | Purpose |
|----------|-------|---------|
| `preprocess_ecg()` | ~150-170 | ECG filtering |
| `preprocess_ppg()` | ~172-185 | PPG filtering |
| `detect_peaks()` | ~187-210 | R-peak / systolic peak detection |
| `estimate_hr_bpm()` | ~212-222 | HR from peak intervals |
| `compute_ecg_sqi()` | ~224-250 | ECG signal quality |
| `compute_ppg_sqi()` | ~252-270 | PPG signal quality |
| `compute_motion_score()` | ~272-295 | IMU motion analysis |
| `process_spo2_window()` | ~320-365 | SpO2 stats + drop detection |
| `process_temp_window()` | ~367-380 | Temperature processing |

---

## Running the Pipeline

### Quick Start

```powershell
# Start everything (broker + all services)
.\run_all.bat
```

### Manual Start

```powershell
# Terminal 1: Start MQTT broker
docker compose up -d

# Terminal 2: Start edge preprocessor
python src/edge_preprocessor.py --debug

# Terminal 3: Start viewer with plots
python src/viewer.py --plot

# Terminal 4: Start replayer (all sensors)
python src/replayer.py --mode synthetic --duration-sec 300
```

### Replayer Options

```powershell
# All sensors (default)
python src/replayer.py --mode synthetic --duration-sec 300

# ECG only (legacy mode)
python src/replayer.py --mode synthetic --ecg-only

# Custom HR
python src/replayer.py --mode synthetic --hr-bpm 80

# Inject SpO2 drop event
python src/replayer.py --mode synthetic --inject-spo2-drop

# Disable specific sensors
python src/replayer.py --no-imu --no-temp
```

### Viewer Options

```powershell
# Console only
python src/viewer.py

# With dashboard plots
python src/viewer.py --plot
```

---

## Code Structure

```
edge_mqtt_demo/
├── src/
│   ├── common.py           # Topics, schemas, validation, utilities
│   ├── replayer.py         # Multi-sensor synthetic data generator
│   ├── edge_preprocessor.py # Multi-sensor processing pipeline
│   ├── viewer.py           # Multi-sensor dashboard viewer
│   └── visualizer.py       # ECG waveform visualization
├── docker-compose.yml      # Mosquitto MQTT broker
├── run_all.bat             # Start everything
├── run_visualizer.bat      # Start visualizer only
├── requirements.txt        # Python dependencies
└── DOCUMENTATION.md        # Original documentation
```

---

## Hardware Integration Guide

When real ESP32 hardware is ready:

### 1. Change Topic Prefix

Replace `sim/` with `mat/` in the ESP32 firmware:

```cpp
// ESP32 firmware
#define MQTT_TOPIC_ECG "mat/patient1/ecg"
#define MQTT_TOPIC_PPG "mat/patient1/ppg"
// etc.
```

### 2. Update Edge Preprocessor

Edit `common.py` to use `mat/` prefix:

```python
TOPIC_ECG_RAW = "mat/{device_id}/ecg"
# etc.
```

### 3. Deploy to Raspberry Pi

1. Copy `src/` folder to Raspberry Pi
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python edge_preprocessor.py --broker-host <mqtt-broker-ip>`

### 4. Sensor-Specific ESP32 Code

```cpp
// ECG (MAX30003 or AD8232)
void publishECGChunk(float* samples, int count, int fs) {
    StaticJsonDocument<4096> doc;
    doc["patient_id"] = PATIENT_ID;
    doc["t0_ms"] = millis() + epoch_offset;
    doc["fs_hz"] = fs;
    JsonArray arr = doc.createNestedArray("samples");
    for (int i = 0; i < count; i++) {
        arr.add(samples[i]);
    }
    char buffer[4096];
    serializeJson(doc, buffer);
    mqttClient.publish(TOPIC_ECG, buffer);
}

// SpO2 (MAX30102)
void publishSpO2(float value) {
    StaticJsonDocument<256> doc;
    doc["patient_id"] = PATIENT_ID;
    doc["t_ms"] = millis() + epoch_offset;
    doc["value"] = value;
    char buffer[256];
    serializeJson(doc, buffer);
    mqttClient.publish(TOPIC_SPO2, buffer);
}
```

---

## Troubleshooting

### No data in viewer

1. Check MQTT broker is running: `docker ps`
2. Check replayer is publishing: Look for `[replayer] ECG: chunk=...` messages
3. Check edge preprocessor subscriptions: Look for `[edge] subscribed to...`

### No graphs showing

1. Ensure matplotlib is installed: `pip install matplotlib`
2. Run viewer with `--plot` flag
3. Wait 5+ seconds for first window to complete

### Motion artifact warnings

This is expected when IMU detects motion. The system automatically:
- Adds "motion_artifact" note to ECG/PPG features
- Degrades SQI scores
- Publishes motion_artifact events

### SpO2 drop events

Use `--inject-spo2-drop` flag with replayer to test:
```powershell
python src/replayer.py --mode synthetic --inject-spo2-drop
```

---

## Future Enhancements

1. **AI Classification**: Add neural network models for arrhythmia detection
2. **Multi-Patient**: Support multiple patients simultaneously
3. **Persistence**: Add InfluxDB for time-series storage
4. **Web Dashboard**: React/Vue frontend with real-time updates
5. **Alerts**: Push notifications for critical events
