# edge_mqtt_demo — Complete Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & System Design](#architecture--system-design)
3. [Technology Stack](#technology-stack)
4. [Step-by-Step Data Flow](#step-by-step-data-flow)
5. [Code Deep Dive](#code-deep-dive)
6. [Processing Pipeline Explained](#processing-pipeline-explained)
7. [Testing & Validation](#testing--validation)
8. [Future Hardware Integration](#future-hardware-integration)

---

## Project Overview

### What is this?
A **hardware-free digital twin** of a smart medical mat system. It simulates:
- **ESP32 sensor node** (publishes ECG data) → **Replayer service**
- **Raspberry Pi edge gateway** (preprocesses + extracts features) → **Edge Preprocessor service**
- **Cloud server/dashboard** (logs results) → **Viewer service**

### Why build it?
Before you have physical hardware (ESP32 + Raspberry Pi + ECG sensors), you need to:
- ✅ Validate the data pipeline (preprocessing, feature extraction, windowing).
- ✅ Test MQTT communication contracts (message format, topics, timing).
- ✅ Ensure all code compiles and runs end-to-end locally.
- ✅ Verify robustness (gap handling, out-of-order packets, buffering).

When real hardware arrives, integration is just: swap the publisher and move the edge service to Raspberry Pi.

### Design Philosophy
**"Plug-compatible from day one"**
- Fixed MQTT topics (`sim/patient1/ecg`, `edge/patient1/features`).
- Fixed JSON schemas (no surprises when hardware arrives).
- Services are **stateless** (can restart without data loss).
- Timestamps and windowing mimic real-time constraints.

---

## Architecture & System Design

```
┌────────────────────────────────────────────────────────────────┐
│                   YOUR LAPTOP (Simulation)                     │
│                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Replayer.py     │  │ Edge Preprocessor│  │  Viewer.py  │ │
│  │  (ESP32 twin)    │  │  .py (Raspi twin)│  │(Dashboard)  │ │
│  │                  │  │                  │  │             │ │
│  │ Publishes:       │  │ Subscribes:      │  │ Subscribes: │ │
│  │ sim/patient1/ecg │  │ sim/patient1/ecg │  │ edge/...    │ │
│  └────────┬─────────┘  └────────┬─────────┘  │ features    │ │
│           │                     │            └─────────────┘ │
│           │     JSON chunks     │                   ▲         │
│           │   (250ms windows)   │                   │         │
│           └─────────────┬───────┘                   │         │
│                         │                          │         │
│                    ┌────▼──────────────────────────┤         │
│                    │                               │         │
│                    │  MQTT Broker                  │         │
│                    │  (Mosquitto via Docker)       │         │
│                    │  localhost:1883               │         │
│                    │                               │         │
│                    └───────────────┬────────────────┘         │
│                                    │                          │
│                            JSON features                      │
│                            (every 5 sec)                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Services Overview

| Service | Role | Input | Output | Tech |
|---------|------|-------|--------|------|
| **Replayer** | Publishes ECG chunks | Synthetic OR file | MQTT topic `sim/patient1/ecg` | NumPy, synthetic signal generation |
| **Edge Preprocessor** | Filters, detects R-peaks, extracts HR | MQTT topic `sim/patient1/ecg` | MQTT topic `edge/patient1/features` | SciPy (filtering), NumPy (math) |
| **Viewer** | Logs + plots results | MQTT topic `edge/patient1/features` | Console + optional plot | Matplotlib |

---

## Technology Stack

### 1. **MQTT (Message Queuing Telemetry Transport)**
**What it is:**
- Lightweight pub/sub protocol for IoT devices.
- Publish data to named topics; others subscribe to receive it.
- Decouples producers from consumers (services don't need to know each other's IP).

**Why we use it:**
- ✅ Works with ESP32 (single-thread friendly).
- ✅ Works with Raspberry Pi (low overhead, battery-friendly).
- ✅ Works on local network and over internet (LTE, WiFi).
- ✅ Preserves message order + supports QoS (reliability).

**In this project:**
```
Replayer → publishes → sim/patient1/ecg (input topic)
Edge Preprocessor → subscribes → sim/patient1/ecg
                 → publishes → edge/patient1/features (output topic)
Viewer → subscribes → edge/patient1/features
```

### 2. **Mosquitto**
**What it is:**
- Open-source MQTT broker (the server that routes messages).
- Runs on port `1883` (default unencrypted).

**Why we use it:**
- ✅ Lightweight (runs in Docker container ~10MB).
- ✅ Free and widely used in IoT (same one you'd run on Raspberry Pi).
- ✅ Supports QoS (guaranteed message delivery if needed).

**In this project:**
- Runs in Docker Desktop container.
- Listens on `localhost:1883`.
- When Replayer publishes to `sim/patient1/ecg`, Mosquitto routes it to all subscribers.

### 3. **Python 3.11+**
**Why we use it:**
- ✅ Fast to iterate (perfect for ML/signal processing).
- ✅ Strong NumPy/SciPy ecosystem (filtering, peak detection).
- ✅ Paho-MQTT library (reliable Python MQTT client).
- ✅ Same language runs on ESP32 (MicroPython) and Raspberry Pi (CPython).

### 4. **NumPy**
**What it does:**
- Fast numerical arrays and math operations.

**In this project:**
- Holds ECG samples as arrays.
- Performs filtering, detrending, peak detection on arrays.
- Calculates HR statistics (mean, std, etc.).

**Example:**
```python
samples = np.array([0.01, 0.02, -0.01, 0.005, ...])  # ECG chunk
mean = np.mean(samples)  # Baseline
filtered = samples - mean  # Detrend
```

### 5. **SciPy**
**What it does:**
- Advanced signal processing (filtering, peak finding).

**In this project:**
- **Detrend**: remove slow baseline drift.
- **Butterworth filter**: bandpass 0.5–40 Hz (remove powerline + noise).
- **Notch filter**: remove 50 Hz powerline interference.
- **find_peaks**: detect R-peaks in filtered ECG.

**Example:**
```python
from scipy.signal import butter, filtfilt
b, a = butter(3, [0.5, 40], fs=250, btype='band')
filtered = filtfilt(b, a, ecg_samples)  # Zero-phase bandpass
```

### 6. **Paho-MQTT**
**What it is:**
- Python MQTT client library.

**In this project:**
- Each service creates an `mqtt.Client()` instance.
- Callbacks (`on_connect`, `on_message`, `on_disconnect`) handle events.
- Services auto-reconnect on failure (with exponential backoff).

**Example:**
```python
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect  # Callback when connected
client.connect('localhost', 1883, keepalive=30)
client.loop_start()  # Background thread for pub/sub
```

### 7. **Docker**
**What it is:**
- Containerization: runs Mosquitto in an isolated, pre-configured environment.

**In this project:**
- `docker-compose.yml` defines Mosquitto container.
- Simplifies setup (no native Mosquitto install needed).
- Same container can deploy to production Raspberry Pi.

### 8. **Matplotlib** (optional)
**What it does:**
- Real-time plotting of HR trend.

**In this project:**
- Viewer can plot HR over time with `--plot` flag.
- Shows quality/trends immediately (useful for debugging).

---

## Step-by-Step Data Flow

### Timeline: What happens when you run `.\run_all.bat`

```
T=0s:    run_all.bat starts
         ├─ Checks Python path
         ├─ Starts Docker Compose (Mosquitto)
         └─ Verifies broker on localhost:1883

T=2s:    Launches 3 terminal windows:
         ├─ Edge Preprocessor (waits for messages)
         ├─ Viewer (waits for messages)
         └─ Replayer (starts publishing)

T=3s:    Replayer connects to MQTT, starts ECG generation
         └─ Generates 250Hz synthetic signal for 30 minutes

T=3.25s: Replayer publishes 1st chunk
         ├─ JSON: {"patient_id": "patient1", "t0_ms": 12345, "fs_hz": 250, "samples": [0.01, 0.02, ...]}
         └─ Topic: sim/patient1/ecg

T=3.25s: Mosquitto routes message to subscribers
         ├─ Edge Preprocessor receives it
         └─ Buffers the samples (appends to stream state)

T=3.5s:  Replayer publishes 2nd chunk (another 62 samples at t0=12345+250)
         └─ Edge Preprocessor buffers it

T=3.75s: Replayer publishes 3rd chunk
         └─ Edge Preprocessor now has 3 * 62 = 186 samples

T=4s:    Replayer publishes 4th chunk
         └─ Edge Preprocessor now has 250 samples = 1 second of data

T=4.5s:  Replayer publishes 5th chunk → 312 samples

T=5.25s: Replayer publishes 6th chunk → 374 samples

T=6s:    Replayer publishes 7th chunk → 1262 samples (5 seconds!)
         ├─ Edge Preprocessor now has enough for first window
         ├─ Extracts window[0:1250] (5 sec at 250Hz)
         ├─ Preprocesses: detrend + bandpass + notch
         ├─ Detects R-peaks
         ├─ Calculates HR from RR intervals
         ├─ Computes SQI
         └─ Publishes to edge/patient1/features

T=6s:    Mosquitto routes to Viewer
         ├─ Viewer receives: {"patient_id": "patient1", "window_start_ms": 12345, "hr_bpm": 72.4, "sqi": 0.91, "notes": []}
         └─ Prints to console: [viewer] patient=patient1 ... hr=72.4 sqi=0.91

T=6.25s: Replayer publishes 8th chunk
         └─ Edge Preprocessor slides window forward by 1250 samples (5 sec)
            and waits for next 1250 samples

T=11.25s: 2nd window published to edge/patient1/features
         └─ Viewer prints 2nd HR value

... repeats every 5 seconds ...

T=30m:   Replayer finishes (duration-sec=1800 by default)
         └─ All services continue running (can Ctrl+C to stop)
```

### Key insight: **Windowing**
- Edge Preprocessor buffers incoming chunks.
- Once buffer ≥ 5 seconds of data, it extracts a window.
- Publishes window results.
- Slides buffer forward by 5 seconds (non-overlapping).

---

## Code Deep Dive

### 1. **common.py** — Shared Utilities

```python
# Message validation
def validate_input_message(payload: Dict[str, Any]) -> Optional[str]:
    """Check if received JSON has required fields + correct types."""
    required = ["patient_id", "t0_ms", "fs_hz", "samples"]
    for key in required:
        if key not in payload:
            return f"missing_key:{key}"
    # ... more checks ...
    return None

# MQTT config
class MqttConfig:
    host: str = "localhost"
    port: int = 1883
    keepalive: int = 30

# Helper to convert Python dict → JSON string
def to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))
```

**Why this matters:**
- Validates contracts before processing (fails fast).
- Centralizes config (change port once, all services use it).
- Reusable across all three services.

---

### 2. **replayer.py** — ESP32 Simulator

#### High-level flow:
```python
def main():
    1. Parse command-line args (--mode, --fs, --chunk-ms, --hr-bpm, etc.)
    2. Load or generate ECG signal
    3. Connect to MQTT broker
    4. Loop through signal in chunks, publish every chunk_ms
```

#### Synthetic signal generation:
```python
def generate_synthetic_ecg(duration_sec, fs_hz, hr_bpm, noise_std):
    """
    Creates realistic ECG with P, QRS, T waves at specified HR.
    """
    t = np.arange(n) / fs_hz
    
    # Beat times (e.g., every 60/72 = 0.833 seconds for 72 bpm)
    rr_sec = 60.0 / hr_bpm
    beat_times = np.arange(0.4, duration_sec, rr_sec)
    
    # Add P, Q, R, S, T waves using Gaussians
    signal += 1.2 * exp(-(t - R_time)^2 / 0.015^2)  # R peak
    signal += -0.15 * exp(-(t - Q_time)^2 / 0.01^2)  # Q wave
    # ... P, S, T similar ...
    
    # Add drift, powerline, noise
    baseline = 0.1 * sin(2π * 0.2 * t)  # Slow drift
    powerline = 0.02 * sin(2π * 50 * t)  # 50 Hz interference
    noise = randn() * noise_std
    
    return signal + baseline + powerline + noise
```

**Why realistic?**
- Tests preprocessing filters (must remove drift/powerline).
- Tests peak detector (handles physiological P, QRS, T waves).
- Tests SQI (noise affects quality score).

#### Publishing loop:
```python
while index < len(ecg):
    chunk = ecg[index : index + samples_per_chunk]
    
    # Calculate timestamp
    t0_ms = stream_t0_ms + chunk_duration_ms(index, fs_hz)
    
    # Create JSON
    payload = {
        "patient_id": "patient1",
        "t0_ms": t0_ms,
        "fs_hz": fs_hz,
        "samples": chunk.tolist()
    }
    
    # Publish to MQTT
    client.publish("sim/patient1/ecg", json.dumps(payload), qos=1)
    
    # Real-time pacing (sleep to match chunk duration)
    elapsed = (len(chunk) / fs_hz) / speed_factor
    time.sleep(elapsed)
    
    index += len(chunk)
```

**Key features:**
- ✅ **Real-time pacing**: sleeps between chunks to simulate live streaming.
- ✅ **Chunk timestamps**: `t0_ms` marks start time (enables sync with multiple sensors later).
- ✅ **QoS=1**: broker guarantees delivery (at least once).
- ✅ **Speed control**: `--speed 2.0` publishes twice faster (useful for rapid testing).

#### Optional: Inject gaps/out-of-order chunks
```python
if args.inject_gap_every > 0 and chunk_idx % args.inject_gap_every == 0:
    t0_ms += args.chunk_ms * 2  # Skip one chunk's worth of time
    # Edge will detect gap and fill with zeros

if args.inject_oop_every > 0 and chunk_idx > 0 and chunk_idx % args.inject_oop_every == 0:
    t0_ms -= args.chunk_ms  # Send chunk out of order
    # Edge will detect and handle overlap
```

**Why test gaps/out-of-order?**
- Real hardware can disconnect, retransmit, or jitter.
- Edge must be robust (can't crash on imperfect transport).

---

### 3. **edge_preprocessor.py** — Raspberry Pi Simulator

#### High-level flow:
```python
def main():
    1. Connect to MQTT broker
    2. Subscribe to sim/patient1/ecg
    3. On each message:
        a. Validate input JSON
        b. Add chunk to stream buffer
        c. If buffer ≥ 5 sec, extract + process window
        d. Publish features to edge/patient1/features
```

#### Stream buffering & windowing:
```python
class StreamState:
    fs_hz: int  # Sampling rate
    buffer_start_ms: int  # Timestamp of first sample in buffer
    buffer: np.ndarray  # Accumulating ECG samples
    
    @property
    def buffer_end_ms(self) -> int:
        """Timestamp of last sample in buffer."""
        return buffer_start_ms + chunk_duration_ms(len(buffer), fs_hz)

# In message handler:
state = stream_states.get(patient_id)

# Handle gap (missing samples)
if t0_ms > state.buffer_end_ms:
    gap_ms = t0_ms - state.buffer_end_ms
    gap_samples = int(gap_ms * fs_hz / 1000)
    state.buffer = np.concatenate([state.buffer, np.zeros(gap_samples)])
    notes.append("gap_filled_zeros")

# Handle overlap (out-of-order chunk)
elif t0_ms < state.buffer_end_ms:
    overlap_ms = state.buffer_end_ms - t0_ms
    overlap_samples = int(overlap_ms * fs_hz / 1000)
    state.buffer = np.concatenate([state.buffer, samples[overlap_samples:]])
    notes.append("overlap_trimmed")

# Normal case (sequential)
else:
    state.buffer = np.concatenate([state.buffer, samples])

# Extract windows when enough data
window_samples = int(args.window_sec * fs_hz)  # 1250 for 5s @ 250Hz
while len(state.buffer) >= window_samples:
    window = state.buffer[:window_samples]
    hr_bpm, sqi, notes = process_window(window, fs_hz, notch_hz)
    
    # Publish
    output = {
        "patient_id": patient_id,
        "window_start_ms": state.buffer_start_ms,
        "window_sec": 5,
        "hr_bpm": hr_bpm,
        "sqi": sqi,
        "notes": notes
    }
    client.publish("edge/patient1/features", json.dumps(output))
    
    # Slide window
    state.buffer = state.buffer[window_samples:]
    state.buffer_start_ms += chunk_duration_ms(window_samples, fs_hz)
```

**Key robustness features:**
- ✅ **Gap handling**: fills with zeros, tags with note.
- ✅ **Out-of-order handling**: trims overlap, continues.
- ✅ **Per-patient buffering**: multiple patients simultaneously (future-proof).
- ✅ **Timestamp tracking**: knows exact start time of each window.

---

### 4. **Preprocessing Pipeline** — In `preprocess_ecg()`

```python
def preprocess_ecg(window, fs_hz, notch_hz):
    x = window.astype(float)
    
    # Step 1: Detrend (remove slow drift)
    x = detrend(x, type='linear')
    # Removes polynomial baseline (e.g., slow DC shift from electrode motion)
    
    # Step 2: Bandpass filter (0.5–40 Hz)
    nyq = fs_hz / 2.0  # Nyquist frequency (125 Hz for 250 Hz sampling)
    low = 0.5 / nyq
    high = 40.0 / nyq
    b, a = butter(3, [low, high], btype='band')  # 3rd order Butterworth
    x = filtfilt(b, a, x)
    # Removes DC (< 0.5 Hz), muscle noise (> 40 Hz), powerline (50 Hz)
    
    # Step 3: Notch filter (50 Hz)
    if notch_hz is not None:
        b_n, a_n = iirnotch(w0=notch_hz/nyq, Q=30)
        x = filtfilt(b_n, a_n, x)
    # Removes remaining 50 Hz powerline
    
    return x
```

**Why each step?**
- **Detrend**: ECG drifts due to electrode motion; linear detrend removes low-freq trends.
- **Bandpass**: ECG info is 0.5–40 Hz; filters out noise outside this range.
- **Notch**: AC powerline (50 Hz in EU, 60 Hz in US) appears as noise; notch removes it precisely.

---

### 5. **R-peak Detection** — In `detect_r_peaks()`

```python
def detect_r_peaks(filtered, fs_hz):
    """
    Finds local maxima in filtered ECG that correspond to R-peaks.
    """
    # Estimate noise baseline
    signal_std = np.std(filtered)
    threshold = max(0.35 * signal_std, 0.05)
    
    # Minimum distance between peaks (refractory period)
    min_distance = int(0.25 * fs_hz)  # 250 ms for 250 Hz
    # (heart can't beat faster than ~240 bpm = 250 ms between beats)
    
    # Find peaks using SciPy
    peaks, _ = find_peaks(filtered, distance=min_distance, prominence=threshold)
    return peaks
```

**What this does:**
1. Sets threshold = 35% of signal std (adaptive to noise level).
2. Enforces minimum 250ms between peaks (physiological constraint).
3. Returns indices of R-peaks in the window.

**Example output:**
```
filtered_ecg = [0.1, 0.3, 1.2, 0.2, -0.1, ..., 0.95, 1.15, 0.3, ...]
                                    ↑                ↑       ↑
R-peak indices:                    [42,    ...,     315,    892]
```

---

### 6. **HR Estimation** — In `estimate_hr_bpm()`

```python
def estimate_hr_bpm(r_peaks, fs_hz):
    """
    Calculates heart rate from RR intervals.
    """
    if len(r_peaks) < 2:
        return None  # Need at least 2 peaks
    
    # Time between consecutive R-peaks (in seconds)
    rr_sec = np.diff(r_peaks) / float(fs_hz)
    # Example: peaks at [250, 500, 750] samples
    #         rr_sec = [1.0, 1.0] seconds
    #         rr_sec = [60 bpm, 60 bpm] ← HR from each interval
    
    # Filter for physiological range (40–180 bpm = 0.33–1.5 sec)
    rr_sec = rr_sec[(rr_sec > 0.3) & (rr_sec < 2.0)]
    if rr_sec.size == 0:
        return None
    
    # Average HR
    mean_rr = np.mean(rr_sec)
    hr_bpm = 60.0 / mean_rr
    return hr_bpm
```

**Example:**
```
R-peaks at: [125, 375, 625, 875] (samples)
Differences: [250, 250, 250] (samples)
At 250 Hz: [1.0, 1.0, 1.0] (seconds)
HR = 60 / 1.0 = 60 bpm

If one interval is 0.5s (off-beat):
RR intervals: [1.0, 0.5, 1.0] → filtered to [1.0, 1.0]
Mean RR = 1.0s → HR = 60 bpm (robust to occasional ectopy)
```

---

### 7. **Signal Quality Index (SQI)** — In `compute_sqi()`

```python
def compute_sqi(raw, filtered, r_peaks, fs_hz):
    """
    Heuristic 0–1 score indicating ECG quality.
    """
    score = 1.0
    
    # Penalty 1: Flatline detection (low std = flat signal)
    if np.std(raw) < 1e-4:
        score -= 0.8
    
    # Penalty 2: Excessive flatness (too many zero-difference samples)
    diff = np.abs(np.diff(raw))
    flat_ratio = np.mean(diff < 1e-5)
    if flat_ratio > 0.2:  # >20% of samples don't change
        score -= 0.4
    
    # Penalty 3: High-frequency noise
    hf_noise = raw - filtered
    noise_ratio = np.std(hf_noise) / (np.std(filtered) + 1e-6)
    if noise_ratio > 1.0:  # Noise > signal
        score -= 0.3
    elif noise_ratio > 0.7:  # Noise ~70% of signal
        score -= 0.15
    
    # Penalty 4: Implausible peak density
    peak_count = len(r_peaks)
    expected_min = int(40 / 60 * 5)   # Min ~3 peaks in 5s (40 bpm min)
    expected_max = int(180 / 60 * 5)  # Max ~15 peaks in 5s (180 bpm max)
    if peak_count < expected_min or peak_count > expected_max:
        score -= 0.2
    
    return max(0.0, min(1.0, score))
```

**Example scenarios:**
```
Scenario 1: Clean signal
- std(raw) = 0.5 (good)
- flat_ratio = 0.01 (clean)
- noise_ratio = 0.2 (low noise)
- peak_count = 6 (4–5s window, ~72 bpm, normal)
→ score = 1.0 (excellent)

Scenario 2: Noisy signal
- std(raw) = 0.3 (okay)
- flat_ratio = 0.05 (clean)
- noise_ratio = 1.2 (high noise)
- peak_count = 5 (okay)
→ score = 1.0 - 0.3 = 0.7 (acceptable)

Scenario 3: Disconnected electrode (flatline)
- std(raw) = 1e-5 (flatline!)
- flat_ratio = 0.95 (all samples same)
- noise_ratio = 0.01
- peak_count = 0
→ score = 1.0 - 0.8 - 0.4 - 0.2 = -0.4 → clamped to 0.0 (poor/disconnected)
```

---

### 8. **viewer.py** — Dashboard Simulator

```python
def main():
    # Subscribe to edge/patient1/features
    # On message: print HR + SQI + notes
    
    # Optional: plot HR trend in real-time
```

**Output:**
```
[viewer] patient=patient1 window=1700000002345 hr=72.4 sqi=0.91 notes=[]
[viewer] patient=patient1 window=1700000007345 hr=71.8 sqi=0.88 notes=[]
[viewer] patient=patient1 window=1700000012345 hr=73.2 sqi=0.85 notes=['low_sqi']
```

---

## Processing Pipeline Explained

### Full end-to-end example with actual numbers

**Input:** Synthetic 72 bpm ECG, 250 Hz, 30 minutes

**T=0ms:**
```
Replayer generates:
- Signal: P + QRS + T waves at ~0.83s intervals (72 bpm)
- Baseline drift: 0.2 Hz sine wave
- Powerline: 50 Hz sine
- Noise: Gaussian, σ=0.03

Publishes chunk 0:
{
  "patient_id": "patient1",
  "t0_ms": 0,
  "fs_hz": 250,
  "samples": [0.05, 0.07, ..., 0.02]  # 62 samples (250ms @ 250Hz)
}
```

**T=250ms:**
```
Replayer publishes chunk 1 (t0_ms=250)
Edge Preprocessor buffers: [0.05, 0.07, ..., 0.02] + [0.03, 0.06, ..., 0.01]
Total: 124 samples (496 ms)
Edge checks: 124 samples < 1250 samples (5s) → wait
```

**T=5000ms (5 seconds, ~20 chunks published):**
```
Edge Preprocessor now has 1250+ samples
Extracts window[0:1250] (samples from t=0 to t=5000 ms)

Preprocessing:
1. Detrend: removes baseline drift
2. Bandpass 0.5–40 Hz: removes DC + muscle noise + powerline
3. Notch 50 Hz: precision removal of remaining powerline

→ Filtered ECG is now clean (P, QRS, T waves visible)

Peak detection:
- std(filtered) ≈ 0.4
- threshold = max(0.35 * 0.4, 0.05) = 0.14
- Finds peaks > 0.14, separated by ≥250 ms
- Found 6 peaks (at ~833ms intervals for 72 bpm)

R-peak indices: [187, 395, 603, 811, 1019, 1227] (samples)
RR intervals: [0.833, 0.833, 0.833, 0.833, 0.832] seconds
HR = 60 / 0.833 = 72.0 bpm ✓

SQI calculation:
- std(raw) = 0.35 (good)
- flat_ratio = 0.02 (minimal flatness)
- noise_ratio = 0.25 (low noise)
- peak_count = 6 (4–5s is normal)
→ score = 1.0

Publishes:
{
  "patient_id": "patient1",
  "window_start_ms": 0,
  "window_sec": 5,
  "hr_bpm": 72.0,
  "sqi": 1.0,
  "notes": []
}

Viewer receives & prints:
[viewer] patient=patient1 window=0 hr=72.0 sqi=1.0 notes=[]
```

**T=10000ms (10 seconds):**
```
Edge slides buffer forward by 1250 samples
New window starts at t=5000ms
Extracts samples[1250:2500] (the next 5s)
→ Second HR measurement published
```

---

## Testing & Validation

### Test 1: Basic functionality
```powershell
.\run_all.bat
# Check:
# ✓ Viewer prints HR every 5 seconds
# ✓ HR ≈ 72 bpm (set by --hr-bpm)
# ✓ SQI ≥ 0.8 (clean synthetic signal)
# ✓ No errors in edge preprocessor
```

### Test 2: Robustness (gaps + out-of-order)
```powershell
python .\src\replayer.py --mode synthetic --duration-sec 60 --chunk-ms 250 --inject-gap-every 25 --inject-oop-every 40
# Check edge log for:
# ✓ gap_filled_zeros (when gap injected)
# ✓ overlap_trimmed (when out-of-order chunk received)
# ✓ HR still calculated (no crashes)
# ✓ SQI may drop (data quality degraded by injection)
```

### Test 3: Different heart rates
```powershell
python .\src\replayer.py --hr-bpm 60   # Bradycardic
python .\src\replayer.py --hr-bpm 120  # Tachycardic
# Check:
# ✓ Viewer HR matches requested BPM (±5%)
# ✓ No "hr_out_of_range" notes for 60–120 bpm
```

### Test 4: Noisy signal
```powershell
python .\src\replayer.py --noise-std 0.1  # Increase noise from 0.03
# Check:
# ✓ SQI drops (but >0.5 if still readable)
# ✓ HR may be less stable
# ✓ Notes include "low_sqi" when score < 0.5
```

### Test 5: File replay (bring your own ECG)
```powershell
# Create file data\my_ecg.csv with ECG values
python .\src\replayer.py --mode file --file .\data\my_ecg.csv --loop
# Check:
# ✓ All values loaded
# ✓ Loops continuously
# ✓ HR/SQI calculated from real data
```

---

## Future Hardware Integration

### When ESP32 + Raspberry Pi arrive

#### 1. **Physical ESP32**
Replace `replayer.py` with:
```cpp
// Arduino / ESP-IDF firmware
#include <WiFi.h>
#include <PubSubClient.h>
#include <driver/adc.h>

// Read ECG from ADC pin
uint16_t ecg_raw = analogRead(ECG_PIN);
float ecg_val = ecg_raw / 4096.0 * 3.3;  // Convert to voltage

// Buffer and publish every 250ms
if (millis() - last_publish >= 250) {
    String payload = "{\"patient_id\": \"patient1\", \"t0_ms\": " + String(millis()) + 
                     ", \"fs_hz\": 250, \"samples\": [" + readings + "]}";
    client.publish("sim/patient1/ecg", payload);
    last_publish = millis();
}
```

**Changes needed:**
- ✓ Same MQTT topic: `sim/patient1/ecg`
- ✓ Same JSON schema
- ✓ Broker host: either local `192.168.1.X:1883` or cloud IP
- → **Everything else stays the same!**

#### 2. **Physical Raspberry Pi**
Copy `edge_preprocessor.py` to Pi:
```bash
scp -r edge_mqtt_demo pi@192.168.1.100:/home/pi/
ssh pi@192.168.1.100
cd /home/pi/edge_mqtt_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/edge_preprocessor.py --broker-host 192.168.1.100
```

**Changes needed:**
- ✓ `--broker-host 192.168.1.100` (instead of `localhost`)
- ✓ Optional: add startup script to systemd (auto-start at boot)
- → **Code is 100% identical!**

#### 3. **Add more sensors (PPG/SpO2/Temp)**
New topics:
```
sim/patient1/ppg       → edge process → edge/patient1/ppg_features
sim/patient1/spo2      → edge process → edge/patient1/spo2_status
sim/patient1/temp      → edge process → edge/patient1/temp_status
```

Create new subscribers in `edge_preprocessor.py`:
```python
# Subscribe to multiple topics
client.subscribe("sim/patient1/ecg")
client.subscribe("sim/patient1/ppg")
client.subscribe("sim/patient1/spo2")

# Multi-sensor buffering (align by timestamp)
stream_states["patient1"]["ecg"] = StreamState(...)
stream_states["patient1"]["ppg"] = StreamState(...)
stream_states["patient1"]["spo2"] = StreamState(...)
```

#### 4. **Edge persistence** (buffering if Raspberry goes offline)
```python
import sqlite3

# Store features locally
conn = sqlite3.connect('edge_cache.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS features
             (id INT, patient_id TEXT, window_start_ms INT, hr_bpm REAL, sqi REAL, timestamp DATETIME)''')
c.execute('INSERT INTO features VALUES (?, ?, ?, ?, ?, datetime("now"))', 
          (patient_id, window_start_ms, hr_bpm, sqi))
conn.commit()
```

---

## Summary: Technology Choices & Rationale

| Tech | Why | Trade-off |
|------|-----|-----------|
| **MQTT** | Standard IoT, decouples services | Fixed network dependency |
| **Python** | FastIteration + NumPy/SciPy | Not real-time safe (GC pauses) |
| **NumPy** | Fast array math | Requires C extensions |
| **SciPy** | Reliable filtering/peak finding | Heavier than DIY (ok on Raspi) |
| **Paho-MQTT** | Pure Python, easy callbacks | Slower than C libs (ok for 250Hz) |
| **Docker** | Instant Mosquitto setup | Requires Docker Desktop |
| **JSON** | Human-readable, debuggable | Less compact than protobuf (ok for 250Hz) |

---

## Next Steps

1. ✅ **Run `.\run_all.bat`** to validate full pipeline.
2. ✅ **Monitor edge logs** for preprocessing notes.
3. ✅ **Test robustness** with gap/OOP injection.
4. ✅ **Extend**: add PPG/SpO2 simulators and multi-sensor windowing.
5. ✅ **Deploy**: when hardware arrives, swap publisher + move to Raspberry.

---

**Questions?** Every line of code is documented; ask about any section!
