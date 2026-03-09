from __future__ import annotations

"""
Edge Gateway / Multi-Sensor Preprocessor
==========================================
Subscribes to all sensor topics, buffers data, processes in 5-second windows,
extracts features, detects events, and publishes to output topics.

Input topics (from ESP32/simulator):
- sim/patient1/ecg (250 Hz chunks)
- sim/patient1/ppg (100 Hz chunks)
- sim/patient1/imu (50 Hz triplet chunks)
- sim/patient1/spo2 (1 Hz samples)
- sim/patient1/temp (0.2 Hz samples)

Output topics:
- edge/patient1/features (unified features every 5 seconds)
- edge/patient1/events (events on conditions)

Future hardware: Same code runs on Raspberry Pi with real ESP32 data.
"""

import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import paho.mqtt.client as mqtt

from common import (
    DEFAULT_DEVICE_ID,
    EventType,
    MqttConfig,
    SensorConfig,
    Severity,
    chunk_duration_ms,
    create_event,
    create_features,
    get_input_topics,
    get_output_topics,
    parse_json,
    parse_topic,
    safe_float_list,
    safe_imu_triplet_list,
    to_json,
    validate_chunk_message,
    validate_imu_chunk_message,
    validate_sample_message,
)

# SciPy is optional but recommended for better signal processing
try:
    from scipy.signal import butter, detrend, filtfilt, find_peaks, iirnotch

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# DATA BUFFERS
# =============================================================================

@dataclass
class ChunkBuffer:
    """Buffer for high-rate chunk-based streams (ECG, PPG)."""
    fs_hz: int
    buffer_start_ms: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    @property
    def buffer_end_ms(self) -> int:
        if len(self.buffer) == 0:
            return self.buffer_start_ms
        return self.buffer_start_ms + chunk_duration_ms(len(self.buffer), self.fs_hz)

    def append_chunk(self, t0_ms: int, samples: np.ndarray) -> List[str]:
        """Append chunk with timestamp handling. Returns notes about issues."""
        notes: List[str] = []
        n = len(samples)

        if len(self.buffer) == 0:
            self.buffer_start_ms = t0_ms
            self.buffer = samples.copy()
            return notes

        end_ms = self.buffer_end_ms

        if t0_ms > end_ms:
            # Gap detected - fill with zeros
            gap_ms = t0_ms - end_ms
            gap_samples = int(round(gap_ms * self.fs_hz / 1000.0))
            if gap_samples > 0:
                notes.append("gap_filled_zeros")
                self.buffer = np.concatenate([self.buffer, np.zeros(gap_samples, dtype=float)])
            self.buffer = np.concatenate([self.buffer, samples])

        elif t0_ms < end_ms:
            # Overlap detected - trim
            overlap_ms = end_ms - t0_ms
            overlap_samples = int(round(overlap_ms * self.fs_hz / 1000.0))
            if overlap_samples >= n:
                notes.append("duplicate_chunk_dropped")
            else:
                if overlap_samples > 0:
                    notes.append("overlap_trimmed")
                self.buffer = np.concatenate([self.buffer, samples[overlap_samples:]])
        else:
            # Perfect alignment
            self.buffer = np.concatenate([self.buffer, samples])

        return notes

    def extract_window(self, window_samples: int) -> Tuple[np.ndarray, int]:
        """Extract a window from the buffer. Returns (window, window_start_ms)."""
        if len(self.buffer) < window_samples:
            return np.array([], dtype=float), self.buffer_start_ms

        window = self.buffer[:window_samples].copy()
        window_start = self.buffer_start_ms

        # Advance buffer
        self.buffer = self.buffer[window_samples:]
        self.buffer_start_ms += chunk_duration_ms(window_samples, self.fs_hz)

        return window, window_start

    def has_window(self, window_samples: int) -> bool:
        return len(self.buffer) >= window_samples


@dataclass
class ImuBuffer:
    """Buffer for IMU triplet data."""
    fs_hz: int
    buffer_start_ms: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))

    @property
    def buffer_end_ms(self) -> int:
        if len(self.buffer) == 0:
            return self.buffer_start_ms
        return self.buffer_start_ms + chunk_duration_ms(len(self.buffer), self.fs_hz)

    def append_chunk(self, t0_ms: int, samples: np.ndarray) -> List[str]:
        """Append IMU chunk. samples should be (N, 3) array."""
        notes: List[str] = []

        if samples.ndim == 1:
            samples = samples.reshape(-1, 3)

        if len(self.buffer) == 0:
            self.buffer_start_ms = t0_ms
            self.buffer = samples.copy()
            return notes

        end_ms = self.buffer_end_ms

        if t0_ms > end_ms:
            gap_ms = t0_ms - end_ms
            gap_samples = int(round(gap_ms * self.fs_hz / 1000.0))
            if gap_samples > 0:
                notes.append("gap_filled_zeros")
                zeros = np.zeros((gap_samples, 3), dtype=float)
                self.buffer = np.vstack([self.buffer, zeros])
            self.buffer = np.vstack([self.buffer, samples])

        elif t0_ms < end_ms:
            overlap_ms = end_ms - t0_ms
            overlap_samples = int(round(overlap_ms * self.fs_hz / 1000.0))
            if overlap_samples >= len(samples):
                notes.append("duplicate_chunk_dropped")
            else:
                if overlap_samples > 0:
                    notes.append("overlap_trimmed")
                self.buffer = np.vstack([self.buffer, samples[overlap_samples:]])
        else:
            self.buffer = np.vstack([self.buffer, samples])

        return notes

    def extract_window(self, window_samples: int) -> Tuple[np.ndarray, int]:
        if len(self.buffer) < window_samples:
            return np.zeros((0, 3), dtype=float), self.buffer_start_ms

        window = self.buffer[:window_samples].copy()
        window_start = self.buffer_start_ms

        self.buffer = self.buffer[window_samples:]
        self.buffer_start_ms += chunk_duration_ms(window_samples, self.fs_hz)

        return window, window_start

    def has_window(self, window_samples: int) -> bool:
        return len(self.buffer) >= window_samples


@dataclass
class SampleBuffer:
    """Buffer for low-rate sample-based streams (SpO2, Temperature)."""
    samples: List[Tuple[int, float]] = field(default_factory=list)  # (t_ms, value)

    def add_sample(self, t_ms: int, value: float) -> None:
        self.samples.append((t_ms, value))

    def extract_window(self, window_start_ms: int, window_end_ms: int) -> List[float]:
        """Extract samples within the time window."""
        values = [v for t, v in self.samples if window_start_ms <= t < window_end_ms]
        # Remove old samples
        self.samples = [(t, v) for t, v in self.samples if t >= window_start_ms]
        return values


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def preprocess_ecg(window: np.ndarray, fs_hz: int, notch_hz: Optional[float]) -> np.ndarray:
    """Apply ECG preprocessing: detrend, bandpass, notch filter."""
    x = window.astype(float)

    if SCIPY_AVAILABLE:
        x = detrend(x, type="linear")

        nyq = fs_hz / 2.0
        low = 0.5 / nyq
        high = 40.0 / nyq
        b, a = butter(3, [low, min(high, 0.99)], btype="band")
        x = filtfilt(b, a, x)

        if notch_hz is not None and 0 < notch_hz < nyq:
            b_n, a_n = iirnotch(w0=notch_hz / nyq, Q=30)
            x = filtfilt(b_n, a_n, x)
    else:
        # Fallback: simple mean removal
        x = x - np.mean(x)

    return x


def preprocess_ppg(window: np.ndarray, fs_hz: int) -> np.ndarray:
    """Apply PPG preprocessing: detrend, bandpass 0.5-8 Hz."""
    x = window.astype(float)

    if SCIPY_AVAILABLE:
        x = detrend(x, type="linear")

        nyq = fs_hz / 2.0
        low = 0.5 / nyq
        high = 8.0 / nyq
        b, a = butter(2, [low, min(high, 0.99)], btype="band")
        x = filtfilt(b, a, x)
    else:
        x = x - np.mean(x)

    return x


def detect_peaks(filtered: np.ndarray, fs_hz: int, min_hr: float = 40, max_hr: float = 180) -> np.ndarray:
    """Detect peaks (R-peaks for ECG, systolic peaks for PPG)."""
    if filtered.size < fs_hz:
        return np.array([], dtype=int)

    signal_std = float(np.std(filtered))
    if signal_std < 1e-6:
        return np.array([], dtype=int)

    threshold = max(0.35 * signal_std, 0.05)
    # Minimum peak distance based on max HR
    min_distance = int((60.0 / max_hr) * fs_hz)

    if SCIPY_AVAILABLE:
        peaks, _ = find_peaks(filtered, distance=min_distance, prominence=threshold)
        return peaks.astype(int)

    # Fallback: simple local maxima
    candidates = np.where(
        (filtered[1:-1] > filtered[:-2])
        & (filtered[1:-1] > filtered[2:])
        & (filtered[1:-1] > threshold)
    )[0] + 1

    selected: List[int] = []
    for idx in candidates:
        if not selected or (idx - selected[-1]) >= min_distance:
            selected.append(int(idx))
    return np.asarray(selected, dtype=int)


def estimate_hr_bpm(peaks: np.ndarray, fs_hz: int) -> Optional[float]:
    """Estimate heart rate from peak intervals."""
    if len(peaks) < 2:
        return None

    intervals_sec = np.diff(peaks) / float(fs_hz)
    # Filter physiological RR intervals (0.3s to 2.0s = 30-200 bpm)
    valid = intervals_sec[(intervals_sec > 0.3) & (intervals_sec < 2.0)]
    if len(valid) == 0:
        return None

    return float(60.0 / np.mean(valid))


def compute_ecg_sqi(raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray, fs_hz: int) -> float:
    """Compute Signal Quality Index for ECG."""
    score = 1.0

    # Check for flat signal
    raw_std = float(np.std(raw))
    if raw_std < 1e-4:
        score -= 0.8

    # Check for excessive flat segments
    diff = np.abs(np.diff(raw)) if raw.size > 1 else np.array([0.0])
    flat_ratio = float(np.mean(diff < 1e-5))
    if flat_ratio > 0.2:
        score -= 0.4

    # Check noise ratio
    hf_noise = raw - filtered
    noise_ratio = float(np.std(hf_noise) / (np.std(filtered) + 1e-6))
    if noise_ratio > 1.0:
        score -= 0.3
    elif noise_ratio > 0.7:
        score -= 0.15

    # Check peak count (expected 3-15 peaks in 5s for 40-180 bpm)
    expected_min = 3
    expected_max = 15
    if len(peaks) < expected_min or len(peaks) > expected_max:
        score -= 0.2

    return float(np.clip(score, 0.0, 1.0))


def compute_ppg_sqi(raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray, fs_hz: int) -> float:
    """Compute Signal Quality Index for PPG."""
    score = 1.0

    raw_std = float(np.std(raw))
    if raw_std < 1e-4:
        score -= 0.8

    # Check peak regularity
    if len(peaks) >= 3:
        intervals = np.diff(peaks)
        interval_std = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
        if interval_std > 0.3:
            score -= 0.3

    # Check noise
    hf_noise = raw - filtered
    noise_ratio = float(np.std(hf_noise) / (np.std(filtered) + 1e-6))
    if noise_ratio > 1.5:
        score -= 0.3

    return float(np.clip(score, 0.0, 1.0))


def compute_motion_score(imu_window: np.ndarray) -> Tuple[float, List[str]]:
    """
    Compute motion score from IMU data.
    Returns (score, notes) where score is 0-1 (0=still, 1=high motion).
    """
    notes: List[str] = []

    if imu_window.size == 0:
        return 0.0, ["no_imu_data"]

    # Compute magnitude of acceleration
    magnitudes = np.sqrt(np.sum(imu_window ** 2, axis=1))

    # Remove gravity baseline (~1g when still)
    magnitudes_detrended = magnitudes - 1.0

    # Compute variance as motion indicator
    variance = float(np.var(magnitudes_detrended))

    # Normalize to 0-1 scale (empirical thresholds)
    # variance < 0.01 -> still
    # variance > 0.5 -> high motion
    score = float(np.clip(variance / 0.5, 0.0, 1.0))

    if score > 0.5:
        notes.append("high_motion")
    elif score > 0.2:
        notes.append("moderate_motion")

    return score, notes


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def process_ecg_window(
    raw: np.ndarray,
    fs_hz: int,
    notch_hz: Optional[float],
    motion_score: float
) -> Dict[str, Any]:
    """Process ECG window and return features."""
    if len(raw) < fs_hz:
        return {"hr_bpm": None, "sqi": 0.0, "notes": ["insufficient_data"]}

    filtered = preprocess_ecg(raw, fs_hz, notch_hz)
    peaks = detect_peaks(filtered, fs_hz)
    hr_bpm = estimate_hr_bpm(peaks, fs_hz)
    sqi = compute_ecg_sqi(raw, filtered, peaks, fs_hz)

    notes: List[str] = []
    if hr_bpm is None:
        notes.append("insufficient_data")
    elif hr_bpm < 40 or hr_bpm > 180:
        notes.append("hr_out_of_range")

    if sqi < 0.5:
        notes.append("low_sqi")

    # Degrade SQI note if high motion
    if motion_score > 0.5:
        notes.append("motion_artifact")
        sqi = max(0.0, sqi - 0.2)

    return {
        "hr_bpm": round(hr_bpm, 2) if hr_bpm is not None else None,
        "sqi": round(sqi, 3),
        "notes": notes
    }


def process_ppg_window(
    raw: np.ndarray,
    fs_hz: int,
    motion_score: float
) -> Dict[str, Any]:
    """Process PPG window and return features."""
    if len(raw) < fs_hz:
        return {"hr_bpm": None, "sqi": 0.0, "notes": ["insufficient_data"]}

    filtered = preprocess_ppg(raw, fs_hz)
    peaks = detect_peaks(filtered, fs_hz)
    hr_bpm = estimate_hr_bpm(peaks, fs_hz)
    sqi = compute_ppg_sqi(raw, filtered, peaks, fs_hz)

    notes: List[str] = []
    if hr_bpm is None:
        notes.append("insufficient_data")
    elif hr_bpm < 40 or hr_bpm > 180:
        notes.append("hr_out_of_range")

    if sqi < 0.5:
        notes.append("low_sqi")

    if motion_score > 0.5:
        notes.append("motion_artifact")
        sqi = max(0.0, sqi - 0.3)  # PPG more affected by motion

    return {
        "hr_bpm": round(hr_bpm, 2) if hr_bpm is not None else None,
        "sqi": round(sqi, 3),
        "notes": notes
    }


def process_spo2_window(values: List[float], drop_state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process SpO2 values in window.
    drop_state tracks ongoing drop events.
    Returns (features, events).
    """
    events: List[Dict[str, Any]] = []

    if not values:
        return {"mean": None, "min": None, "drops": 0}, events

    mean_val = float(np.mean(values))
    min_val = float(np.min(values))
    drop_count = drop_state.get("drop_count", 0)

    # Check for SpO2 drop (value < 93 for >= 10 seconds)
    threshold = 93.0
    below_threshold = [v for v in values if v < threshold]

    if below_threshold:
        # Accumulate time below threshold
        drop_state["below_time_sec"] = drop_state.get("below_time_sec", 0) + len(values)
        drop_state["min_value"] = min(drop_state.get("min_value", 100), min(below_threshold))

        if drop_state["below_time_sec"] >= 10 and not drop_state.get("drop_reported", False):
            # Report drop event
            drop_count += 1
            drop_state["drop_count"] = drop_count
            drop_state["drop_reported"] = True

            severity = Severity.LOW
            if drop_state["min_value"] < 85:
                severity = Severity.HIGH
            elif drop_state["min_value"] < 90:
                severity = Severity.MODERATE

            events.append({
                "type": EventType.SPO2_DROP,
                "severity": severity,
                "details": {
                    "min_value": round(drop_state["min_value"], 1),
                    "duration_sec": drop_state["below_time_sec"]
                }
            })
    else:
        # Reset drop tracking
        drop_state["below_time_sec"] = 0
        drop_state["min_value"] = 100
        drop_state["drop_reported"] = False

    return {
        "mean": round(mean_val, 1),
        "min": round(min_val, 1),
        "drops": drop_count
    }, events


def process_temp_window(values: List[float]) -> Dict[str, Any]:
    """Process temperature values in window."""
    if not values:
        return {"mean": None}

    mean_val = float(np.mean(values))

    # Sanity check
    if mean_val < 30 or mean_val > 45:
        return {"mean": None, "notes": ["out_of_range"]}

    return {"mean": round(mean_val, 2)}


# =============================================================================
# PATIENT STATE
# =============================================================================

@dataclass
class PatientState:
    """State for a single patient's sensor streams."""
    patient_id: str
    config: SensorConfig

    # Buffers
    ecg_buffer: ChunkBuffer = field(default_factory=lambda: ChunkBuffer(fs_hz=250))
    ppg_buffer: ChunkBuffer = field(default_factory=lambda: ChunkBuffer(fs_hz=100))
    imu_buffer: ImuBuffer = field(default_factory=lambda: ImuBuffer(fs_hz=50))
    spo2_buffer: SampleBuffer = field(default_factory=SampleBuffer)
    temp_buffer: SampleBuffer = field(default_factory=SampleBuffer)

    # Persistent state
    spo2_drop_state: Dict[str, Any] = field(default_factory=dict)
    last_hr_ecg: Optional[float] = None
    last_hr_ppg: Optional[float] = None

    def __post_init__(self):
        self.ecg_buffer = ChunkBuffer(fs_hz=int(self.config.ecg_fs_hz))
        self.ppg_buffer = ChunkBuffer(fs_hz=int(self.config.ppg_fs_hz))
        self.imu_buffer = ImuBuffer(fs_hz=int(self.config.imu_fs_hz))


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class EdgePreprocessor:
    """Multi-sensor edge preprocessor."""

    def __init__(
        self,
        mqtt_client: mqtt.Client,
        device_id: str,
        config: SensorConfig,
        debug: bool = False
    ):
        self.client = mqtt_client
        self.device_id = device_id
        self.config = config
        self.debug = debug

        self.lock = threading.Lock()
        self.patients: Dict[str, PatientState] = {}

        self.input_topics = get_input_topics(device_id)
        self.output_topics = get_output_topics(device_id)

    def get_or_create_patient(self, patient_id: str) -> PatientState:
        """Get or create patient state."""
        if patient_id not in self.patients:
            self.patients[patient_id] = PatientState(patient_id=patient_id, config=self.config)
        return self.patients[patient_id]

    def handle_ecg_chunk(self, patient_id: str, t0_ms: int, fs_hz: int, samples: np.ndarray) -> None:
        """Handle incoming ECG chunk."""
        with self.lock:
            state = self.get_or_create_patient(patient_id)
            if state.ecg_buffer.fs_hz != fs_hz:
                state.ecg_buffer = ChunkBuffer(fs_hz=fs_hz)
            state.ecg_buffer.append_chunk(t0_ms, samples)
            self._try_process_window(state)

    def handle_ppg_chunk(self, patient_id: str, t0_ms: int, fs_hz: int, samples: np.ndarray) -> None:
        """Handle incoming PPG chunk."""
        with self.lock:
            state = self.get_or_create_patient(patient_id)
            if state.ppg_buffer.fs_hz != fs_hz:
                state.ppg_buffer = ChunkBuffer(fs_hz=fs_hz)
            state.ppg_buffer.append_chunk(t0_ms, samples)
            self._try_process_window(state)

    def handle_imu_chunk(self, patient_id: str, t0_ms: int, fs_hz: int, samples: np.ndarray) -> None:
        """Handle incoming IMU chunk."""
        with self.lock:
            state = self.get_or_create_patient(patient_id)
            if state.imu_buffer.fs_hz != fs_hz:
                state.imu_buffer = ImuBuffer(fs_hz=fs_hz)
            state.imu_buffer.append_chunk(t0_ms, samples)

    def handle_spo2_sample(self, patient_id: str, t_ms: int, value: float) -> None:
        """Handle incoming SpO2 sample."""
        with self.lock:
            state = self.get_or_create_patient(patient_id)
            state.spo2_buffer.add_sample(t_ms, value)

    def handle_temp_sample(self, patient_id: str, t_ms: int, value: float) -> None:
        """Handle incoming temperature sample."""
        with self.lock:
            state = self.get_or_create_patient(patient_id)
            state.temp_buffer.add_sample(t_ms, value)

    def _try_process_window(self, state: PatientState) -> None:
        """Try to process a complete window if enough data available."""
        window_sec = self.config.window_sec
        ecg_window_samples = int(window_sec * state.ecg_buffer.fs_hz)

        # Check if ECG has enough data (ECG is the primary timing reference)
        if not state.ecg_buffer.has_window(ecg_window_samples):
            return

        # Extract ECG window
        ecg_raw, window_start_ms = state.ecg_buffer.extract_window(ecg_window_samples)
        window_end_ms = window_start_ms + int(window_sec * 1000)

        # Extract other streams aligned to this window
        ppg_window_samples = int(window_sec * state.ppg_buffer.fs_hz)
        ppg_raw, _ = state.ppg_buffer.extract_window(ppg_window_samples)

        imu_window_samples = int(window_sec * state.imu_buffer.fs_hz)
        imu_raw, _ = state.imu_buffer.extract_window(imu_window_samples)

        spo2_values = state.spo2_buffer.extract_window(window_start_ms, window_end_ms)
        temp_values = state.temp_buffer.extract_window(window_start_ms, window_end_ms)

        # Process IMU first (needed for motion artifact detection)
        motion_score, motion_notes = compute_motion_score(imu_raw)

        # Process each sensor
        ecg_features = process_ecg_window(
            ecg_raw, state.ecg_buffer.fs_hz, self.config.notch_hz, motion_score
        )
        ppg_features = process_ppg_window(ppg_raw, state.ppg_buffer.fs_hz, motion_score)
        spo2_features, spo2_events = process_spo2_window(spo2_values, state.spo2_drop_state)
        temp_features = process_temp_window(temp_values)
        motion_features = {"score": round(motion_score, 3), "notes": motion_notes}

        # Update last HR values
        if ecg_features["hr_bpm"] is not None:
            state.last_hr_ecg = ecg_features["hr_bpm"]
        if ppg_features["hr_bpm"] is not None:
            state.last_hr_ppg = ppg_features["hr_bpm"]

        # Build features message
        features = create_features(
            patient_id=state.patient_id,
            window_start_ms=window_start_ms,
            window_sec=window_sec,
            ecg=ecg_features,
            ppg=ppg_features,
            spo2=spo2_features,
            temp_c=temp_features,
            motion=motion_features
        )

        # Publish features
        self.client.publish(self.output_topics["features"], to_json(features), qos=1)
        print(f"[edge] patient={state.patient_id} window={window_start_ms} "
              f"ecg_hr={ecg_features['hr_bpm']} ppg_hr={ppg_features['hr_bpm']} "
              f"spo2={spo2_features['mean']} temp={temp_features.get('mean')} motion={motion_score:.2f}")

        # Generate events
        events: List[Dict[str, Any]] = []

        # SpO2 events
        for evt in spo2_events:
            events.append(create_event(
                patient_id=state.patient_id,
                t_ms=window_start_ms,
                event_type=evt["type"],
                severity=evt["severity"],
                details=evt["details"]
            ))

        # HR events (tachycardia/bradycardia)
        hr_val = ecg_features["hr_bpm"] or ppg_features["hr_bpm"]
        if hr_val is not None:
            if hr_val > 100:
                severity = Severity.HIGH if hr_val > 150 else Severity.MODERATE if hr_val > 120 else Severity.LOW
                events.append(create_event(
                    patient_id=state.patient_id,
                    t_ms=window_start_ms,
                    event_type=EventType.TACHYCARDIA,
                    severity=severity,
                    details={"hr_bpm": hr_val}
                ))
            elif hr_val < 50:
                severity = Severity.HIGH if hr_val < 40 else Severity.MODERATE if hr_val < 45 else Severity.LOW
                events.append(create_event(
                    patient_id=state.patient_id,
                    t_ms=window_start_ms,
                    event_type=EventType.BRADYCARDIA,
                    severity=severity,
                    details={"hr_bpm": hr_val}
                ))

        # Low quality event
        avg_sqi = (ecg_features["sqi"] + ppg_features["sqi"]) / 2
        if avg_sqi < 0.4:
            events.append(create_event(
                patient_id=state.patient_id,
                t_ms=window_start_ms,
                event_type=EventType.LOW_QUALITY,
                severity=Severity.MODERATE if avg_sqi < 0.2 else Severity.LOW,
                details={"ecg_sqi": ecg_features["sqi"], "ppg_sqi": ppg_features["sqi"]}
            ))

        # Motion artifact event
        if motion_score > 0.5:
            events.append(create_event(
                patient_id=state.patient_id,
                t_ms=window_start_ms,
                event_type=EventType.MOTION_ARTIFACT,
                severity=Severity.HIGH if motion_score > 0.8 else Severity.MODERATE,
                details={"motion_score": motion_score}
            ))

        # Publish events
        for evt in events:
            self.client.publish(self.output_topics["events"], to_json(evt), qos=1)
            print(f"[edge] EVENT: {evt['type']} severity={evt['severity']} details={evt['details']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge gateway: preprocess multi-sensor data + publish features/events"
    )
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--disable-notch", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # Legacy compatibility
    parser.add_argument("--input-topic", default=None, help="(Deprecated) Use device-id instead")
    parser.add_argument("--output-topic", default=None, help="(Deprecated) Use device-id instead")

    args = parser.parse_args()

    # Print capabilities
    print(f"[edge] SciPy available: {SCIPY_AVAILABLE}")
    if not SCIPY_AVAILABLE:
        print("[edge] WARNING: Running without SciPy - using basic preprocessing only")

    # Create configuration
    config = SensorConfig(
        window_sec=args.window_sec,
        notch_hz=None if args.disable_notch else args.notch_hz
    )

    # Create MQTT client
    mqtt_cfg = MqttConfig(host=args.broker_host, port=args.broker_port)
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="edge-preprocessor",
    )

    # Create processor
    processor = EdgePreprocessor(
        mqtt_client=client,
        device_id=args.device_id,
        config=config,
        debug=args.debug
    )

    def on_connect(c: mqtt.Client, _userdata, _connect_flags, reason_code, _properties):
        if reason_code == 0:
            print(f"[edge] connected to MQTT {mqtt_cfg.host}:{mqtt_cfg.port}")

            # Subscribe to all input topics
            for sensor, topic in processor.input_topics.items():
                c.subscribe(topic, qos=1)
                print(f"[edge] subscribed to {topic}")
        else:
            print(f"[edge] MQTT connection failed reason={reason_code}")

    def on_disconnect(_c: mqtt.Client, _userdata, _disconnect_flags, reason_code, _properties):
        print(f"[edge] disconnected reason={reason_code}")

    def on_message(_c: mqtt.Client, _userdata, msg: mqtt.MQTTMessage):
        try:
            payload = parse_json(msg.payload)
        except Exception as ex:
            print(f"[edge] invalid JSON payload: {ex}")
            return

        # Parse topic to determine sensor type
        prefix, device_id, sensor_type = parse_topic(msg.topic)
        if sensor_type is None:
            print(f"[edge] unknown topic format: {msg.topic}")
            return

        patient_id = payload.get("patient_id", device_id or "unknown")

        if sensor_type == "ecg":
            err = validate_chunk_message(payload)
            if err:
                print(f"[edge] invalid ECG message: {err}")
                return
            samples = np.asarray(safe_float_list(payload["samples"]), dtype=float)
            processor.handle_ecg_chunk(
                patient_id, int(payload["t0_ms"]), int(payload["fs_hz"]), samples
            )

        elif sensor_type == "ppg":
            err = validate_chunk_message(payload)
            if err:
                print(f"[edge] invalid PPG message: {err}")
                return
            samples = np.asarray(safe_float_list(payload["samples"]), dtype=float)
            processor.handle_ppg_chunk(
                patient_id, int(payload["t0_ms"]), int(payload["fs_hz"]), samples
            )

        elif sensor_type == "imu":
            err = validate_imu_chunk_message(payload)
            if err:
                print(f"[edge] invalid IMU message: {err}")
                return
            triplets = safe_imu_triplet_list(payload["samples"])
            samples = np.array(triplets, dtype=float)
            processor.handle_imu_chunk(
                patient_id, int(payload["t0_ms"]), int(payload["fs_hz"]), samples
            )

        elif sensor_type == "spo2":
            err = validate_sample_message(payload)
            if err:
                print(f"[edge] invalid SpO2 message: {err}")
                return
            processor.handle_spo2_sample(
                patient_id, int(payload["t_ms"]), float(payload["value"])
            )

        elif sensor_type == "temp":
            err = validate_sample_message(payload)
            if err:
                print(f"[edge] invalid Temp message: {err}")
                return
            processor.handle_temp_sample(
                patient_id, int(payload["t_ms"]), float(payload["value"])
            )

        else:
            if args.debug:
                print(f"[edge] ignoring unknown sensor type: {sensor_type}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    print(f"[edge] connecting to {mqtt_cfg.host}:{mqtt_cfg.port}...")
    while True:
        try:
            client.connect(mqtt_cfg.host, mqtt_cfg.port, mqtt_cfg.keepalive)
            break
        except OSError as ex:
            print(f"[edge] broker not reachable ({ex}); retrying in 2s...")
            time.sleep(2)

    client.loop_start()

    print("[edge] running... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[edge] stopping")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
