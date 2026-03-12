from __future__ import annotations

"""
Feature Extraction Module — Smart Medical Mat
===============================================
Implements all features from:
  - Smart_Medical_Mat_Feature_Extraction_Complet.pdf
  - medical_signal_formulas_explained.pdf

Sensors covered:
  1. ECG (ADS1292R): HR, HRV, QRS, abnormal beats
  2. Respiration (derived from ECG impedance / PPG modulation)
  3. SpO2 + PPG (MAX86141): SpO2 stats, pulse rate, amplitude, PTT
  4. Temperature (TMP117): mean, variation, fever detection
  5. Movement / IMU (ICM-42688): magnitude, count, immobility, agitation
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, detrend, filtfilt, find_peaks, iirnotch

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# 1. ECG FEATURE EXTRACTION (ADS1292R)
# =============================================================================

def preprocess_ecg(window: np.ndarray, fs_hz: int, notch_hz: Optional[float] = 50.0) -> np.ndarray:
    x = window.astype(float)
    if SCIPY_AVAILABLE:
        x = detrend(x, type="linear")
        nyq = fs_hz / 2.0
        low, high = 0.5 / nyq, min(40.0 / nyq, 0.99)
        b, a = butter(3, [low, high], btype="band")
        x = filtfilt(b, a, x)
        if notch_hz is not None and 0 < notch_hz < nyq:
            b_n, a_n = iirnotch(w0=notch_hz / nyq, Q=30)
            x = filtfilt(b_n, a_n, x)
    else:
        x = x - np.mean(x)
    return x


def detect_r_peaks(filtered: np.ndarray, fs_hz: int) -> np.ndarray:
    if filtered.size < fs_hz:
        return np.array([], dtype=int)
    signal_std = float(np.std(filtered))
    if signal_std < 1e-6:
        return np.array([], dtype=int)
    threshold = max(0.35 * signal_std, 0.05)
    min_distance = int((60.0 / 200.0) * fs_hz)  # max 200 bpm
    if SCIPY_AVAILABLE:
        peaks, _ = find_peaks(filtered, distance=min_distance, prominence=threshold)
        return peaks.astype(int)
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


def compute_rr_intervals(peaks: np.ndarray, fs_hz: int) -> np.ndarray:
    """RR_i = t(R_{i+1}) - t(R_i) in seconds."""
    if len(peaks) < 2:
        return np.array([], dtype=float)
    rr = np.diff(peaks) / float(fs_hz)
    # Keep only physiological intervals (0.3s–2.0s = 30–200 bpm)
    return rr[(rr > 0.3) & (rr < 2.0)]


def compute_hr_from_rr(rr_intervals: np.ndarray) -> Dict[str, Optional[float]]:
    """HR = 60 / RR for each interval, then mean/min/max."""
    if len(rr_intervals) == 0:
        return {"hr_mean": None, "hr_min": None, "hr_max": None}
    hr_all = 60.0 / rr_intervals
    return {
        "hr_mean": round(float(np.mean(hr_all)), 2),
        "hr_min": round(float(np.min(hr_all)), 2),
        "hr_max": round(float(np.max(hr_all)), 2),
    }


def compute_hrv_sdnn(rr_intervals: np.ndarray) -> Optional[float]:
    """SDNN = std(RR intervals)."""
    if len(rr_intervals) < 2:
        return None
    return round(float(np.std(rr_intervals, ddof=1)) * 1000, 2)  # ms


def compute_hrv_rmssd(rr_intervals: np.ndarray) -> Optional[float]:
    """RMSSD = sqrt(mean(diff(RR)^2))."""
    if len(rr_intervals) < 2:
        return None
    diffs = np.diff(rr_intervals)
    return round(float(np.sqrt(np.mean(diffs ** 2))) * 1000, 2)  # ms


def compute_qrs_duration(filtered: np.ndarray, peaks: np.ndarray, fs_hz: int) -> Optional[float]:
    """Estimate QRS = t(S) - t(Q) using a fixed search window around R peak."""
    if len(peaks) == 0:
        return None
    qrs_durations: List[float] = []
    search_before = int(0.06 * fs_hz)  # 60ms before R
    search_after = int(0.06 * fs_hz)   # 60ms after R
    for rp in peaks:
        q_start = max(0, rp - search_before)
        s_end = min(len(filtered), rp + search_after)
        if q_start >= s_end:
            continue
        segment = filtered[q_start:s_end]
        # Q = first local minimum before peak, S = first local minimum after peak
        before_peak = segment[: rp - q_start]
        after_peak = segment[rp - q_start:]
        if len(before_peak) > 0 and len(after_peak) > 0:
            q_idx = q_start + int(np.argmin(before_peak))
            s_idx = rp + int(np.argmin(after_peak))
            dur_ms = (s_idx - q_idx) / float(fs_hz) * 1000
            if 40 < dur_ms < 200:  # physiological range
                qrs_durations.append(dur_ms)
    if not qrs_durations:
        return None
    return round(float(np.median(qrs_durations)), 2)


def detect_abnormal_beats(rr_intervals: np.ndarray, threshold_factor: float = 1.5) -> int:
    """|RR_i - RR_mean| > threshold * std(RR) -> abnormal."""
    if len(rr_intervals) < 3:
        return 0
    rr_mean = np.mean(rr_intervals)
    rr_std = np.std(rr_intervals)
    if rr_std < 1e-6:
        return 0
    abnormal = np.sum(np.abs(rr_intervals - rr_mean) > threshold_factor * rr_std)
    return int(abnormal)


def compute_ecg_sqi(raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray, fs_hz: int) -> float:
    score = 1.0
    raw_std = float(np.std(raw))
    if raw_std < 1e-4:
        score -= 0.8
    diff = np.abs(np.diff(raw)) if raw.size > 1 else np.array([0.0])
    flat_ratio = float(np.mean(diff < 1e-5))
    if flat_ratio > 0.2:
        score -= 0.4
    hf_noise = raw - filtered
    noise_ratio = float(np.std(hf_noise) / (np.std(filtered) + 1e-6))
    if noise_ratio > 1.0:
        score -= 0.3
    elif noise_ratio > 0.7:
        score -= 0.15
    expected_min, expected_max = 3, 15
    if len(peaks) < expected_min or len(peaks) > expected_max:
        score -= 0.2
    return float(np.clip(score, 0.0, 1.0))


def extract_ecg_features(
    raw: np.ndarray, fs_hz: int,
    notch_hz: Optional[float] = 50.0,
    motion_score: float = 0.0,
) -> Dict[str, Any]:
    """Full ECG feature extraction per PDF spec."""
    if len(raw) < fs_hz:
        return {
            "hr_mean": None, "hr_min": None, "hr_max": None,
            "rr_mean_ms": None,
            "hrv_sdnn": None, "hrv_rmssd": None,
            "qrs_duration_ms": None, "abnormal_beats": 0,
            "sqi": 0.0, "notes": ["insufficient_data"],
        }

    filtered = preprocess_ecg(raw, fs_hz, notch_hz)
    peaks = detect_r_peaks(filtered, fs_hz)
    rr = compute_rr_intervals(peaks, fs_hz)
    hr = compute_hr_from_rr(rr)
    sqi = compute_ecg_sqi(raw, filtered, peaks, fs_hz)

    notes: List[str] = []
    if hr["hr_mean"] is None:
        notes.append("insufficient_peaks")
    if sqi < 0.5:
        notes.append("low_sqi")
    if motion_score > 0.5:
        notes.append("motion_artifact")
        sqi = max(0.0, sqi - 0.2)

    return {
        "hr_mean": hr["hr_mean"],
        "hr_min": hr["hr_min"],
        "hr_max": hr["hr_max"],
        "rr_mean_ms": round(float(np.mean(rr)) * 1000, 2) if len(rr) > 0 else None,
        "hrv_sdnn": compute_hrv_sdnn(rr),
        "hrv_rmssd": compute_hrv_rmssd(rr),
        "qrs_duration_ms": compute_qrs_duration(filtered, peaks, fs_hz),
        "abnormal_beats": detect_abnormal_beats(rr),
        "sqi": round(sqi, 3),
        "notes": notes,
        # Internal: pass peaks for PTT computation
        "_peaks": peaks,
        "_filtered": filtered,
    }


# =============================================================================
# 2. RESPIRATION FEATURES (derived from ECG/PPG)
# =============================================================================

def extract_respiration_features(
    ecg_filtered: np.ndarray, fs_hz: int,
) -> Dict[str, Any]:
    """Estimate respiration from ECG amplitude modulation (R-peak envelope)."""
    if not SCIPY_AVAILABLE or len(ecg_filtered) < fs_hz * 3:
        return {"rate_bpm": None, "amplitude": None}

    # Compute envelope via R-peak amplitudes
    peaks = detect_r_peaks(ecg_filtered, fs_hz)
    if len(peaks) < 4:
        return {"rate_bpm": None, "amplitude": None}

    peak_amps = ecg_filtered[peaks]
    peak_times = peaks / float(fs_hz)

    # Interpolate to uniform sampling (4 Hz for respiration analysis)
    resp_fs = 4.0
    t_uniform = np.arange(peak_times[0], peak_times[-1], 1.0 / resp_fs)
    if len(t_uniform) < 4:
        return {"rate_bpm": None, "amplitude": None}
    envelope = np.interp(t_uniform, peak_times, peak_amps)
    envelope = envelope - np.mean(envelope)

    # Bandpass 0.1–0.5 Hz (6–30 breaths/min)
    nyq = resp_fs / 2.0
    low, high = 0.1 / nyq, min(0.5 / nyq, 0.99)
    b, a = butter(2, [low, high], btype="band")
    resp_signal = filtfilt(b, a, envelope)

    # Count zero crossings (positive) -> breathing cycles
    crossings = np.where(np.diff(np.sign(resp_signal)) > 0)[0]
    duration_sec = len(t_uniform) / resp_fs
    if duration_sec < 1 or len(crossings) < 1:
        return {"rate_bpm": None, "amplitude": None}

    rate_bpm = round((len(crossings) / duration_sec) * 60, 1)
    amplitude = round(float(np.max(resp_signal) - np.min(resp_signal)), 4)

    # Sanity check
    if rate_bpm < 4 or rate_bpm > 60:
        return {"rate_bpm": None, "amplitude": amplitude}

    return {"rate_bpm": rate_bpm, "amplitude": amplitude}


# =============================================================================
# 3. PPG + SpO2 FEATURES (MAX86141)
# =============================================================================

def preprocess_ppg(window: np.ndarray, fs_hz: int) -> np.ndarray:
    x = window.astype(float)
    if SCIPY_AVAILABLE:
        x = detrend(x, type="linear")
        nyq = fs_hz / 2.0
        low, high = 0.5 / nyq, min(8.0 / nyq, 0.99)
        b, a = butter(2, [low, high], btype="band")
        x = filtfilt(b, a, x)
    else:
        x = x - np.mean(x)
    return x


def detect_ppg_peaks(filtered: np.ndarray, fs_hz: int) -> np.ndarray:
    if filtered.size < fs_hz:
        return np.array([], dtype=int)
    signal_std = float(np.std(filtered))
    if signal_std < 1e-6:
        return np.array([], dtype=int)
    threshold = max(0.3 * signal_std, 0.02)
    min_distance = int((60.0 / 200.0) * fs_hz)
    if SCIPY_AVAILABLE:
        peaks, _ = find_peaks(filtered, distance=min_distance, prominence=threshold)
        return peaks.astype(int)
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


def compute_ppg_sqi(raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray) -> float:
    score = 1.0
    raw_std = float(np.std(raw))
    if raw_std < 1e-4:
        score -= 0.8
    if len(peaks) >= 3:
        intervals = np.diff(peaks)
        interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
        if interval_cv > 0.3:
            score -= 0.3
    hf_noise = raw - filtered
    noise_ratio = float(np.std(hf_noise) / (np.std(filtered) + 1e-6))
    if noise_ratio > 1.5:
        score -= 0.3
    return float(np.clip(score, 0.0, 1.0))


def compute_ptt(
    ecg_peaks: np.ndarray, ppg_peaks: np.ndarray,
    ecg_fs: int, ppg_fs: int,
) -> Optional[float]:
    """PTT = t(PPG_peak) - t(R_peak). Returns median PTT in ms."""
    if len(ecg_peaks) == 0 or len(ppg_peaks) == 0:
        return None
    ecg_times = ecg_peaks / float(ecg_fs)
    ppg_times = ppg_peaks / float(ppg_fs)
    ptts: List[float] = []
    for r_t in ecg_times:
        # Find the first PPG peak after this R peak
        candidates = ppg_times[ppg_times > r_t]
        if len(candidates) == 0:
            continue
        ptt_sec = candidates[0] - r_t
        if 0.05 < ptt_sec < 0.5:  # physiological range 50–500 ms
            ptts.append(ptt_sec * 1000)
    if not ptts:
        return None
    return round(float(np.median(ptts)), 2)


def extract_ppg_features(
    raw: np.ndarray, fs_hz: int,
    motion_score: float = 0.0,
    ecg_peaks: Optional[np.ndarray] = None,
    ecg_fs: Optional[int] = None,
) -> Dict[str, Any]:
    """Full PPG feature extraction per PDF spec."""
    if len(raw) < fs_hz:
        return {
            "pulse_rate": None, "amplitude": None,
            "ptt_ms": None, "sqi": 0.0, "notes": ["insufficient_data"],
        }

    filtered = preprocess_ppg(raw, fs_hz)
    peaks = detect_ppg_peaks(filtered, fs_hz)

    # Pulse rate (like HR from PPG)
    rr = compute_rr_intervals(peaks, fs_hz)
    pulse_rate = round(float(60.0 / np.mean(rr)), 2) if len(rr) > 0 else None

    # Amplitude = PPGmax - PPGmin
    amplitude = round(float(np.max(filtered) - np.min(filtered)), 4)

    # PTT
    ptt = None
    if ecg_peaks is not None and ecg_fs is not None:
        ptt = compute_ptt(ecg_peaks, peaks, ecg_fs, fs_hz)

    sqi = compute_ppg_sqi(raw, filtered, peaks)
    notes: List[str] = []
    if pulse_rate is None:
        notes.append("insufficient_peaks")
    if sqi < 0.5:
        notes.append("low_sqi")
    if motion_score > 0.5:
        notes.append("motion_artifact")
        sqi = max(0.0, sqi - 0.3)

    return {
        "pulse_rate": pulse_rate,
        "amplitude": amplitude,
        "ptt_ms": ptt,
        "sqi": round(sqi, 3),
        "notes": notes,
    }


# =============================================================================
# 4. SpO2 FEATURES (MAX86141)
# =============================================================================

def extract_spo2_features(
    values: List[float],
    window_sec: float,
    drop_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """SpO2 feature extraction: mean, min, desaturation count/index."""
    events: List[Dict[str, Any]] = []

    if not values:
        return {
            "mean": None, "min": None,
            "desaturation_count": 0, "desat_index_per_hr": None,
        }, events

    mean_val = round(float(np.mean(values)), 1)
    min_val = round(float(np.min(values)), 1)

    # Desaturation: SpO2 < 90%
    desat_events = [v for v in values if v < 90.0]
    desat_count = drop_state.get("total_desats", 0)

    if desat_events:
        drop_state["below_time_sec"] = drop_state.get("below_time_sec", 0) + len(desat_events)
        drop_state["min_value"] = min(drop_state.get("min_value", 100), min(desat_events))

        if drop_state["below_time_sec"] >= 10 and not drop_state.get("drop_reported", False):
            desat_count += 1
            drop_state["total_desats"] = desat_count
            drop_state["drop_reported"] = True
            events.append({
                "type": "spo2_drop",
                "min_value": round(drop_state["min_value"], 1),
                "duration_sec": drop_state["below_time_sec"],
            })
    else:
        drop_state["below_time_sec"] = 0
        drop_state["min_value"] = 100
        drop_state["drop_reported"] = False

    # Desaturation index = desats per hour
    monitoring_hours = drop_state.get("monitoring_sec", 0) + window_sec
    drop_state["monitoring_sec"] = monitoring_hours
    desat_index = None
    if monitoring_hours > 0:
        desat_index = round(desat_count / (monitoring_hours / 3600.0), 2) if monitoring_hours > 60 else None

    return {
        "mean": mean_val,
        "min": min_val,
        "desaturation_count": desat_count,
        "desat_index_per_hr": desat_index,
    }, events


# =============================================================================
# 5. TEMPERATURE FEATURES (TMP117)
# =============================================================================

def extract_temp_features(values: List[float]) -> Dict[str, Any]:
    """Temperature: mean, variation, fever detection."""
    if not values:
        return {"mean": None, "variation": None, "fever": False}

    valid = [v for v in values if 30.0 <= v <= 45.0]
    if not valid:
        return {"mean": None, "variation": None, "fever": False}

    mean_val = round(float(np.mean(valid)), 2)
    variation = round(float(max(valid) - min(valid)), 2)
    fever = mean_val > 38.0

    return {"mean": mean_val, "variation": variation, "fever": fever}


# =============================================================================
# 6. MOVEMENT / IMU FEATURES (ICM-42688)
# =============================================================================

def extract_motion_features(
    imu_window: np.ndarray,
    fs_hz: int,
    magnitude_threshold: float = 1.2,
) -> Dict[str, Any]:
    """Movement features: magnitude, count, immobility, agitation index."""
    if imu_window.size == 0:
        return {
            "mvt_count": 0, "immobility_sec": 0.0,
            "agitation_index": 0.0, "score": 0.0,
            "notes": ["no_imu_data"],
        }

    # Magnitude = sqrt(x^2 + y^2 + z^2)
    magnitudes = np.sqrt(np.sum(imu_window ** 2, axis=1))
    duration_sec = len(magnitudes) / float(fs_hz)

    # Movement count: samples where magnitude > threshold (above gravity ~1g)
    above = magnitudes > magnitude_threshold
    mvt_count = int(np.sum(above))

    # Immobility: time where magnitude is near 1g (still)
    still_mask = magnitudes < magnitude_threshold
    immobility_sec = round(float(np.sum(still_mask)) / fs_hz, 2)

    # Agitation index = movements per second
    agitation_index = round(mvt_count / max(duration_sec, 0.01), 3)

    # Normalized motion score 0–1
    variance = float(np.var(magnitudes - 1.0))
    score = float(np.clip(variance / 0.5, 0.0, 1.0))

    notes: List[str] = []
    if score > 0.5:
        notes.append("high_motion")
    elif score > 0.2:
        notes.append("moderate_motion")

    return {
        "mvt_count": mvt_count,
        "immobility_sec": immobility_sec,
        "agitation_index": agitation_index,
        "score": round(score, 3),
        "notes": notes,
    }
