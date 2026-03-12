from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# TOPIC DEFINITIONS
# =============================================================================
# Input topics (from ESP32/simulator) - pattern: sim/<device_id>/<sensor>
# In production, replace "sim" with "mat" for real ESP32 hardware.

DEFAULT_DEVICE_ID = "patient1"

# High-rate chunk streams (ECG, PPG, IMU)
TOPIC_ECG_RAW = "sim/{device_id}/ecg"
TOPIC_PPG_RAW = "sim/{device_id}/ppg"
TOPIC_IMU_RAW = "sim/{device_id}/imu"

# Low-rate sample streams (SpO2, Temperature)
TOPIC_SPO2 = "sim/{device_id}/spo2"
TOPIC_TEMP = "sim/{device_id}/temp"

# Output topics (from edge gateway)
TOPIC_FEATURES = "edge/{device_id}/features"
TOPIC_EVENTS = "edge/{device_id}/events"

# Legacy compatibility (ECG-only mode)
INPUT_TOPIC = TOPIC_ECG_RAW.format(device_id=DEFAULT_DEVICE_ID)
OUTPUT_TOPIC = TOPIC_FEATURES.format(device_id=DEFAULT_DEVICE_ID)


def get_input_topics(device_id: str) -> Dict[str, str]:
    """Get all input topics for a device."""
    return {
        "ecg": TOPIC_ECG_RAW.format(device_id=device_id),
        "ppg": TOPIC_PPG_RAW.format(device_id=device_id),
        "imu": TOPIC_IMU_RAW.format(device_id=device_id),
        "spo2": TOPIC_SPO2.format(device_id=device_id),
        "temp": TOPIC_TEMP.format(device_id=device_id),
    }


def get_output_topics(device_id: str) -> Dict[str, str]:
    """Get all output topics for a device."""
    return {
        "features": TOPIC_FEATURES.format(device_id=device_id),
        "events": TOPIC_EVENTS.format(device_id=device_id),
    }


def parse_topic(topic: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse topic string into (prefix, device_id, sensor_type).
    Example: "sim/patient1/ecg" -> ("sim", "patient1", "ecg")
    """
    parts = topic.split("/")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], parts[1], None
    return None, None, None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MqttConfig:
    host: str = "localhost"
    port: int = 1883
    keepalive: int = 30


@dataclass
class SensorConfig:
    """Configuration for sensor streams."""
    # Sampling rates (Hz)
    ecg_fs_hz: int = 250
    ppg_fs_hz: int = 100
    imu_fs_hz: int = 50
    spo2_interval_sec: float = 1.0
    temp_interval_sec: float = 5.0

    # Processing parameters
    window_sec: float = 5.0
    notch_hz: float = 50.0

    # Thresholds
    hr_min_bpm: float = 40.0
    hr_max_bpm: float = 180.0
    spo2_drop_threshold: float = 93.0
    spo2_drop_duration_sec: float = 10.0
    temp_min_c: float = 30.0
    temp_max_c: float = 45.0
    motion_high_threshold: float = 0.5


# =============================================================================
# TIME UTILITIES
# =============================================================================

def now_ms() -> int:
    """Current time in milliseconds since epoch."""
    return int(time.time() * 1000)


def chunk_duration_ms(samples_len: int, fs_hz: float) -> int:
    """Calculate duration of a chunk in milliseconds."""
    return int(round((samples_len / float(fs_hz)) * 1000.0))


def samples_in_window(window_sec: float, fs_hz: int) -> int:
    """Calculate number of samples in a time window."""
    return int(round(window_sec * fs_hz))


# =============================================================================
# JSON UTILITIES
# =============================================================================

def to_json(payload: Dict[str, Any]) -> str:
    """Serialize payload to compact JSON string."""
    return json.dumps(payload, separators=(",", ":"))


def parse_json(raw: bytes) -> Dict[str, Any]:
    """Parse JSON bytes to dictionary."""
    return json.loads(raw.decode("utf-8"))


# =============================================================================
# MESSAGE VALIDATION
# =============================================================================

def validate_chunk_message(payload: Dict[str, Any]) -> Optional[str]:
    """
    Validate high-rate chunk message (ECG, PPG, IMU).
    Schema: {patient_id, t0_ms, fs_hz, samples}
    """
    required = ["patient_id", "t0_ms", "fs_hz", "samples"]
    for key in required:
        if key not in payload:
            return f"missing_key:{key}"

    if not isinstance(payload["patient_id"], str):
        return "invalid_patient_id"
    if not isinstance(payload["t0_ms"], (int, float)):
        return "invalid_t0_ms"
    if not isinstance(payload["fs_hz"], (int, float)):
        return "invalid_fs_hz"
    if not isinstance(payload["samples"], list):
        return "invalid_samples"
    if len(payload["samples"]) == 0:
        return "empty_samples"

    return None


def validate_sample_message(payload: Dict[str, Any]) -> Optional[str]:
    """
    Validate low-rate sample message (SpO2, Temperature).
    Schema: {patient_id, t_ms, value}
    """
    required = ["patient_id", "t_ms", "value"]
    for key in required:
        if key not in payload:
            return f"missing_key:{key}"

    if not isinstance(payload["patient_id"], str):
        return "invalid_patient_id"
    if not isinstance(payload["t_ms"], (int, float)):
        return "invalid_t_ms"
    if not isinstance(payload["value"], (int, float)):
        return "invalid_value"

    return None


def validate_imu_chunk_message(payload: Dict[str, Any]) -> Optional[str]:
    """
    Validate IMU chunk message with triplet samples.
    Schema: {patient_id, t0_ms, fs_hz, samples: [[ax,ay,az], ...]}
    """
    err = validate_chunk_message(payload)
    if err:
        return err

    # Validate that samples are triplets
    for i, sample in enumerate(payload["samples"]):
        if not isinstance(sample, list) or len(sample) != 3:
            return f"invalid_imu_sample_at_{i}"

    return None


# Legacy alias for backward compatibility
def validate_input_message(payload: Dict[str, Any]) -> Optional[str]:
    """Legacy validation for ECG chunk messages."""
    return validate_chunk_message(payload)


def validate_features_message(payload: Dict[str, Any]) -> Optional[str]:
    """
    Validate unified features output message.
    Schema: {patient_id, window_start_ms, window_sec, ecg, ppg, spo2, temp_c, motion}
    """
    required = ["patient_id", "window_start_ms", "window_sec"]
    for key in required:
        if key not in payload:
            return f"missing_key:{key}"

    if not isinstance(payload["patient_id"], str):
        return "invalid_patient_id"
    if not isinstance(payload["window_start_ms"], (int, float)):
        return "invalid_window_start_ms"
    if not isinstance(payload["window_sec"], (int, float)):
        return "invalid_window_sec"

    return None


def validate_event_message(payload: Dict[str, Any]) -> Optional[str]:
    """
    Validate event output message.
    Schema: {patient_id, t_ms, type, severity, details}
    """
    required = ["patient_id", "t_ms", "type", "severity"]
    for key in required:
        if key not in payload:
            return f"missing_key:{key}"

    valid_types = {
        "spo2_drop", "hypoxemia", "tachycardia", "bradycardia",
        "fever", "apnea_suspicion",
        "low_quality", "motion_artifact", "sensor_detached"
    }
    if payload["type"] not in valid_types:
        return f"invalid_event_type:{payload['type']}"

    valid_severities = {"low", "moderate", "high", "critical"}
    if payload["severity"] not in valid_severities:
        return f"invalid_severity:{payload['severity']}"

    return None


# Legacy alias
def validate_output_message(payload: Dict[str, Any]) -> Optional[str]:
    """Legacy validation for old ECG-only output format."""
    required = ["patient_id", "window_start_ms", "window_sec", "hr_bpm", "sqi", "notes"]
    for key in required:
        if key not in payload:
            return f"missing_key:{key}"
    return None


# =============================================================================
# DATA CONVERSION UTILITIES
# =============================================================================

def normalize_topic_patient(topic: str) -> Optional[str]:
    """Extract patient/device ID from topic."""
    _, device_id, _ = parse_topic(topic)
    return device_id


def safe_float_list(values: List[Any]) -> List[float]:
    """Convert list of values to floats, replacing invalid with 0.0."""
    out: List[float] = []
    for val in values:
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def safe_imu_triplet_list(values: List[Any]) -> List[Tuple[float, float, float]]:
    """Convert list of IMU triplets to tuples of floats."""
    out: List[Tuple[float, float, float]] = []
    for val in values:
        try:
            if isinstance(val, (list, tuple)) and len(val) >= 3:
                out.append((float(val[0]), float(val[1]), float(val[2])))
            else:
                out.append((0.0, 0.0, 0.0))
        except (TypeError, ValueError):
            out.append((0.0, 0.0, 0.0))
    return out


# =============================================================================
# EVENT TYPES AND SEVERITY
# =============================================================================

class EventType:
    SPO2_DROP = "spo2_drop"
    HYPOXEMIA = "hypoxemia"
    TACHYCARDIA = "tachycardia"
    BRADYCARDIA = "bradycardia"
    FEVER = "fever"
    APNEA_SUSPICION = "apnea_suspicion"
    LOW_QUALITY = "low_quality"
    MOTION_ARTIFACT = "motion_artifact"
    SENSOR_DETACHED = "sensor_detached"


class Severity:
    LOW = "low"          # VERT  — surveillance standard
    MODERATE = "moderate"  # JAUNE — vérification demandée
    HIGH = "high"        # ORANGE — intervention recommandée
    CRITICAL = "critical"  # ROUGE — alerte immédiate


# Alert color mapping (PDF Table 6)
SEVERITY_COLOR = {
    Severity.LOW: "green",
    Severity.MODERATE: "yellow",
    Severity.HIGH: "orange",
    Severity.CRITICAL: "red",
}


def create_event(
    patient_id: str,
    t_ms: int,
    event_type: str,
    severity: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a properly formatted event message."""
    return {
        "patient_id": patient_id,
        "t_ms": t_ms,
        "type": event_type,
        "severity": severity,
        "details": details or {}
    }


def create_features(
    patient_id: str,
    window_start_ms: int,
    window_sec: float,
    ecg: Optional[Dict[str, Any]] = None,
    ppg: Optional[Dict[str, Any]] = None,
    spo2: Optional[Dict[str, Any]] = None,
    temp_c: Optional[Dict[str, Any]] = None,
    motion: Optional[Dict[str, Any]] = None,
    respiration: Optional[Dict[str, Any]] = None,
    ml: Optional[Dict[str, Any]] = None,
    rules: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a properly formatted features message (PDF-compliant)."""
    return {
        "patient_id": patient_id,
        "window_start_ms": window_start_ms,
        "window_sec": window_sec,
        # --- ECG features (ADS1292R) ---
        "ecg": ecg or {
            "hr_mean": None, "hr_min": None, "hr_max": None,
            "rr_mean_ms": None,
            "hrv_sdnn": None, "hrv_rmssd": None,
            "qrs_duration_ms": None,
            "abnormal_beats": 0,
            "sqi": 0.0, "notes": ["no_data"],
        },
        # --- PPG features (MAX86141) ---
        "ppg": ppg or {
            "pulse_rate": None, "amplitude": None,
            "ptt_ms": None,
            "sqi": 0.0, "notes": ["no_data"],
        },
        # --- SpO2 features (MAX86141) ---
        "spo2": spo2 or {
            "mean": None, "min": None,
            "desaturation_count": 0, "desat_index_per_hr": None,
        },
        # --- Temperature features (TMP117) ---
        "temp_c": temp_c or {
            "mean": None, "variation": None, "fever": False,
        },
        # --- Movement features (ICM-42688) ---
        "motion": motion or {
            "mvt_count": 0, "immobility_sec": 0.0,
            "agitation_index": 0.0, "score": 0.0,
            "notes": ["no_data"],
        },
        # --- Respiration features (derived) ---
        "respiration": respiration or {
            "rate_bpm": None, "amplitude": None,
        },
        # --- Clinical rules result ---
        "rules": rules or {"triggered": [], "severity": Severity.LOW},
        # --- ML inference result ---
        "ml": ml or {
            "risk_score": None,
            "deterioration_prob": None,
            "event_class": "normal",
        },
        # --- Final decision ---
        "decision": decision or {
            "severity": Severity.LOW,
            "color": "green",
            "action": "surveillance_standard",
        },
    }
