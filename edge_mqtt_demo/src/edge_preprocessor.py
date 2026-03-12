from __future__ import annotations

"""
Edge AI Gateway — Smart Medical Mat
======================================
Full edge processing pipeline per PDF architecture §1.2:
  a) Ingestion & synchronization
  b) Signal preprocessing
  c) Feature extraction
  d) Clinical rules engine
  e) ML inference (XGBoost / Random Forest / Logistic Regression)
  f) Decision engine & alert generation

Input topics  (ESP32/simulator): sim/<device>/ecg|ppg|imu|spo2|temp
Output topics (edge gateway):    edge/<device>/features|events

Simulation mode: runs on laptop. Same code deploys to Raspberry Pi.
"""

import argparse
import csv
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import paho.mqtt.client as mqtt

from common import (
    DEFAULT_DEVICE_ID,
    MqttConfig,
    SensorConfig,
    chunk_duration_ms,
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
from feature_extraction import (
    extract_ecg_features,
    extract_motion_features,
    extract_ppg_features,
    extract_respiration_features,
    extract_spo2_features,
    extract_temp_features,
)
from clinical_rules import evaluate_clinical_rules
from decision_engine import make_decision
from ml_inference import MLInferenceEngine

try:
    from scipy.signal import find_peaks  # noqa: F401 — availability check

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


# Signal processing and feature extraction are now in feature_extraction.py
# Clinical rules are now in clinical_rules.py
# ML inference is now in ml_inference.py
# Decision fusion is now in decision_engine.py


# =============================================================================
# CSV FEATURE LOGGER
# =============================================================================

CSV_COLUMNS = [
    "timestamp", "patient_id", "window_start_ms", "window_sec",
    # ECG
    "ecg_hr_mean", "ecg_hr_min", "ecg_hr_max",
    "ecg_hrv_sdnn", "ecg_hrv_rmssd", "ecg_qrs_duration_ms",
    "ecg_abnormal_beats", "ecg_sqi",
    # PPG
    "ppg_pulse_rate", "ppg_amplitude", "ppg_ptt_ms", "ppg_sqi",
    # SpO2
    "spo2_mean", "spo2_min", "spo2_desaturation_count",
    # Temperature
    "temp_mean", "temp_variation", "temp_fever",
    # Motion
    "motion_mvt_count", "motion_immobility", "motion_agitation_index", "motion_score",
    # Respiration
    "resp_rate_bpm", "resp_amplitude",
    # Rules
    "rules_triggered", "rules_severity",
    # ML
    "ml_risk_score", "ml_event_class", "ml_deterioration_prob",
    # Decision
    "decision_severity", "decision_color", "decision_action",
]


class CsvFeatureLogger:
    """Logs extracted features to a CSV file for each window."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"features_{ts}.csv")
        self._file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        print(f"[edge] CSV logging to: {self.filepath}")

    def log(self, features: Dict[str, Any]) -> None:
        ecg = features.get("ecg", {})
        ppg = features.get("ppg", {})
        spo2 = features.get("spo2", {})
        temp = features.get("temp_c", {})
        motion = features.get("motion", {})
        resp = features.get("respiration", {})
        rules = features.get("rules", {})
        ml = features.get("ml", {})
        dec = features.get("decision", {})

        row = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": features.get("patient_id", ""),
            "window_start_ms": features.get("window_start_ms", ""),
            "window_sec": features.get("window_sec", ""),
            # ECG
            "ecg_hr_mean": ecg.get("hr_mean", ""),
            "ecg_hr_min": ecg.get("hr_min", ""),
            "ecg_hr_max": ecg.get("hr_max", ""),
            "ecg_hrv_sdnn": ecg.get("hrv_sdnn", ""),
            "ecg_hrv_rmssd": ecg.get("hrv_rmssd", ""),
            "ecg_qrs_duration_ms": ecg.get("qrs_duration_ms", ""),
            "ecg_abnormal_beats": ecg.get("abnormal_beats", ""),
            "ecg_sqi": ecg.get("sqi", ""),
            # PPG
            "ppg_pulse_rate": ppg.get("pulse_rate", ""),
            "ppg_amplitude": ppg.get("amplitude", ""),
            "ppg_ptt_ms": ppg.get("ptt_ms", ""),
            "ppg_sqi": ppg.get("sqi", ""),
            # SpO2
            "spo2_mean": spo2.get("mean", ""),
            "spo2_min": spo2.get("min", ""),
            "spo2_desaturation_count": spo2.get("desaturation_count", ""),
            # Temp
            "temp_mean": temp.get("mean", ""),
            "temp_variation": temp.get("variation", ""),
            "temp_fever": temp.get("fever", ""),
            # Motion
            "motion_mvt_count": motion.get("mvt_count", ""),
            "motion_immobility": motion.get("immobility", ""),
            "motion_agitation_index": motion.get("agitation_index", ""),
            "motion_score": motion.get("score", ""),
            # Respiration
            "resp_rate_bpm": resp.get("rate_bpm", ""),
            "resp_amplitude": resp.get("amplitude", ""),
            # Rules
            "rules_triggered": ";".join(rules.get("triggered", [])),
            "rules_severity": rules.get("severity", ""),
            # ML
            "ml_risk_score": ml.get("risk_score", ""),
            "ml_event_class": ml.get("event_class", ""),
            "ml_deterioration_prob": ml.get("deterioration_prob", ""),
            # Decision
            "decision_severity": dec.get("severity", ""),
            "decision_color": dec.get("color", ""),
            "decision_action": dec.get("action", ""),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


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
    """Multi-sensor Edge AI Gateway."""

    def __init__(
        self,
        mqtt_client: mqtt.Client,
        device_id: str,
        config: SensorConfig,
        models_dir: str,
        debug: bool = False,
    ):
        self.client = mqtt_client
        self.device_id = device_id
        self.config = config
        self.debug = debug

        self.lock = threading.Lock()
        self.patients: Dict[str, PatientState] = {}

        self.input_topics = get_input_topics(device_id)
        self.output_topics = get_output_topics(device_id)

        # Initialize ML inference engine
        self.ml_engine = MLInferenceEngine(models_dir)

        # CSV logger (set externally via set_csv_logger)
        self.csv_logger: Optional[CsvFeatureLogger] = None

    def set_csv_logger(self, logger: CsvFeatureLogger) -> None:
        self.csv_logger = logger

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
        """Full pipeline: extract → rules → ML → decision → publish."""
        window_sec = self.config.window_sec
        ecg_window_samples = int(window_sec * state.ecg_buffer.fs_hz)

        if not state.ecg_buffer.has_window(ecg_window_samples):
            return

        # --- a) Extract raw windows aligned to ECG timing ---
        ecg_raw, window_start_ms = state.ecg_buffer.extract_window(ecg_window_samples)
        window_end_ms = window_start_ms + int(window_sec * 1000)

        ppg_window_samples = int(window_sec * state.ppg_buffer.fs_hz)
        ppg_raw, _ = state.ppg_buffer.extract_window(ppg_window_samples)

        imu_window_samples = int(window_sec * state.imu_buffer.fs_hz)
        imu_raw, _ = state.imu_buffer.extract_window(imu_window_samples)

        spo2_values = state.spo2_buffer.extract_window(window_start_ms, window_end_ms)
        temp_values = state.temp_buffer.extract_window(window_start_ms, window_end_ms)

        # --- b+c) Feature extraction (all sensors) ---
        motion_features = extract_motion_features(imu_raw, state.imu_buffer.fs_hz)
        motion_score = motion_features["score"]

        ecg_features = extract_ecg_features(
            ecg_raw, state.ecg_buffer.fs_hz,
            notch_hz=self.config.notch_hz,
            motion_score=motion_score,
        )

        # Get internal ECG peaks for PPG PTT calculation
        ecg_peaks = ecg_features.pop("_peaks", np.array([], dtype=int))
        ecg_filtered = ecg_features.pop("_filtered", np.array([], dtype=float))

        ppg_features = extract_ppg_features(
            ppg_raw, state.ppg_buffer.fs_hz,
            motion_score=motion_score,
            ecg_peaks=ecg_peaks,
            ecg_fs=state.ecg_buffer.fs_hz,
        )

        spo2_features, spo2_raw_events = extract_spo2_features(
            spo2_values, window_sec, state.spo2_drop_state,
        )

        temp_features = extract_temp_features(temp_values)

        resp_features = extract_respiration_features(ecg_filtered, state.ecg_buffer.fs_hz)

        # Update last HR values
        if ecg_features.get("hr_mean") is not None:
            state.last_hr_ecg = ecg_features["hr_mean"]
        if ppg_features.get("pulse_rate") is not None:
            state.last_hr_ppg = ppg_features["pulse_rate"]

        # --- d) Clinical rules engine ---
        rules_result = evaluate_clinical_rules(
            patient_id=state.patient_id,
            t_ms=window_start_ms,
            ecg_features=ecg_features,
            ppg_features=ppg_features,
            spo2_features=spo2_features,
            temp_features=temp_features,
            motion_features=motion_features,
            respiration_features=resp_features,
        )

        # --- e) ML inference ---
        ml_result = self.ml_engine.predict(
            ecg=ecg_features,
            ppg=ppg_features,
            spo2=spo2_features,
            temp=temp_features,
            motion=motion_features,
            respiration=resp_features,
        )

        # --- f) Decision engine ---
        decision = make_decision(rules_result, ml_result)

        # --- Build and publish features ---
        features = create_features(
            patient_id=state.patient_id,
            window_start_ms=window_start_ms,
            window_sec=window_sec,
            ecg=ecg_features,
            ppg=ppg_features,
            spo2=spo2_features,
            temp_c=temp_features,
            motion=motion_features,
            respiration=resp_features,
            rules={"triggered": rules_result["triggered"], "severity": rules_result["severity"]},
            ml=ml_result,
            decision=decision,
        )

        self.client.publish(self.output_topics["features"], to_json(features), qos=1)

        # Log to CSV file
        if self.csv_logger:
            self.csv_logger.log(features)

        print(f"[edge] patient={state.patient_id} window={window_start_ms} "
              f"HR={ecg_features.get('hr_mean')} SpO2={spo2_features.get('mean')} "
              f"Temp={temp_features.get('mean')} Motion={motion_score:.2f} "
              f"ML_risk={ml_result.get('risk_score')} "
              f"Decision={decision['color'].upper()} ({decision['action']})")

        # --- Publish events (from rules engine) ---
        for evt in rules_result.get("events", []):
            self.client.publish(self.output_topics["events"], to_json(evt), qos=1)
            print(f"[edge] EVENT: {evt['type']} severity={evt['severity']} details={evt.get('details', {})}")


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
    parser.add_argument("--models-dir", default=None,
                        help="Directory for ML model files (default: <project>/models)")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for CSV feature logs (default: <project>/data/logs)")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable CSV file logging")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Resolve models directory
    if args.models_dir:
        models_dir = args.models_dir
    else:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    # Print capabilities
    print(f"[edge] SciPy available: {SCIPY_AVAILABLE}")
    if not SCIPY_AVAILABLE:
        print("[edge] WARNING: Running without SciPy - using basic preprocessing only")
    print(f"[edge] Models directory: {models_dir}")

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
        models_dir=models_dir,
        debug=args.debug,
    )

    # Set up CSV feature logger
    csv_logger = None
    if not args.no_log:
        log_dir = args.log_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs"
        )
        csv_logger = CsvFeatureLogger(log_dir)
        processor.set_csv_logger(csv_logger)

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
        if csv_logger:
            csv_logger.close()
            print(f"[edge] Features saved to: {csv_logger.filepath}")


if __name__ == "__main__":
    main()
