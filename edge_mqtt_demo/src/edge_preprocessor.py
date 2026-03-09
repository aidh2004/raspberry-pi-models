from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import paho.mqtt.client as mqtt

from common import (
    INPUT_TOPIC,
    OUTPUT_TOPIC,
    MqttConfig,
    chunk_duration_ms,
    parse_json,
    safe_float_list,
    to_json,
    validate_input_message,
)

try:
    from scipy.signal import butter, detrend, filtfilt, find_peaks, iirnotch

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class StreamState:
    fs_hz: int
    buffer_start_ms: int
    buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    @property
    def buffer_end_ms(self) -> int:
        return self.buffer_start_ms + chunk_duration_ms(len(self.buffer), self.fs_hz)


def preprocess_ecg(window: np.ndarray, fs_hz: int, notch_hz: Optional[float]) -> np.ndarray:
    x = window.astype(float)

    if SCIPY_AVAILABLE:
        x = detrend(x, type="linear")

        nyq = fs_hz / 2.0
        low = 0.5 / nyq
        high = 40.0 / nyq
        b, a = butter(3, [low, min(high, 0.99)], btype="band")
        x = filtfilt(b, a, x)

        if notch_hz is not None and notch_hz > 0 and notch_hz < nyq:
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
    min_distance = int(0.25 * fs_hz)

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


def estimate_hr_bpm(r_peaks: np.ndarray, fs_hz: int) -> Optional[float]:
    if len(r_peaks) < 2:
        return None

    rr_sec = np.diff(r_peaks) / float(fs_hz)
    rr_sec = rr_sec[(rr_sec > 0.3) & (rr_sec < 2.0)]
    if rr_sec.size == 0:
        return None

    return float(60.0 / np.mean(rr_sec))


def compute_sqi(raw: np.ndarray, filtered: np.ndarray, r_peaks: np.ndarray, fs_hz: int) -> float:
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

    peak_count = len(r_peaks)
    expected_min = int(40 / 60 * 5)  # 5s window
    expected_max = int(180 / 60 * 5)
    if peak_count < expected_min or peak_count > expected_max:
        score -= 0.2

    return float(np.clip(score, 0.0, 1.0))


def process_window(window: np.ndarray, fs_hz: int, notch_hz: Optional[float]) -> Tuple[Optional[float], float, List[str]]:
    notes: List[str] = []

    if len(window) < fs_hz:
        return None, 0.0, ["insufficient_data"]

    filtered = preprocess_ecg(window, fs_hz, notch_hz)
    r_peaks = detect_r_peaks(filtered, fs_hz)
    hr_bpm = estimate_hr_bpm(r_peaks, fs_hz)

    if hr_bpm is None:
        notes.append("insufficient_data")
    else:
        if hr_bpm < 40 or hr_bpm > 180:
            notes.append("hr_out_of_range")

    sqi = compute_sqi(window, filtered, r_peaks, fs_hz)
    if sqi < 0.5:
        notes.append("low_sqi")

    return hr_bpm, sqi, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Raspberry simulator: preprocess ECG + publish features")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--input-topic", default=INPUT_TOPIC)
    parser.add_argument("--output-topic", default=OUTPUT_TOPIC)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--disable-notch", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    stream_states: Dict[str, StreamState] = {}
    lock = threading.Lock()

    mqtt_cfg = MqttConfig(host=args.broker_host, port=args.broker_port)
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="edge-preprocessor",
    )

    def on_connect(c: mqtt.Client, _userdata, _connect_flags, reason_code, _properties):
        if reason_code == 0:
            print(f"[edge] connected to MQTT {mqtt_cfg.host}:{mqtt_cfg.port}")
            c.subscribe(args.input_topic, qos=1)
            print(f"[edge] subscribed to {args.input_topic}")
        else:
            print(f"[edge] MQTT connection failed reason={reason_code}")

    def on_disconnect(_c: mqtt.Client, _userdata, _disconnect_flags, reason_code, _properties):
        print(f"[edge] disconnected reason={reason_code}")

    def on_message(c: mqtt.Client, _userdata, msg: mqtt.MQTTMessage):
        try:
            payload = parse_json(msg.payload)
        except Exception as ex:
            print(f"[edge] invalid JSON payload: {ex}")
            return

        err = validate_input_message(payload)
        if err:
            print(f"[edge] invalid input schema: {err}")
            return

        patient_id = payload["patient_id"]
        t0_ms = int(payload["t0_ms"])
        fs_hz = int(payload["fs_hz"])
        samples = np.asarray(safe_float_list(payload["samples"]), dtype=float)
        n = len(samples)
        notes_stream: List[str] = []

        with lock:
            state = stream_states.get(patient_id)
            if state is None:
                state = StreamState(fs_hz=fs_hz, buffer_start_ms=t0_ms, buffer=np.array([], dtype=float))
                stream_states[patient_id] = state

            if state.fs_hz != fs_hz:
                notes_stream.append("fs_changed_stream_reset")
                state.fs_hz = fs_hz
                state.buffer = np.array([], dtype=float)
                state.buffer_start_ms = t0_ms

            end_ms = state.buffer_end_ms

            if len(state.buffer) == 0:
                state.buffer_start_ms = t0_ms
                state.buffer = samples.copy()
            else:
                if t0_ms > end_ms:
                    gap_ms = t0_ms - end_ms
                    gap_samples = int(round(gap_ms * state.fs_hz / 1000.0))
                    if gap_samples > 0:
                        notes_stream.append("gap_filled_zeros")
                        state.buffer = np.concatenate([state.buffer, np.zeros(gap_samples, dtype=float)])
                    state.buffer = np.concatenate([state.buffer, samples])

                elif t0_ms < end_ms:
                    overlap_ms = end_ms - t0_ms
                    overlap_samples = int(round(overlap_ms * state.fs_hz / 1000.0))
                    if overlap_samples >= n:
                        notes_stream.append("old_or_duplicate_chunk_dropped")
                    else:
                        if overlap_samples > 0:
                            notes_stream.append("overlap_trimmed")
                        state.buffer = np.concatenate([state.buffer, samples[overlap_samples:]])
                else:
                    state.buffer = np.concatenate([state.buffer, samples])

            window_samples = int(round(args.window_sec * state.fs_hz))

            while len(state.buffer) >= window_samples:
                window = state.buffer[:window_samples].copy()
                hr_bpm, sqi, notes = process_window(
                    window,
                    state.fs_hz,
                    None if args.disable_notch else args.notch_hz,
                )

                output = {
                    "patient_id": patient_id,
                    "window_start_ms": int(state.buffer_start_ms),
                    "window_sec": int(args.window_sec),
                    "hr_bpm": None if hr_bpm is None else round(float(hr_bpm), 2),
                    "sqi": round(float(sqi), 3),
                    "notes": notes_stream + notes,
                }

                c.publish(args.output_topic, to_json(output), qos=1)
                print(
                    f"[edge] patient={patient_id} start={output['window_start_ms']} "
                    f"hr={output['hr_bpm']} sqi={output['sqi']} notes={output['notes']}"
                )

                state.buffer = state.buffer[window_samples:]
                state.buffer_start_ms += chunk_duration_ms(window_samples, state.fs_hz)
                notes_stream = []

        if args.debug:
            print(f"[edge][debug] in_chunk t0={t0_ms} n={n} fs={fs_hz}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    while True:
        try:
            client.connect(mqtt_cfg.host, mqtt_cfg.port, mqtt_cfg.keepalive)
            break
        except OSError as ex:
            print(f"[edge] broker not reachable at {mqtt_cfg.host}:{mqtt_cfg.port} ({ex}); retrying in 2s...")
            time.sleep(2)

    client.loop_start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[edge] stopping (Ctrl+C)")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
