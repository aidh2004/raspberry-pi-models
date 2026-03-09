from __future__ import annotations

import argparse
import threading
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
from matplotlib.animation import FuncAnimation

from common import INPUT_TOPIC, MqttConfig, parse_json, safe_float_list, validate_input_message

try:
    from scipy.signal import butter, detrend, filtfilt, find_peaks, iirnotch

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def preprocess_ecg(window: np.ndarray, fs_hz: int, notch_hz: float = None) -> np.ndarray:
    """Same preprocessing as edge_preprocessor.py"""
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
    """Same R-peak detection as edge_preprocessor.py"""
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
        (filtered[1:-1] > filtered[:-2]) & (filtered[1:-1] > filtered[2:]) & (filtered[1:-1] > threshold)
    )[0] + 1

    selected: List[int] = []
    for idx in candidates:
        if not selected or (idx - selected[-1]) >= min_distance:
            selected.append(int(idx))
    return np.asarray(selected, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw vs processed ECG in real-time")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--input-topic", default=INPUT_TOPIC)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--disable-notch", action="store_true")
    args = parser.parse_args()

    if not SCIPY_AVAILABLE:
        print("[viz] WARNING: SciPy not available, using basic preprocessing only")

    # Shared state
    lock = threading.Lock()
    raw_buffer: List[float] = []
    filtered_buffer: List[float] = []
    peaks_buffer: List[int] = []
    fs_hz = 250
    max_samples = int(args.window_sec * fs_hz)

    mqtt_cfg = MqttConfig(host=args.broker_host, port=args.broker_port)
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="ecg-visualizer")

    def on_connect(c: mqtt.Client, _userdata, _connect_flags, reason_code, _properties):
        if reason_code == 0:
            print(f"[viz] connected to MQTT {mqtt_cfg.host}:{mqtt_cfg.port}")
            c.subscribe(args.input_topic, qos=1)
            print(f"[viz] subscribed to {args.input_topic}")
        else:
            print(f"[viz] MQTT connection failed reason={reason_code}")

    def on_disconnect(_c: mqtt.Client, _userdata, _disconnect_flags, reason_code, _properties):
        print(f"[viz] disconnected reason={reason_code}")

    def on_message(_c: mqtt.Client, _userdata, msg: mqtt.MQTTMessage):
        nonlocal fs_hz
        try:
            payload = parse_json(msg.payload)
        except Exception as ex:
            print(f"[viz] invalid JSON payload: {ex}")
            return

        err = validate_input_message(payload)
        if err:
            print(f"[viz] invalid input schema: {err}")
            return

        fs_hz = int(payload["fs_hz"])
        samples = safe_float_list(payload["samples"])

        with lock:
            raw_buffer.extend(samples)

            # Keep only last window_sec worth of data
            if len(raw_buffer) > max_samples:
                raw_buffer[:] = raw_buffer[-max_samples:]

            # Process when we have enough data
            if len(raw_buffer) >= fs_hz:
                raw_array = np.array(raw_buffer, dtype=float)
                filtered_array = preprocess_ecg(
                    raw_array, fs_hz, None if args.disable_notch else args.notch_hz
                )
                peaks = detect_r_peaks(filtered_array, fs_hz)

                filtered_buffer[:] = filtered_array.tolist()
                peaks_buffer[:] = peaks.tolist()

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    print(f"[viz] connecting to broker at {mqtt_cfg.host}:{mqtt_cfg.port}...")
    while True:
        try:
            client.connect(mqtt_cfg.host, mqtt_cfg.port, mqtt_cfg.keepalive)
            break
        except OSError as ex:
            print(f"[viz] broker not reachable ({ex}); retrying in 2s...")
            time.sleep(2)

    client.loop_start()

    # Setup matplotlib figure
    print("[viz] setting up real-time plot...")
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("ECG Signal Processing Pipeline", fontsize=14, fontweight="bold")

    def update_plot(_frame):
        with lock:
            if len(raw_buffer) < fs_hz:
                return  # Not enough data yet

            raw = np.array(raw_buffer, dtype=float)
            filtered = np.array(filtered_buffer, dtype=float) if filtered_buffer else raw
            peaks = np.array(peaks_buffer, dtype=int) if peaks_buffer else np.array([])

            # Time axis (in seconds)
            t = np.arange(len(raw)) / fs_hz

            # Plot 1: Raw signal
            ax1.clear()
            ax1.plot(t, raw, "b-", linewidth=0.8, label="Raw ECG")
            ax1.set_ylabel("Amplitude (raw)", fontsize=10)
            ax1.set_title("1. Raw ECG Signal (with baseline drift + noise)", fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right")

            # Plot 2: Filtered signal
            ax2.clear()
            ax2.plot(t, filtered, "g-", linewidth=0.8, label="Filtered ECG")
            ax2.set_ylabel("Amplitude (filtered)", fontsize=10)
            ax2.set_title(
                "2. After Preprocessing (detrend + bandpass 0.5-40Hz + notch 50Hz)",
                fontsize=11,
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right")

            # Plot 3: Filtered + R-peaks
            ax3.clear()
            ax3.plot(t, filtered, "g-", linewidth=0.8, label="Filtered ECG")

            if len(peaks) > 0:
                valid_peaks = peaks[peaks < len(filtered)]
                ax3.plot(
                    t[valid_peaks],
                    filtered[valid_peaks],
                    "ro",
                    markersize=8,
                    label=f"R-peaks ({len(valid_peaks)} detected)",
                )

                # Calculate HR if enough peaks
                if len(valid_peaks) >= 2:
                    rr_intervals = np.diff(valid_peaks) / fs_hz
                    rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
                    if len(rr_intervals) > 0:
                        hr_bpm = 60.0 / np.mean(rr_intervals)
                        ax3.text(
                            0.02,
                            0.95,
                            f"HR: {hr_bpm:.1f} bpm",
                            transform=ax3.transAxes,
                            fontsize=12,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
                        )

            ax3.set_xlabel("Time (seconds)", fontsize=10)
            ax3.set_ylabel("Amplitude (filtered)", fontsize=10)
            ax3.set_title("3. Filtered ECG + Detected R-peaks", fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc="upper right")

            plt.tight_layout()

    print("[viz] starting live visualization...")
    print("[viz] waiting for ECG data from replayer...")

    # Animate
    ani = FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\n[viz] stopping (Ctrl+C)")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
