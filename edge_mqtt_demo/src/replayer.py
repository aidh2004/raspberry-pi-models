from __future__ import annotations

"""
ESP32 Simulator / Multi-Sensor Replayer
========================================
Publishes synthetic sensor data to MQTT topics, simulating the ESP32 hardware.

Supported sensors:
- ECG (250 Hz, chunk-based)
- PPG (100 Hz, chunk-based)
- IMU (50 Hz, chunk-based with [ax, ay, az] triplets)
- SpO2 (1 Hz, sample-based)
- Temperature (0.2 Hz, sample-based)

Topics (device_id = patient1 by default):
- sim/patient1/ecg
- sim/patient1/ppg
- sim/patient1/imu
- sim/patient1/spo2
- sim/patient1/temp

Future hardware: Replace this script with real ESP32 publishing to mat/<device_id>/...
"""

import argparse
import csv
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import paho.mqtt.client as mqtt

from common import (
    DEFAULT_DEVICE_ID,
    MqttConfig,
    SensorConfig,
    chunk_duration_ms,
    get_input_topics,
    now_ms,
    to_json,
)


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_synthetic_ecg(
    duration_sec: float,
    fs_hz: int,
    hr_bpm: float,
    noise_std: float = 0.03
) -> np.ndarray:
    """
    Generate synthetic ECG with P-QRS-T morphology, baseline wander,
    50Hz powerline interference, and Gaussian noise.
    """
    n = int(duration_sec * fs_hz)
    t_full = np.arange(n) / fs_hz

    rr_sec = 60.0 / hr_bpm
    beat_samples = int(round(rr_sec * fs_hz))
    if beat_samples < 10:
        beat_samples = 10
    t_beat = np.arange(beat_samples) / fs_hz
    center = rr_sec / 2.0

    beat = np.zeros(beat_samples)
    beat += 1.2 * np.exp(-0.5 * ((t_beat - center) / 0.015) ** 2)            # R
    beat += -0.15 * np.exp(-0.5 * ((t_beat - (center - 0.04)) / 0.01) ** 2)  # Q
    beat += -0.25 * np.exp(-0.5 * ((t_beat - (center + 0.02)) / 0.012) ** 2) # S
    beat += 0.18 * np.exp(-0.5 * ((t_beat - (center - 0.18)) / 0.03) ** 2)   # P
    beat += 0.28 * np.exp(-0.5 * ((t_beat - (center + 0.22)) / 0.05) ** 2)   # T

    num_beats = int(np.ceil(n / beat_samples))
    signal = np.tile(beat, num_beats)[:n]

    baseline = 0.1 * np.sin(2 * np.pi * 0.2 * t_full)
    powerline = 0.02 * np.sin(2 * np.pi * 50.0 * t_full)
    noise = np.random.normal(0.0, noise_std, size=n)

    return signal + baseline + powerline + noise


def generate_synthetic_ppg(
    duration_sec: float,
    fs_hz: int,
    hr_bpm: float,
    noise_std: float = 0.02
) -> np.ndarray:
    """
    Generate synthetic PPG (photoplethysmogram) signal.
    PPG has a characteristic sawtooth-like waveform with dicrotic notch.
    """
    n = int(duration_sec * fs_hz)
    t_full = np.arange(n) / fs_hz

    rr_sec = 60.0 / hr_bpm
    beat_samples = int(round(rr_sec * fs_hz))
    if beat_samples < 10:
        beat_samples = 10
    t_beat = np.arange(beat_samples) / fs_hz

    # PPG waveform: systolic peak + dicrotic notch + diastolic component
    systolic_peak = 0.7
    diastolic_peak = 0.3
    peak_time = rr_sec * 0.15
    notch_time = rr_sec * 0.35
    diastolic_time = rr_sec * 0.45

    beat = np.zeros(beat_samples)
    # Systolic rise and peak
    beat += systolic_peak * np.exp(-0.5 * ((t_beat - peak_time) / 0.04) ** 2)
    # Dicrotic notch (small dip)
    beat += -0.1 * np.exp(-0.5 * ((t_beat - notch_time) / 0.02) ** 2)
    # Diastolic component
    beat += diastolic_peak * np.exp(-0.5 * ((t_beat - diastolic_time) / 0.06) ** 2)

    num_beats = int(np.ceil(n / beat_samples))
    signal = np.tile(beat, num_beats)[:n]

    # Add baseline and noise
    baseline = 0.5 + 0.05 * np.sin(2 * np.pi * 0.1 * t_full)
    noise = np.random.normal(0.0, noise_std, size=n)

    return signal + baseline + noise


def generate_synthetic_imu(
    duration_sec: float,
    fs_hz: int,
    motion_events: Optional[List[Tuple[float, float, float]]] = None,
    base_noise_std: float = 0.01
) -> np.ndarray:
    """
    Generate synthetic IMU data (accelerometer).
    Returns array of shape (n, 3) with [ax, ay, az] triplets.

    motion_events: List of (start_time, duration, intensity) for motion bursts.
    """
    n = int(duration_sec * fs_hz)
    t = np.arange(n) / fs_hz

    # Baseline: gravity on Z-axis (assuming device lying flat) + noise
    ax = np.random.normal(0.0, base_noise_std, n)
    ay = np.random.normal(0.0, base_noise_std, n)
    az = np.ones(n) * 1.0 + np.random.normal(0.0, base_noise_std, n)  # ~1g on Z

    # Add motion events
    if motion_events is None:
        # Default: random motion bursts
        num_events = max(1, int(duration_sec / 30))  # ~1 event per 30s
        motion_events = []
        for _ in range(num_events):
            start = random.uniform(10, max(11, duration_sec - 10))
            dur = random.uniform(1.0, 3.0)
            intensity = random.uniform(0.3, 1.0)
            motion_events.append((start, dur, intensity))

    for start_t, dur, intensity in motion_events:
        start_idx = int(start_t * fs_hz)
        end_idx = int((start_t + dur) * fs_hz)
        end_idx = min(end_idx, n)
        if start_idx < n:
            burst_len = end_idx - start_idx
            ax[start_idx:end_idx] += intensity * np.random.normal(0, 0.5, burst_len)
            ay[start_idx:end_idx] += intensity * np.random.normal(0, 0.5, burst_len)
            az[start_idx:end_idx] += intensity * np.random.normal(0, 0.3, burst_len)

    return np.column_stack([ax, ay, az])


def generate_synthetic_spo2(
    duration_sec: float,
    sample_interval_sec: float = 1.0,
    baseline: float = 98.0,
    noise_std: float = 0.5,
    drop_events: Optional[List[Tuple[float, float, float]]] = None
) -> List[Tuple[float, float]]:
    """
    Generate synthetic SpO2 readings.
    Returns list of (time_sec, spo2_value) tuples.

    drop_events: List of (start_time, duration, drop_to_value) for desaturation events.
    """
    samples: List[Tuple[float, float]] = []
    t = 0.0

    while t < duration_sec:
        value = baseline + np.random.normal(0, noise_std)

        # Check for drop events
        if drop_events:
            for start_t, dur, drop_val in drop_events:
                if start_t <= t < start_t + dur:
                    # Gradual drop and recovery
                    progress = (t - start_t) / dur
                    if progress < 0.5:
                        # Dropping phase
                        value = baseline - (baseline - drop_val) * (progress * 2)
                    else:
                        # Recovery phase
                        value = drop_val + (baseline - drop_val) * ((progress - 0.5) * 2)
                    break

        # Clamp to realistic range
        value = max(70.0, min(100.0, value))
        samples.append((t, value))
        t += sample_interval_sec

    return samples


def generate_synthetic_temp(
    duration_sec: float,
    sample_interval_sec: float = 5.0,
    baseline: float = 37.0,
    drift_rate: float = 0.01,
    noise_std: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Generate synthetic temperature readings.
    Returns list of (time_sec, temp_celsius) tuples.
    """
    samples: List[Tuple[float, float]] = []
    t = 0.0
    current_temp = baseline

    while t < duration_sec:
        # Slow drift (random walk)
        current_temp += drift_rate * np.random.normal(0, 1)
        value = current_temp + np.random.normal(0, noise_std)

        # Clamp to realistic range
        value = max(34.0, min(42.0, value))
        samples.append((t, value))
        t += sample_interval_sec

    return samples


# =============================================================================
# FILE LOADERS
# =============================================================================

def load_ecg_file(path: Path) -> np.ndarray:
    """Load ECG data from CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {path}")

    values: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            for token in row:
                token = token.strip()
                if not token:
                    continue
                try:
                    values.append(float(token))
                except ValueError:
                    continue

    if not values:
        raise ValueError("No numeric ECG values in file")

    return np.asarray(values, dtype=float)


# =============================================================================
# SENSOR STREAM CLASSES
# =============================================================================

@dataclass
class ChunkStreamConfig:
    """Configuration for high-rate chunk-based streams."""
    topic: str
    fs_hz: int
    chunk_ms: int
    data: np.ndarray
    is_imu: bool = False


@dataclass
class SampleStreamConfig:
    """Configuration for low-rate sample-based streams."""
    topic: str
    samples: List[Tuple[float, float]]  # (time_sec, value)


class MultiSensorReplayer:
    """
    Replays multiple sensor streams to MQTT with synchronized timestamps.
    """

    def __init__(
        self,
        mqtt_client: mqtt.Client,
        patient_id: str,
        stream_t0_ms: int,
        speed: float = 1.0,
        verbose: bool = True
    ):
        self.client = mqtt_client
        self.patient_id = patient_id
        self.stream_t0_ms = stream_t0_ms
        self.speed = max(0.1, speed)
        self.verbose = verbose
        self.running = False

        self.chunk_streams: List[ChunkStreamConfig] = []
        self.sample_streams: List[SampleStreamConfig] = []

    def add_chunk_stream(self, config: ChunkStreamConfig) -> None:
        """Add a chunk-based stream (ECG, PPG, IMU)."""
        self.chunk_streams.append(config)

    def add_sample_stream(self, config: SampleStreamConfig) -> None:
        """Add a sample-based stream (SpO2, Temperature)."""
        self.sample_streams.append(config)

    def _publish_chunk(
        self,
        topic: str,
        t0_ms: int,
        fs_hz: int,
        samples: np.ndarray,
        is_imu: bool = False
    ) -> None:
        """Publish a chunk message to MQTT."""
        if is_imu:
            # IMU samples are triplets
            samples_list = samples.tolist()
        else:
            samples_list = samples.astype(float).tolist()

        payload = {
            "patient_id": self.patient_id,
            "t0_ms": int(t0_ms),
            "fs_hz": int(fs_hz),
            "samples": samples_list,
        }
        self.client.publish(topic, to_json(payload), qos=1)

    def _publish_sample(self, topic: str, t_ms: int, value: float) -> None:
        """Publish a sample message to MQTT."""
        payload = {
            "patient_id": self.patient_id,
            "t_ms": int(t_ms),
            "value": float(value),
        }
        self.client.publish(topic, to_json(payload), qos=1)

    def _run_chunk_stream(self, config: ChunkStreamConfig) -> None:
        """Thread worker for a chunk-based stream."""
        samples_per_chunk = max(1, int(round(config.fs_hz * (config.chunk_ms / 1000.0))))
        index = 0
        chunk_idx = 0

        while self.running and index < len(config.data):
            end_idx = min(index + samples_per_chunk, len(config.data))
            chunk = config.data[index:end_idx]
            if len(chunk) == 0:
                break

            t0_ms = self.stream_t0_ms + chunk_duration_ms(index, config.fs_hz)
            self._publish_chunk(config.topic, t0_ms, config.fs_hz, chunk, config.is_imu)

            if self.verbose and chunk_idx % 20 == 0:
                sensor_name = config.topic.split("/")[-1].upper()
                print(f"[replayer] {sensor_name}: chunk={chunk_idx} samples={len(chunk)} t0={t0_ms}")

            elapsed = (len(chunk) / config.fs_hz) / self.speed
            time.sleep(elapsed)

            index = end_idx
            chunk_idx += 1

        if self.verbose:
            sensor_name = config.topic.split("/")[-1].upper()
            print(f"[replayer] {sensor_name}: stream completed ({chunk_idx} chunks)")

    def _run_sample_stream(self, config: SampleStreamConfig) -> None:
        """Thread worker for a sample-based stream."""
        sensor_name = config.topic.split("/")[-1].upper()
        sample_idx = 0

        for time_sec, value in config.samples:
            if not self.running:
                break

            t_ms = self.stream_t0_ms + int(time_sec * 1000)
            self._publish_sample(config.topic, t_ms, value)

            if self.verbose and sample_idx % 10 == 0:
                print(f"[replayer] {sensor_name}: sample={sample_idx} value={value:.2f} t={t_ms}")

            sample_idx += 1

            # Calculate sleep until next sample
            if sample_idx < len(config.samples):
                next_time_sec = config.samples[sample_idx][0]
                sleep_sec = (next_time_sec - time_sec) / self.speed
                if sleep_sec > 0:
                    # Sleep in small increments to check running flag
                    end_time = time.time() + sleep_sec
                    while self.running and time.time() < end_time:
                        time.sleep(min(0.1, end_time - time.time()))

        if self.verbose:
            print(f"[replayer] {sensor_name}: stream completed ({sample_idx} samples)")

    def run(self, block: bool = True) -> List[threading.Thread]:
        """
        Start all sensor streams.
        If block=True, waits for all streams to complete.
        Returns list of threads.
        """
        self.running = True
        threads: List[threading.Thread] = []

        # Start chunk streams
        for config in self.chunk_streams:
            t = threading.Thread(target=self._run_chunk_stream, args=(config,), daemon=True)
            t.start()
            threads.append(t)

        # Start sample streams
        for config in self.sample_streams:
            t = threading.Thread(target=self._run_sample_stream, args=(config,), daemon=True)
            t.start()
            threads.append(t)

        if block:
            try:
                for t in threads:
                    while t.is_alive():
                        t.join(timeout=0.5)
            except KeyboardInterrupt:
                self.running = False
                raise

        return threads

    def stop(self) -> None:
        """Stop all streams."""
        self.running = False


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESP32 simulator: publish multi-sensor data to MQTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All sensors, synthetic data, 5 minutes:
  python replayer.py --mode synthetic --duration-sec 300

  # ECG only (legacy mode):
  python replayer.py --mode synthetic --ecg-only

  # Custom HR:
  python replayer.py --mode synthetic --hr-bpm 80 --duration-sec 600

  # With SpO2 drop event:
  python replayer.py --mode synthetic --inject-spo2-drop
        """
    )

    # Connection settings
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--patient-id", default=DEFAULT_DEVICE_ID)

    # Mode settings
    parser.add_argument("--mode", choices=["synthetic", "file"], default="synthetic")
    parser.add_argument("--file", type=str, default="", help="ECG file path (file mode only)")
    parser.add_argument("--loop", action="store_true", help="Loop data forever")

    # Timing settings
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = real-time")
    parser.add_argument("--chunk-ms", type=int, default=250, help="Chunk duration in ms")

    # Sensor enable/disable
    parser.add_argument("--ecg-only", action="store_true", help="ECG only (legacy mode)")
    parser.add_argument("--no-ecg", action="store_true")
    parser.add_argument("--no-ppg", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--no-spo2", action="store_true")
    parser.add_argument("--no-temp", action="store_true")

    # ECG settings
    parser.add_argument("--ecg-fs", type=int, default=250)
    parser.add_argument("--hr-bpm", type=float, default=72.0)
    parser.add_argument("--noise-std", type=float, default=0.03)

    # PPG settings
    parser.add_argument("--ppg-fs", type=int, default=100)

    # IMU settings
    parser.add_argument("--imu-fs", type=int, default=50)

    # SpO2 settings
    parser.add_argument("--spo2-interval", type=float, default=1.0)
    parser.add_argument("--spo2-baseline", type=float, default=98.0)
    parser.add_argument("--inject-spo2-drop", action="store_true",
                        help="Inject SpO2 drop event mid-stream")

    # Temperature settings
    parser.add_argument("--temp-interval", type=float, default=5.0)
    parser.add_argument("--temp-baseline", type=float, default=37.0)

    # Debug settings
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.chunk_ms < 50:
        raise ValueError("chunk-ms too low (minimum 50ms)")

    # Get topics for this patient
    topics = get_input_topics(args.patient_id)

    # Generate synthetic data
    print(f"[replayer] generating synthetic data for {args.duration_sec}s...")

    # ECG
    ecg_data = None
    if not args.no_ecg:
        if args.mode == "file" and args.file:
            ecg_data = load_ecg_file(Path(args.file))
            print(f"[replayer] loaded ECG file: {len(ecg_data)} samples")
        else:
            ecg_data = generate_synthetic_ecg(
                args.duration_sec, args.ecg_fs, args.hr_bpm, args.noise_std
            )
            print(f"[replayer] generated synthetic ECG: {len(ecg_data)} samples, {args.hr_bpm} bpm")

    # PPG (correlated with ECG HR)
    ppg_data = None
    if not args.no_ppg and not args.ecg_only:
        ppg_data = generate_synthetic_ppg(
            args.duration_sec, args.ppg_fs, args.hr_bpm, noise_std=0.02
        )
        print(f"[replayer] generated synthetic PPG: {len(ppg_data)} samples")

    # IMU
    imu_data = None
    if not args.no_imu and not args.ecg_only:
        imu_data = generate_synthetic_imu(args.duration_sec, args.imu_fs)
        print(f"[replayer] generated synthetic IMU: {len(imu_data)} samples")

    # SpO2
    spo2_data = None
    if not args.no_spo2 and not args.ecg_only:
        drop_events = None
        if args.inject_spo2_drop:
            # Inject a 15-second drop to 88% starting at 1/3 of duration
            drop_start = args.duration_sec / 3
            drop_events = [(drop_start, 15.0, 88.0)]
            print(f"[replayer] injecting SpO2 drop at t={drop_start:.0f}s")
        spo2_data = generate_synthetic_spo2(
            args.duration_sec, args.spo2_interval, args.spo2_baseline,
            noise_std=0.5, drop_events=drop_events
        )
        print(f"[replayer] generated synthetic SpO2: {len(spo2_data)} samples")

    # Temperature
    temp_data = None
    if not args.no_temp and not args.ecg_only:
        temp_data = generate_synthetic_temp(
            args.duration_sec, args.temp_interval, args.temp_baseline
        )
        print(f"[replayer] generated synthetic Temp: {len(temp_data)} samples")

    # Connect to MQTT
    mqtt_cfg = MqttConfig(host=args.broker_host, port=args.broker_port)
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"sensor-replayer-{random.randint(1000, 9999)}",
    )

    connected_event = threading.Event()

    def on_connect(_c: mqtt.Client, _userdata, _connect_flags, reason_code, _properties):
        if reason_code == 0:
            print(f"[replayer] connected to MQTT {mqtt_cfg.host}:{mqtt_cfg.port}")
            connected_event.set()
        else:
            print(f"[replayer] MQTT connection failed reason={reason_code}")

    def on_disconnect(_c: mqtt.Client, _userdata, _disconnect_flags, reason_code, _properties):
        print(f"[replayer] disconnected reason={reason_code}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    print(f"[replayer] connecting to {mqtt_cfg.host}:{mqtt_cfg.port}...")
    while True:
        try:
            client.connect(mqtt_cfg.host, mqtt_cfg.port, mqtt_cfg.keepalive)
            break
        except OSError as ex:
            print(f"[replayer] broker not reachable ({ex}); retrying in 2s...")
            time.sleep(2)

    client.loop_start()
    connected_event.wait(timeout=10)

    # Create replayer
    replayer = MultiSensorReplayer(
        mqtt_client=client,
        patient_id=args.patient_id,
        stream_t0_ms=now_ms(),
        speed=args.speed,
        verbose=not args.quiet
    )

    # Add streams
    if ecg_data is not None:
        replayer.add_chunk_stream(ChunkStreamConfig(
            topic=topics["ecg"],
            fs_hz=args.ecg_fs,
            chunk_ms=args.chunk_ms,
            data=ecg_data,
            is_imu=False
        ))

    if ppg_data is not None:
        replayer.add_chunk_stream(ChunkStreamConfig(
            topic=topics["ppg"],
            fs_hz=args.ppg_fs,
            chunk_ms=args.chunk_ms,
            data=ppg_data,
            is_imu=False
        ))

    if imu_data is not None:
        replayer.add_chunk_stream(ChunkStreamConfig(
            topic=topics["imu"],
            fs_hz=args.imu_fs,
            chunk_ms=args.chunk_ms,
            data=imu_data,
            is_imu=True
        ))

    if spo2_data is not None:
        replayer.add_sample_stream(SampleStreamConfig(
            topic=topics["spo2"],
            samples=spo2_data
        ))

    if temp_data is not None:
        replayer.add_sample_stream(SampleStreamConfig(
            topic=topics["temp"],
            samples=temp_data
        ))

    # Print summary
    print("\n[replayer] starting streams:")
    for cs in replayer.chunk_streams:
        sensor = cs.topic.split("/")[-1]
        print(f"  - {sensor}: {cs.fs_hz}Hz, {len(cs.data)} samples -> {cs.topic}")
    for ss in replayer.sample_streams:
        sensor = ss.topic.split("/")[-1]
        print(f"  - {sensor}: {len(ss.samples)} samples -> {ss.topic}")
    print()

    # Run
    try:
        if args.loop:
            while True:
                replayer.stream_t0_ms = now_ms()
                replayer.run(block=True)
                print("[replayer] loop: restarting streams...")
        else:
            replayer.run(block=True)
            print("[replayer] all streams completed")
    except KeyboardInterrupt:
        print("\n[replayer] stopping (Ctrl+C)")
        replayer.stop()
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
