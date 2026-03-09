from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import paho.mqtt.client as mqtt

from common import INPUT_TOPIC, MqttConfig, chunk_duration_ms, now_ms, to_json


def generate_synthetic_ecg(duration_sec: float, fs_hz: int, hr_bpm: float, noise_std: float) -> np.ndarray:
    n = int(duration_sec * fs_hz)
    t_full = np.arange(n) / fs_hz

    # Build ONE beat template, then tile it (fast)
    rr_sec = 60.0 / hr_bpm
    beat_samples = int(round(rr_sec * fs_hz))
    t_beat = np.arange(beat_samples) / fs_hz
    center = rr_sec / 2.0

    beat = np.zeros(beat_samples)
    beat += 1.2 * np.exp(-0.5 * ((t_beat - center) / 0.015) ** 2)           # R
    beat += -0.15 * np.exp(-0.5 * ((t_beat - (center - 0.04)) / 0.01) ** 2)  # Q
    beat += -0.25 * np.exp(-0.5 * ((t_beat - (center + 0.02)) / 0.012) ** 2) # S
    beat += 0.18 * np.exp(-0.5 * ((t_beat - (center - 0.18)) / 0.03) ** 2)   # P
    beat += 0.28 * np.exp(-0.5 * ((t_beat - (center + 0.22)) / 0.05) ** 2)   # T

    num_beats = int(np.ceil(n / beat_samples))
    signal = np.tile(beat, num_beats)[:n]

    baseline = 0.1 * np.sin(2 * np.pi * 0.2 * t_full)
    powerline = 0.02 * np.sin(2 * np.pi * 50.0 * t_full)
    noise = np.random.normal(0.0, noise_std, size=n)

    print(f"[replayer] generated {duration_sec}s synthetic ECG ({n} samples, {hr_bpm} bpm)")
    return signal + baseline + powerline + noise


def load_ecg_file(path: Path) -> np.ndarray:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ESP32 simulator: publish ECG chunks to MQTT")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--topic", default=INPUT_TOPIC)
    parser.add_argument("--patient-id", default="patient1")
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--chunk-ms", type=int, default=250, help="Chunk duration in ms (200-500 recommended)")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = real-time, 2.0 = twice faster")
    parser.add_argument("--mode", choices=["synthetic", "file"], default="synthetic")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--hr-bpm", type=float, default=72.0)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--loop", action="store_true", help="Loop file mode forever")
    parser.add_argument("--inject-gap-every", type=int, default=0, help="Inject a gap every N chunks")
    parser.add_argument("--inject-oop-every", type=int, default=0, help="Inject out-of-order chunk every N chunks")
    args = parser.parse_args()

    if args.chunk_ms < 50:
        raise ValueError("chunk-ms too low")

    samples_per_chunk = max(1, int(round(args.fs * (args.chunk_ms / 1000.0))))

    if args.mode == "file":
        if not args.file:
            raise ValueError("--file is required in file mode")
        ecg = load_ecg_file(Path(args.file))
    else:
        ecg = generate_synthetic_ecg(args.duration_sec, args.fs, args.hr_bpm, args.noise_std)

    mqtt_cfg = MqttConfig(host=args.broker_host, port=args.broker_port)
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"ecg-replayer-{random.randint(1000,9999)}",
    )

    def on_connect(_c: mqtt.Client, _userdata, _connect_flags, reason_code, _properties):
        if reason_code == 0:
            print(f"[replayer] connected to MQTT {mqtt_cfg.host}:{mqtt_cfg.port}")
        else:
            print(f"[replayer] MQTT connection failed reason={reason_code}")

    def on_disconnect(_c: mqtt.Client, _userdata, _disconnect_flags, reason_code, _properties):
        print(f"[replayer] disconnected reason={reason_code}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    while True:
        try:
            client.connect(mqtt_cfg.host, mqtt_cfg.port, mqtt_cfg.keepalive)
            break
        except OSError as ex:
            print(f"[replayer] broker not reachable at {mqtt_cfg.host}:{mqtt_cfg.port} ({ex}); retrying in 2s...")
            time.sleep(2)

    client.loop_start()

    index = 0
    chunk_idx = 0
    stream_t0_ms = now_ms()

    try:
        while True:
            if index >= len(ecg):
                if args.mode == "file" and args.loop:
                    index = 0
                else:
                    print("[replayer] completed stream")
                    break

            end_idx = min(index + samples_per_chunk, len(ecg))
            chunk = ecg[index:end_idx]
            if chunk.size == 0:
                break

            t0_ms = stream_t0_ms + chunk_duration_ms(index, args.fs)

            if args.inject_gap_every > 0 and chunk_idx > 0 and chunk_idx % args.inject_gap_every == 0:
                t0_ms += args.chunk_ms * 2

            if args.inject_oop_every > 0 and chunk_idx > 0 and chunk_idx % args.inject_oop_every == 0:
                t0_ms -= args.chunk_ms

            payload = {
                "patient_id": args.patient_id,
                "t0_ms": int(t0_ms),
                "fs_hz": int(args.fs),
                "samples": chunk.astype(float).tolist(),
            }

            info = client.publish(args.topic, to_json(payload), qos=1)
            info.wait_for_publish(timeout=2.0)

            if chunk_idx % 10 == 0:
                print(
                    f"[replayer] topic={args.topic} chunk={chunk_idx} "
                    f"samples={len(payload['samples'])} t0={payload['t0_ms']}"
                )

            elapsed = (len(chunk) / args.fs) / max(args.speed, 0.1)
            time.sleep(elapsed)

            index = end_idx
            chunk_idx += 1

            if args.mode == "file" and index >= len(ecg) and args.loop:
                stream_t0_ms = stream_t0_ms + chunk_duration_ms(len(ecg), args.fs)

    except KeyboardInterrupt:
        print("[replayer] stopping (Ctrl+C)")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
