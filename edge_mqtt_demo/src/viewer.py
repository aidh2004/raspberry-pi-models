from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from typing import Deque, List, Optional

import paho.mqtt.client as mqtt

from common import OUTPUT_TOPIC, MqttConfig, parse_json

try:
    import matplotlib.pyplot as plt

    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Server/dashboard simulator: view edge features")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--topic", default=OUTPUT_TOPIC)
    parser.add_argument("--plot", action="store_true", help="Plot HR trend live (requires matplotlib)")
    parser.add_argument("--max-points", type=int, default=120)
    args = parser.parse_args()

    if args.plot and not MPL_AVAILABLE:
        raise RuntimeError("matplotlib not available; remove --plot or install dependencies")

    mqtt_cfg = MqttConfig(host=args.broker_host, port=args.broker_port)
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="edge-viewer",
    )

    lock = threading.Lock()
    t_vals: Deque[int] = deque(maxlen=args.max_points)
    hr_vals: Deque[float] = deque(maxlen=args.max_points)

    def on_connect(c: mqtt.Client, _userdata, _connect_flags, reason_code, _properties):
        if reason_code == 0:
            print(f"[viewer] connected to MQTT {mqtt_cfg.host}:{mqtt_cfg.port}")
            c.subscribe(args.topic, qos=1)
            print(f"[viewer] subscribed to {args.topic}")
        else:
            print(f"[viewer] MQTT connection failed reason={reason_code}")

    def on_disconnect(_c: mqtt.Client, _userdata, _disconnect_flags, reason_code, _properties):
        print(f"[viewer] disconnected reason={reason_code}")

    def on_message(_c: mqtt.Client, _userdata, msg: mqtt.MQTTMessage):
        try:
            payload = parse_json(msg.payload)
        except Exception as ex:
            print(f"[viewer] invalid JSON payload: {ex}")
            return

        patient_id = payload.get("patient_id", "?")
        win_start = payload.get("window_start_ms", "?")
        hr_bpm = payload.get("hr_bpm", None)
        sqi = payload.get("sqi", None)
        notes: List[str] = payload.get("notes", [])

        print(f"[viewer] patient={patient_id} window={win_start} hr={hr_bpm} sqi={sqi} notes={notes}")

        if hr_bpm is not None:
            with lock:
                t_vals.append(int(win_start))
                hr_vals.append(float(hr_bpm))

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    while True:
        try:
            client.connect(mqtt_cfg.host, mqtt_cfg.port, mqtt_cfg.keepalive)
            break
        except OSError as ex:
            print(f"[viewer] broker not reachable at {mqtt_cfg.host}:{mqtt_cfg.port} ({ex}); retrying in 2s...")
            time.sleep(2)

    client.loop_start()

    try:
        if args.plot:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 4))
            line, = ax.plot([], [], marker="o")
            ax.set_title("Live HR (BPM)")
            ax.set_xlabel("Window index")
            ax.set_ylabel("HR (bpm)")
            ax.grid(True)

            while True:
                with lock:
                    hr_list: List[float] = list(hr_vals)
                x = list(range(len(hr_list)))
                line.set_data(x, hr_list)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.2)
        else:
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("[viewer] stopping (Ctrl+C)")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
