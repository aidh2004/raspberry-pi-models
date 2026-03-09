from __future__ import annotations

"""
Multi-Sensor Dashboard Viewer
==============================
Subscribes to edge features and events topics, displays real-time dashboard.

Topics subscribed:
- edge/patient1/features (unified sensor features)
- edge/patient1/events (alerts and events)

Displays:
- ECG HR + SQI
- PPG HR + SQI
- SpO2 mean/min
- Temperature
- Motion score
- Recent events log
"""

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import paho.mqtt.client as mqtt

from common import (
    DEFAULT_DEVICE_ID,
    MqttConfig,
    get_output_topics,
    parse_json,
    parse_topic,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FeaturesData:
    """Stores time-series of features for plotting."""
    max_points: int = 120

    timestamps: Deque[int] = field(default_factory=lambda: deque(maxlen=120))
    ecg_hr: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    ecg_sqi: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    ppg_hr: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    ppg_sqi: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    spo2_mean: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    spo2_min: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    temp_mean: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    motion_score: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def __post_init__(self):
        self.timestamps = deque(maxlen=self.max_points)
        self.ecg_hr = deque(maxlen=self.max_points)
        self.ecg_sqi = deque(maxlen=self.max_points)
        self.ppg_hr = deque(maxlen=self.max_points)
        self.ppg_sqi = deque(maxlen=self.max_points)
        self.spo2_mean = deque(maxlen=self.max_points)
        self.spo2_min = deque(maxlen=self.max_points)
        self.temp_mean = deque(maxlen=self.max_points)
        self.motion_score = deque(maxlen=self.max_points)

    def add_features(self, payload: Dict[str, Any]) -> None:
        """Add features from a payload."""
        self.timestamps.append(payload.get("window_start_ms", 0))

        ecg = payload.get("ecg", {})
        self.ecg_hr.append(ecg.get("hr_bpm"))
        self.ecg_sqi.append(ecg.get("sqi", 0.0))

        ppg = payload.get("ppg", {})
        self.ppg_hr.append(ppg.get("hr_bpm"))
        self.ppg_sqi.append(ppg.get("sqi", 0.0))

        spo2 = payload.get("spo2", {})
        self.spo2_mean.append(spo2.get("mean"))
        self.spo2_min.append(spo2.get("min"))

        temp = payload.get("temp_c", {})
        self.temp_mean.append(temp.get("mean"))

        motion = payload.get("motion", {})
        self.motion_score.append(motion.get("score", 0.0))


@dataclass
class EventData:
    """Stores recent events."""
    max_events: int = 50
    events: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self):
        self.events = deque(maxlen=self.max_events)

    def add_event(self, payload: Dict[str, Any]) -> None:
        """Add an event."""
        self.events.append(payload)


# =============================================================================
# CONSOLE VIEWER
# =============================================================================

def format_features_console(payload: Dict[str, Any]) -> str:
    """Format features for console output."""
    patient_id = payload.get("patient_id", "?")
    window_ms = payload.get("window_start_ms", 0)

    ecg = payload.get("ecg", {})
    ppg = payload.get("ppg", {})
    spo2 = payload.get("spo2", {})
    temp = payload.get("temp_c", {})
    motion = payload.get("motion", {})

    ecg_hr = ecg.get("hr_bpm")
    ppg_hr = ppg.get("hr_bpm")
    spo2_val = spo2.get("mean")
    temp_val = temp.get("mean")
    motion_val = motion.get("score", 0)

    ecg_hr_str = f"{ecg_hr:.1f}" if ecg_hr else "--"
    ppg_hr_str = f"{ppg_hr:.1f}" if ppg_hr else "--"
    spo2_str = f"{spo2_val:.1f}%" if spo2_val else "--"
    temp_str = f"{temp_val:.1f}°C" if temp_val else "--"
    motion_str = f"{motion_val:.2f}"

    return (
        f"[{patient_id}] ECG HR: {ecg_hr_str} bpm (SQI: {ecg.get('sqi', 0):.2f}) | "
        f"PPG HR: {ppg_hr_str} bpm (SQI: {ppg.get('sqi', 0):.2f}) | "
        f"SpO2: {spo2_str} | Temp: {temp_str} | Motion: {motion_str}"
    )


def format_event_console(payload: Dict[str, Any]) -> str:
    """Format event for console output."""
    patient_id = payload.get("patient_id", "?")
    event_type = payload.get("type", "?")
    severity = payload.get("severity", "?")
    details = payload.get("details", {})

    severity_symbol = {"low": "○", "moderate": "◐", "high": "●"}.get(severity, "?")

    return f"[{patient_id}] {severity_symbol} EVENT: {event_type} ({severity}) - {details}"


# =============================================================================
# MATPLOTLIB VIEWER
# =============================================================================

class DashboardPlotter:
    """Real-time multi-sensor dashboard using matplotlib."""

    def __init__(self, features_data: FeaturesData, events_data: EventData, lock: threading.Lock):
        self.features = features_data
        self.events = events_data
        self.lock = lock

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(3, 2, figsize=(14, 10))
        self.fig.suptitle("Multi-Sensor Patient Monitor", fontsize=14, fontweight="bold")

        # Configure axes
        self.ax_hr = self.axes[0, 0]
        self.ax_sqi = self.axes[0, 1]
        self.ax_spo2 = self.axes[1, 0]
        self.ax_temp = self.axes[1, 1]
        self.ax_motion = self.axes[2, 0]
        self.ax_events = self.axes[2, 1]

        self._setup_axes()

    def _setup_axes(self) -> None:
        """Initial axis setup."""
        self.ax_hr.set_title("Heart Rate (ECG & PPG)")
        self.ax_hr.set_ylabel("HR (bpm)")
        self.ax_hr.set_xlabel("Window")
        self.ax_hr.grid(True, alpha=0.3)
        self.ax_hr.set_ylim(40, 120)

        self.ax_sqi.set_title("Signal Quality Index")
        self.ax_sqi.set_ylabel("SQI (0-1)")
        self.ax_sqi.set_xlabel("Window")
        self.ax_sqi.grid(True, alpha=0.3)
        self.ax_sqi.set_ylim(0, 1.1)

        self.ax_spo2.set_title("SpO2")
        self.ax_spo2.set_ylabel("SpO2 (%)")
        self.ax_spo2.set_xlabel("Window")
        self.ax_spo2.grid(True, alpha=0.3)
        self.ax_spo2.set_ylim(85, 102)
        self.ax_spo2.axhline(y=93, color='r', linestyle='--', alpha=0.5, label="Warning (93%)")

        self.ax_temp.set_title("Temperature")
        self.ax_temp.set_ylabel("Temp (°C)")
        self.ax_temp.set_xlabel("Window")
        self.ax_temp.grid(True, alpha=0.3)
        self.ax_temp.set_ylim(35, 40)

        self.ax_motion.set_title("Motion Score")
        self.ax_motion.set_ylabel("Score (0-1)")
        self.ax_motion.set_xlabel("Window")
        self.ax_motion.grid(True, alpha=0.3)
        self.ax_motion.set_ylim(0, 1.1)
        self.ax_motion.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="High motion")

        self.ax_events.set_title("Recent Events")
        self.ax_events.axis("off")

        plt.tight_layout()

    def update(self, _frame) -> None:
        """Update all plots."""
        with self.lock:
            n = len(self.features.timestamps)
            if n == 0:
                return

            x = list(range(n))

            # Heart Rate plot
            self.ax_hr.clear()
            self.ax_hr.set_title("Heart Rate (ECG & PPG)")
            self.ax_hr.set_ylabel("HR (bpm)")
            self.ax_hr.grid(True, alpha=0.3)

            ecg_hr = [v for v in self.features.ecg_hr]
            ppg_hr = [v for v in self.features.ppg_hr]

            # Plot valid values only
            ecg_x = [i for i, v in enumerate(ecg_hr) if v is not None]
            ecg_y = [v for v in ecg_hr if v is not None]
            ppg_x = [i for i, v in enumerate(ppg_hr) if v is not None]
            ppg_y = [v for v in ppg_hr if v is not None]

            if ecg_y:
                self.ax_hr.plot(ecg_x, ecg_y, "b-o", markersize=3, label=f"ECG ({ecg_y[-1]:.0f})")
            if ppg_y:
                self.ax_hr.plot(ppg_x, ppg_y, "g-s", markersize=3, label=f"PPG ({ppg_y[-1]:.0f})")
            self.ax_hr.legend(loc="upper right")
            self.ax_hr.set_ylim(40, 120)

            # SQI plot
            self.ax_sqi.clear()
            self.ax_sqi.set_title("Signal Quality Index")
            self.ax_sqi.set_ylabel("SQI (0-1)")
            self.ax_sqi.grid(True, alpha=0.3)

            ecg_sqi = list(self.features.ecg_sqi)
            ppg_sqi = list(self.features.ppg_sqi)

            self.ax_sqi.plot(x, ecg_sqi, "b-", linewidth=1.5, label=f"ECG ({ecg_sqi[-1]:.2f})" if ecg_sqi else "ECG")
            self.ax_sqi.plot(x, ppg_sqi, "g-", linewidth=1.5, label=f"PPG ({ppg_sqi[-1]:.2f})" if ppg_sqi else "PPG")
            self.ax_sqi.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            self.ax_sqi.legend(loc="upper right")
            self.ax_sqi.set_ylim(0, 1.1)

            # SpO2 plot
            self.ax_spo2.clear()
            self.ax_spo2.set_title("SpO2")
            self.ax_spo2.set_ylabel("SpO2 (%)")
            self.ax_spo2.grid(True, alpha=0.3)

            spo2_mean = [v for v in self.features.spo2_mean]
            spo2_min = [v for v in self.features.spo2_min]

            mean_x = [i for i, v in enumerate(spo2_mean) if v is not None]
            mean_y = [v for v in spo2_mean if v is not None]
            min_x = [i for i, v in enumerate(spo2_min) if v is not None]
            min_y = [v for v in spo2_min if v is not None]

            if mean_y:
                self.ax_spo2.plot(mean_x, mean_y, "c-o", markersize=3, label=f"Mean ({mean_y[-1]:.1f}%)")
            if min_y:
                self.ax_spo2.plot(min_x, min_y, "m-s", markersize=3, alpha=0.7, label=f"Min ({min_y[-1]:.1f}%)")
            self.ax_spo2.axhline(y=93, color='r', linestyle='--', alpha=0.5)
            self.ax_spo2.legend(loc="upper right")
            self.ax_spo2.set_ylim(85, 102)

            # Temperature plot
            self.ax_temp.clear()
            self.ax_temp.set_title("Temperature")
            self.ax_temp.set_ylabel("Temp (°C)")
            self.ax_temp.grid(True, alpha=0.3)

            temp_mean = [v for v in self.features.temp_mean]
            temp_x = [i for i, v in enumerate(temp_mean) if v is not None]
            temp_y = [v for v in temp_mean if v is not None]

            if temp_y:
                self.ax_temp.plot(temp_x, temp_y, "r-o", markersize=3, label=f"Temp ({temp_y[-1]:.1f}°C)")
                self.ax_temp.legend(loc="upper right")
            self.ax_temp.set_ylim(35, 40)

            # Motion plot
            self.ax_motion.clear()
            self.ax_motion.set_title("Motion Score")
            self.ax_motion.set_ylabel("Score (0-1)")
            self.ax_motion.grid(True, alpha=0.3)

            motion = list(self.features.motion_score)
            self.ax_motion.fill_between(x, 0, motion, alpha=0.3, color="orange")
            self.ax_motion.plot(x, motion, "orange", linewidth=1.5, label=f"Motion ({motion[-1]:.2f})" if motion else "Motion")
            self.ax_motion.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            self.ax_motion.legend(loc="upper right")
            self.ax_motion.set_ylim(0, 1.1)

            # Events panel
            self.ax_events.clear()
            self.ax_events.set_title("Recent Events")
            self.ax_events.axis("off")

            events_list = list(self.events.events)[-10:]  # Last 10 events
            if events_list:
                event_text = "\n".join([
                    f"• {e.get('type', '?')} ({e.get('severity', '?')})"
                    for e in reversed(events_list)
                ])
                self.ax_events.text(
                    0.05, 0.95, event_text,
                    transform=self.ax_events.transAxes,
                    verticalalignment="top",
                    fontsize=9,
                    family="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
                )
            else:
                self.ax_events.text(
                    0.5, 0.5, "No events",
                    transform=self.ax_events.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    color="gray"
                )

        plt.tight_layout()

    def run(self) -> None:
        """Start the animation loop."""
        self.ani = FuncAnimation(self.fig, self.update, interval=500, cache_frame_data=False)
        plt.show(block=True)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-sensor dashboard viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Console-only view:
  python viewer.py

  # With live plots:
  python viewer.py --plot

  # Custom device:
  python viewer.py --device-id patient2 --plot
        """
    )
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--plot", action="store_true", help="Show live plots (requires matplotlib)")
    parser.add_argument("--max-points", type=int, default=120, help="Max data points to display")

    # Legacy compatibility
    parser.add_argument("--topic", default=None, help="(Deprecated) Use device-id instead")

    args = parser.parse_args()

    if args.plot and not MPL_AVAILABLE:
        raise RuntimeError("matplotlib not available; install it or remove --plot")

    # Get topics
    topics = get_output_topics(args.device_id)
    features_topic = topics["features"]
    events_topic = topics["events"]

    # Create data stores
    features_data = FeaturesData(max_points=args.max_points)
    events_data = EventData(max_events=50)
    lock = threading.Lock()

    # Create MQTT client
    mqtt_cfg = MqttConfig(host=args.broker_host, port=args.broker_port)
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="multi-sensor-viewer",
    )

    def on_connect(c: mqtt.Client, _userdata, _connect_flags, reason_code, _properties):
        if reason_code == 0:
            print(f"[viewer] connected to MQTT {mqtt_cfg.host}:{mqtt_cfg.port}")
            c.subscribe(features_topic, qos=1)
            c.subscribe(events_topic, qos=1)
            print(f"[viewer] subscribed to {features_topic}")
            print(f"[viewer] subscribed to {events_topic}")
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

        # Determine message type from topic
        _, _, msg_type = parse_topic(msg.topic)

        if msg_type == "features":
            with lock:
                features_data.add_features(payload)
            print(f"[viewer] {format_features_console(payload)}")

        elif msg_type == "events":
            with lock:
                events_data.add_event(payload)
            print(f"[viewer] {format_event_console(payload)}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    print(f"[viewer] connecting to {mqtt_cfg.host}:{mqtt_cfg.port}...")
    while True:
        try:
            client.connect(mqtt_cfg.host, mqtt_cfg.port, mqtt_cfg.keepalive)
            break
        except OSError as ex:
            print(f"[viewer] broker not reachable ({ex}); retrying in 2s...")
            time.sleep(2)

    client.loop_start()

    try:
        if args.plot:
            print("[viewer] starting dashboard plots...")
            plotter = DashboardPlotter(features_data, events_data, lock)
            plotter.run()
        else:
            print("[viewer] running in console mode (use --plot for GUI)...")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[viewer] stopping")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
