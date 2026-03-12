from __future__ import annotations

"""
Multi-Sensor AI Dashboard Viewer
=================================
Subscribes to edge features and events topics, displays real-time dashboard
with full AI pipeline results.

Topics subscribed:
- edge/patient1/features (unified sensor features + ML + decision)
- edge/patient1/events   (clinical alerts and events)

Displays:
- ECG HR + PPG HR
- Signal Quality Index
- SpO2 mean/min
- Temperature
- HRV (SDNN, RMSSD) + Respiration
- ML predictions (risk, deterioration, class)
- Decision (severity/color/action)
- Recent events log
"""

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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
    # New AI pipeline fields
    hrv_sdnn: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    hrv_rmssd: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    resp_rate: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    ml_risk: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    ml_deterioration: Deque[Optional[float]] = field(default_factory=lambda: deque(maxlen=120))
    ml_class: Deque[str] = field(default_factory=lambda: deque(maxlen=120))
    decision_severity: Deque[str] = field(default_factory=lambda: deque(maxlen=120))
    decision_color: Deque[str] = field(default_factory=lambda: deque(maxlen=120))
    decision_action: Deque[str] = field(default_factory=lambda: deque(maxlen=120))

    def __post_init__(self):
        for f in [
            "timestamps", "ecg_hr", "ecg_sqi", "ppg_hr", "ppg_sqi",
            "spo2_mean", "spo2_min", "temp_mean", "motion_score",
            "hrv_sdnn", "hrv_rmssd", "resp_rate",
            "ml_risk", "ml_deterioration", "ml_class",
            "decision_severity", "decision_color", "decision_action",
        ]:
            setattr(self, f, deque(maxlen=self.max_points))

    def add_features(self, payload: Dict[str, Any]) -> None:
        """Add features from a payload."""
        self.timestamps.append(payload.get("window_start_ms", 0))

        ecg = payload.get("ecg", {})
        self.ecg_hr.append(ecg.get("hr_mean") or ecg.get("hr_bpm"))
        self.ecg_sqi.append(ecg.get("sqi", 0.0))

        ppg = payload.get("ppg", {})
        self.ppg_hr.append(ppg.get("pulse_rate") or ppg.get("hr_bpm"))
        self.ppg_sqi.append(ppg.get("sqi", 0.0))

        spo2 = payload.get("spo2", {})
        self.spo2_mean.append(spo2.get("mean"))
        self.spo2_min.append(spo2.get("min"))

        temp = payload.get("temp_c", {})
        self.temp_mean.append(temp.get("mean"))

        motion = payload.get("motion", {})
        self.motion_score.append(motion.get("score", 0.0))

        # HRV
        self.hrv_sdnn.append(ecg.get("hrv_sdnn"))
        self.hrv_rmssd.append(ecg.get("hrv_rmssd"))

        # Respiration
        resp = payload.get("respiration", {})
        self.resp_rate.append(resp.get("rate_bpm"))

        # ML
        ml = payload.get("ml", {})
        self.ml_risk.append(ml.get("risk_score"))
        self.ml_deterioration.append(ml.get("deterioration_prob"))
        self.ml_class.append(ml.get("event_class", ""))

        # Decision
        decision = payload.get("decision", {})
        self.decision_severity.append(decision.get("severity", ""))
        self.decision_color.append(decision.get("color", ""))
        self.decision_action.append(decision.get("action", ""))


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

SEVERITY_SYMBOLS = {
    "low": "\033[92m[LOW]\033[0m",
    "moderate": "\033[93m[MOD]\033[0m",
    "high": "\033[91m[HIGH]\033[0m",
    "critical": "\033[1;91m[CRIT]\033[0m",
}

COLOR_SYMBOLS = {
    "green": "\033[92m GREEN \033[0m",
    "yellow": "\033[93m YELLOW \033[0m",
    "orange": "\033[33m ORANGE \033[0m",
    "red": "\033[1;91m  RED  \033[0m",
}


def format_features_console(payload: Dict[str, Any]) -> str:
    """Format features for console output with AI results."""
    patient_id = payload.get("patient_id", "?")

    ecg = payload.get("ecg", {})
    ppg = payload.get("ppg", {})
    spo2 = payload.get("spo2", {})
    temp = payload.get("temp_c", {})
    motion = payload.get("motion", {})
    resp = payload.get("respiration", {})
    ml = payload.get("ml", {})
    decision = payload.get("decision", {})
    rules = payload.get("rules", {})

    ecg_hr = ecg.get("hr_mean") or ecg.get("hr_bpm")
    ppg_hr = ppg.get("pulse_rate") or ppg.get("hr_bpm")
    spo2_val = spo2.get("mean")
    temp_val = temp.get("mean")
    motion_val = motion.get("score", 0)
    resp_val = resp.get("rate_bpm")

    ecg_hr_str = f"{ecg_hr:.0f}" if ecg_hr else "--"
    ppg_hr_str = f"{ppg_hr:.0f}" if ppg_hr else "--"
    spo2_str = f"{spo2_val:.0f}%" if spo2_val else "--"
    temp_str = f"{temp_val:.1f}C" if temp_val else "--"
    resp_str = f"{resp_val:.0f}" if resp_val else "--"

    # Decision color
    color = decision.get("color", "green")
    color_str = COLOR_SYMBOLS.get(color, color)
    action = decision.get("action", "")
    ml_class = ml.get("event_class", "")
    risk = ml.get("risk_score")
    risk_str = f"{risk:.2f}" if risk is not None else "--"
    triggered = rules.get("triggered", [])

    return (
        f"[{patient_id}] HR:{ecg_hr_str}/{ppg_hr_str} SpO2:{spo2_str} "
        f"Temp:{temp_str} Resp:{resp_str} Motion:{motion_val:.2f} | "
        f"ML:{ml_class}(risk={risk_str}) | "
        f"{color_str} {action}"
        + (f" | Rules: {triggered}" if triggered else "")
    )


def format_event_console(payload: Dict[str, Any]) -> str:
    """Format event for console output."""
    patient_id = payload.get("patient_id", "?")
    event_type = payload.get("type", "?")
    severity = payload.get("severity", "?")
    details = payload.get("details", {})

    sev_str = SEVERITY_SYMBOLS.get(severity, f"[{severity}]")

    return f"[{patient_id}] {sev_str} EVENT: {event_type} - {details}"


# =============================================================================
# MATPLOTLIB VIEWER
# =============================================================================

# Color map for decision colors in matplotlib
DECISION_MPL_COLORS = {
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "orange": "#e67e22",
    "red": "#e74c3c",
}


class DashboardPlotter:
    """Real-time multi-sensor AI dashboard using matplotlib."""

    def __init__(self, features_data: FeaturesData, events_data: EventData, lock: threading.Lock):
        self.features = features_data
        self.events = events_data
        self.lock = lock

        # Create figure with 4x2 grid
        self.fig, self.axes = plt.subplots(4, 2, figsize=(16, 12))
        self.fig.suptitle("Edge AI Patient Monitor", fontsize=14, fontweight="bold")

        # Assign axes
        self.ax_hr = self.axes[0, 0]
        self.ax_sqi = self.axes[0, 1]
        self.ax_spo2 = self.axes[1, 0]
        self.ax_temp = self.axes[1, 1]
        self.ax_hrv = self.axes[2, 0]
        self.ax_resp_motion = self.axes[2, 1]
        self.ax_ml = self.axes[3, 0]
        self.ax_decision = self.axes[3, 1]

        self._setup_axes()

    def _setup_axes(self) -> None:
        """Initial axis setup."""
        for ax in self.axes.flat:
            ax.grid(True, alpha=0.3)

        self.ax_hr.set_title("Heart Rate (ECG & PPG)")
        self.ax_hr.set_ylabel("HR (bpm)")
        self.ax_hr.set_ylim(40, 160)

        self.ax_sqi.set_title("Signal Quality Index")
        self.ax_sqi.set_ylabel("SQI (0-1)")
        self.ax_sqi.set_ylim(0, 1.1)
        self.ax_sqi.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

        self.ax_spo2.set_title("SpO2")
        self.ax_spo2.set_ylabel("SpO2 (%)")
        self.ax_spo2.set_ylim(80, 102)
        self.ax_spo2.axhline(y=90, color='r', linestyle='--', alpha=0.5, label="Critical (90%)")

        self.ax_temp.set_title("Temperature")
        self.ax_temp.set_ylabel("Temp (°C)")
        self.ax_temp.set_ylim(35, 41)
        self.ax_temp.axhline(y=38, color='r', linestyle='--', alpha=0.5, label="Fever (38°C)")

        self.ax_hrv.set_title("HRV (SDNN & RMSSD)")
        self.ax_hrv.set_ylabel("ms")
        self.ax_hrv.set_ylim(0, 200)

        self.ax_resp_motion.set_title("Respiration & Motion")
        self.ax_resp_motion.set_ylabel("Resp (bpm) / Motion (0-1)")

        self.ax_ml.set_title("ML Predictions")
        self.ax_ml.set_ylabel("Probability (0-1)")
        self.ax_ml.set_ylim(0, 1.1)
        self.ax_ml.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label="High risk")
        self.ax_ml.axhline(y=0.5, color='gold', linestyle='--', alpha=0.5, label="Moderate risk")

        self.ax_decision.set_title("Decision & Events")
        self.ax_decision.axis("off")

        plt.tight_layout()

    @staticmethod
    def _filter_none(data):
        """Return (x_indices, y_values) with None entries removed."""
        x = [i for i, v in enumerate(data) if v is not None]
        y = [v for v in data if v is not None]
        return x, y

    def update(self, _frame) -> None:
        """Update all plots."""
        with self.lock:
            n = len(self.features.timestamps)
            if n == 0:
                return

            x = list(range(n))

            # --- Heart Rate ---
            self.ax_hr.clear()
            self.ax_hr.set_title("Heart Rate (ECG & PPG)")
            self.ax_hr.set_ylabel("HR (bpm)")
            self.ax_hr.grid(True, alpha=0.3)
            self.ax_hr.axhline(y=120, color='r', linestyle='--', alpha=0.4)
            self.ax_hr.axhline(y=60, color='b', linestyle='--', alpha=0.3)

            ecg_x, ecg_y = self._filter_none(self.features.ecg_hr)
            ppg_x, ppg_y = self._filter_none(self.features.ppg_hr)
            if ecg_y:
                self.ax_hr.plot(ecg_x, ecg_y, "b-o", markersize=3, label=f"ECG ({ecg_y[-1]:.0f})")
            if ppg_y:
                self.ax_hr.plot(ppg_x, ppg_y, "g-s", markersize=3, label=f"PPG ({ppg_y[-1]:.0f})")
            self.ax_hr.legend(loc="upper right", fontsize=8)
            self.ax_hr.set_ylim(40, 160)

            # --- SQI ---
            self.ax_sqi.clear()
            self.ax_sqi.set_title("Signal Quality Index")
            self.ax_sqi.set_ylabel("SQI (0-1)")
            self.ax_sqi.grid(True, alpha=0.3)
            ecg_sqi = list(self.features.ecg_sqi)
            ppg_sqi = list(self.features.ppg_sqi)
            self.ax_sqi.plot(x, ecg_sqi, "b-", linewidth=1.5,
                             label=f"ECG ({ecg_sqi[-1]:.2f})" if ecg_sqi else "ECG")
            self.ax_sqi.plot(x, ppg_sqi, "g-", linewidth=1.5,
                             label=f"PPG ({ppg_sqi[-1]:.2f})" if ppg_sqi else "PPG")
            self.ax_sqi.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label="Low quality")
            self.ax_sqi.legend(loc="upper right", fontsize=8)
            self.ax_sqi.set_ylim(0, 1.1)

            # --- SpO2 ---
            self.ax_spo2.clear()
            self.ax_spo2.set_title("SpO2")
            self.ax_spo2.set_ylabel("SpO2 (%)")
            self.ax_spo2.grid(True, alpha=0.3)
            mean_x, mean_y = self._filter_none(self.features.spo2_mean)
            min_x, min_y = self._filter_none(self.features.spo2_min)
            if mean_y:
                self.ax_spo2.plot(mean_x, mean_y, "c-o", markersize=3, label=f"Mean ({mean_y[-1]:.1f}%)")
            if min_y:
                self.ax_spo2.plot(min_x, min_y, "m-s", markersize=3, alpha=0.7, label=f"Min ({min_y[-1]:.1f}%)")
            self.ax_spo2.axhline(y=90, color='r', linestyle='--', alpha=0.5, label="Critical (90%)")
            self.ax_spo2.legend(loc="upper right", fontsize=8)
            self.ax_spo2.set_ylim(80, 102)

            # --- Temperature ---
            self.ax_temp.clear()
            self.ax_temp.set_title("Temperature")
            self.ax_temp.set_ylabel("Temp (°C)")
            self.ax_temp.grid(True, alpha=0.3)
            temp_x, temp_y = self._filter_none(self.features.temp_mean)
            if temp_y:
                self.ax_temp.plot(temp_x, temp_y, "r-o", markersize=3, label=f"Temp ({temp_y[-1]:.1f}°C)")
                self.ax_temp.legend(loc="upper right", fontsize=8)
            self.ax_temp.axhline(y=38, color='r', linestyle='--', alpha=0.5)
            self.ax_temp.set_ylim(35, 41)

            # --- HRV ---
            self.ax_hrv.clear()
            self.ax_hrv.set_title("HRV (SDNN & RMSSD)")
            self.ax_hrv.set_ylabel("ms")
            self.ax_hrv.grid(True, alpha=0.3)
            sdnn_x, sdnn_y = self._filter_none(self.features.hrv_sdnn)
            rmssd_x, rmssd_y = self._filter_none(self.features.hrv_rmssd)
            if sdnn_y:
                self.ax_hrv.plot(sdnn_x, sdnn_y, "purple", linewidth=1.5, marker='o', markersize=3,
                                 label=f"SDNN ({sdnn_y[-1]:.1f}ms)")
            if rmssd_y:
                self.ax_hrv.plot(rmssd_x, rmssd_y, "teal", linewidth=1.5, marker='s', markersize=3,
                                 label=f"RMSSD ({rmssd_y[-1]:.1f}ms)")
            self.ax_hrv.legend(loc="upper right", fontsize=8)
            self.ax_hrv.set_ylim(0, max(200, max(sdnn_y + rmssd_y, default=100) * 1.2))

            # --- Respiration & Motion ---
            self.ax_resp_motion.clear()
            self.ax_resp_motion.set_title("Respiration & Motion")
            self.ax_resp_motion.grid(True, alpha=0.3)
            resp_x, resp_y = self._filter_none(self.features.resp_rate)
            motion_vals = list(self.features.motion_score)
            ax2 = self.ax_resp_motion.twinx()
            if resp_y:
                self.ax_resp_motion.plot(resp_x, resp_y, "darkgreen", linewidth=1.5, marker='o', markersize=3,
                                         label=f"Resp ({resp_y[-1]:.0f} bpm)")
            self.ax_resp_motion.set_ylabel("Resp rate (bpm)", color="darkgreen")
            self.ax_resp_motion.set_ylim(0, 40)
            if motion_vals:
                ax2.fill_between(x, 0, motion_vals, alpha=0.2, color="orange")
                ax2.plot(x, motion_vals, "orange", linewidth=1,
                         label=f"Motion ({motion_vals[-1]:.2f})")
            ax2.set_ylabel("Motion (0-1)", color="orange")
            ax2.set_ylim(0, 1.1)
            # Combine legends
            lines1, labels1 = self.ax_resp_motion.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.ax_resp_motion.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

            # --- ML Predictions ---
            self.ax_ml.clear()
            self.ax_ml.set_title("ML Predictions")
            self.ax_ml.set_ylabel("Probability (0-1)")
            self.ax_ml.grid(True, alpha=0.3)
            risk_x, risk_y = self._filter_none(self.features.ml_risk)
            det_x, det_y = self._filter_none(self.features.ml_deterioration)
            if risk_y:
                self.ax_ml.plot(risk_x, risk_y, "red", linewidth=2, marker='o', markersize=3,
                                label=f"Risk ({risk_y[-1]:.2f})")
            if det_y:
                self.ax_ml.plot(det_x, det_y, "darkviolet", linewidth=2, marker='s', markersize=3,
                                label=f"Deterioration ({det_y[-1]:.2f})")
            self.ax_ml.axhline(y=0.7, color='orange', linestyle='--', alpha=0.4)
            self.ax_ml.axhline(y=0.5, color='gold', linestyle='--', alpha=0.4)
            # Show last ML class
            ml_classes = list(self.features.ml_class)
            if ml_classes and ml_classes[-1]:
                self.ax_ml.text(0.02, 0.95, f"Class: {ml_classes[-1]}",
                                transform=self.ax_ml.transAxes, fontsize=10,
                                fontweight="bold", verticalalignment="top",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            self.ax_ml.legend(loc="upper right", fontsize=8)
            self.ax_ml.set_ylim(0, 1.1)

            # --- Decision & Events ---
            self.ax_decision.clear()
            self.ax_decision.set_title("Decision & Events")
            self.ax_decision.axis("off")

            # Build decision text
            dec_lines = []
            colors = list(self.features.decision_color)
            actions = list(self.features.decision_action)
            severities = list(self.features.decision_severity)
            if colors:
                last_color = colors[-1]
                last_action = actions[-1] if actions else ""
                last_severity = severities[-1] if severities else ""
                dec_color = DECISION_MPL_COLORS.get(last_color, "gray")

                # Large colored decision box
                self.ax_decision.add_patch(plt.Rectangle((0.05, 0.65), 0.9, 0.3,
                                                          transform=self.ax_decision.transAxes,
                                                          facecolor=dec_color, alpha=0.3,
                                                          edgecolor=dec_color, linewidth=2))
                self.ax_decision.text(0.5, 0.8,
                                       f"{last_color.upper()} - {last_severity.upper()}",
                                       transform=self.ax_decision.transAxes,
                                       fontsize=14, fontweight="bold",
                                       horizontalalignment="center", verticalalignment="center",
                                       color=dec_color)
                self.ax_decision.text(0.5, 0.68,
                                       last_action.replace("_", " ").title(),
                                       transform=self.ax_decision.transAxes,
                                       fontsize=10,
                                       horizontalalignment="center", verticalalignment="center",
                                       style="italic")

            # Recent events
            events_list = list(self.events.events)[-8:]
            if events_list:
                event_text = "\n".join([
                    f"  {e.get('type', '?')} ({e.get('severity', '?')})"
                    for e in reversed(events_list)
                ])
                self.ax_decision.text(
                    0.05, 0.55, "Recent Events:\n" + event_text,
                    transform=self.ax_decision.transAxes,
                    verticalalignment="top",
                    fontsize=8,
                    family="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
                )
            else:
                self.ax_decision.text(
                    0.5, 0.35, "No clinical events",
                    transform=self.ax_decision.transAxes,
                    horizontalalignment="center", verticalalignment="center",
                    fontsize=10, color="gray"
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
        description="Multi-sensor AI dashboard viewer",
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
            print("[viewer] starting AI dashboard plots...")
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
