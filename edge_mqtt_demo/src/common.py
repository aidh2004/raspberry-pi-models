from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

INPUT_TOPIC = "sim/patient1/ecg"
OUTPUT_TOPIC = "edge/patient1/features"


@dataclass
class MqttConfig:
    host: str = "localhost"
    port: int = 1883
    keepalive: int = 30


def now_ms() -> int:
    return int(time.time() * 1000)


def to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def parse_json(raw: bytes) -> Dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def validate_input_message(payload: Dict[str, Any]) -> Optional[str]:
    required = ["patient_id", "t0_ms", "fs_hz", "samples"]
    for key in required:
        if key not in payload:
            return f"missing_key:{key}"

    if not isinstance(payload["patient_id"], str):
        return "invalid_patient_id"
    if not isinstance(payload["t0_ms"], int):
        return "invalid_t0_ms"
    if not isinstance(payload["fs_hz"], (int, float)):
        return "invalid_fs_hz"
    if not isinstance(payload["samples"], list):
        return "invalid_samples"
    if len(payload["samples"]) == 0:
        return "empty_samples"

    return None


def validate_output_message(payload: Dict[str, Any]) -> Optional[str]:
    required = ["patient_id", "window_start_ms", "window_sec", "hr_bpm", "sqi", "notes"]
    for key in required:
        if key not in payload:
            return f"missing_key:{key}"
    return None


def chunk_duration_ms(samples_len: int, fs_hz: float) -> int:
    return int(round((samples_len / float(fs_hz)) * 1000.0))


def normalize_topic_patient(topic: str) -> Optional[str]:
    parts = topic.split("/")
    if len(parts) >= 2:
        return parts[1]
    return None


def safe_float_list(values: List[Any]) -> List[float]:
    out: List[float] = []
    for val in values:
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            out.append(0.0)
    return out
