from __future__ import annotations

"""
Clinical Rules Engine — Smart Medical Mat
============================================
Implements rule-based anomaly detection from the PDF architecture:

Table 4 — Clinical rules:
  SpO2 < 90%             → CRITICAL  (hypoxemia)
  HR > 120 bpm           → HIGH      (tachycardia)
  HR < 45 bpm            → HIGH      (bradycardia)
  Temperature > 38°C     → MODERATE  (fever)

Section 6 — Additional detection conditions:
  HR > 100 bpm           → tachycardia (moderate)
  HR < 60 bpm            → bradycardia (moderate)
  Frequent nocturnal desaturations → apnea suspicion

All rules run BEFORE ML models.  Results are combined in decision_engine.
"""

from typing import Any, Dict, List, Optional

from common import EventType, Severity, create_event


def evaluate_clinical_rules(
    patient_id: str,
    t_ms: int,
    ecg_features: Dict[str, Any],
    ppg_features: Dict[str, Any],
    spo2_features: Dict[str, Any],
    temp_features: Dict[str, Any],
    motion_features: Dict[str, Any],
    respiration_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate all clinical rules against extracted features.

    Returns:
        {
            "triggered": [list of triggered rule dicts],
            "events": [list of event messages for MQTT],
            "severity": highest severity among triggered rules,
        }
    """
    triggered: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    max_severity = Severity.LOW

    # --- Get best HR estimate (ECG preferred, PPG fallback) ---
    hr_val = ecg_features.get("hr_mean") or ppg_features.get("pulse_rate")

    # === RULE 1: SpO2 < 90% → CRITICAL (hypoxemia) ===
    spo2_min = spo2_features.get("min")
    if spo2_min is not None and spo2_min < 90.0:
        severity = Severity.CRITICAL
        rule = {
            "rule": "spo2_critical",
            "condition": f"SpO2_min={spo2_min} < 90%",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.HYPOXEMIA,
            severity=severity,
            details={"spo2_min": spo2_min},
        ))
        max_severity = _higher_severity(max_severity, severity)

    # === RULE 2: HR > 120 bpm → HIGH (tachycardia) ===
    if hr_val is not None and hr_val > 120:
        severity = Severity.HIGH
        rule = {
            "rule": "tachycardia_high",
            "condition": f"HR={hr_val} > 120 bpm",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.TACHYCARDIA,
            severity=severity,
            details={"hr_bpm": hr_val},
        ))
        max_severity = _higher_severity(max_severity, severity)

    # === RULE 2b: HR > 100 bpm → MODERATE (tachycardia) ===
    elif hr_val is not None and hr_val > 100:
        severity = Severity.MODERATE
        rule = {
            "rule": "tachycardia_moderate",
            "condition": f"HR={hr_val} > 100 bpm",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.TACHYCARDIA,
            severity=severity,
            details={"hr_bpm": hr_val},
        ))
        max_severity = _higher_severity(max_severity, severity)

    # === RULE 3: HR < 45 bpm → HIGH (bradycardia) ===
    if hr_val is not None and hr_val < 45:
        severity = Severity.HIGH
        rule = {
            "rule": "bradycardia_high",
            "condition": f"HR={hr_val} < 45 bpm",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.BRADYCARDIA,
            severity=severity,
            details={"hr_bpm": hr_val},
        ))
        max_severity = _higher_severity(max_severity, severity)

    # === RULE 3b: HR < 60 bpm → MODERATE (bradycardia) ===
    elif hr_val is not None and hr_val < 60:
        severity = Severity.MODERATE
        rule = {
            "rule": "bradycardia_moderate",
            "condition": f"HR={hr_val} < 60 bpm",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.BRADYCARDIA,
            severity=severity,
            details={"hr_bpm": hr_val},
        ))
        max_severity = _higher_severity(max_severity, severity)

    # === RULE 4: Temperature > 38°C → MODERATE (fever) ===
    temp_mean = temp_features.get("mean")
    if temp_mean is not None and temp_mean > 38.0:
        severity = Severity.MODERATE
        rule = {
            "rule": "fever",
            "condition": f"Temp={temp_mean}°C > 38°C",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.FEVER,
            severity=severity,
            details={"temp_c": temp_mean},
        ))
        max_severity = _higher_severity(max_severity, severity)

    # === RULE 5: Frequent desaturations → apnea suspicion ===
    desat_index = spo2_features.get("desat_index_per_hr")
    if desat_index is not None and desat_index >= 5.0:
        severity = Severity.HIGH if desat_index >= 15 else Severity.MODERATE
        rule = {
            "rule": "apnea_suspicion",
            "condition": f"desat_index={desat_index}/hr >= 5",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.APNEA_SUSPICION,
            severity=severity,
            details={"desat_index_per_hr": desat_index},
        ))
        max_severity = _higher_severity(max_severity, severity)

    # === RULE 6: Low signal quality ===
    ecg_sqi = ecg_features.get("sqi", 1.0)
    ppg_sqi = ppg_features.get("sqi", 1.0)
    avg_sqi = (ecg_sqi + ppg_sqi) / 2.0
    if avg_sqi < 0.4:
        severity = Severity.MODERATE if avg_sqi < 0.2 else Severity.LOW
        rule = {
            "rule": "low_quality",
            "condition": f"avg_sqi={avg_sqi:.2f} < 0.4",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.LOW_QUALITY,
            severity=severity,
            details={"ecg_sqi": ecg_sqi, "ppg_sqi": ppg_sqi},
        ))
        max_severity = _higher_severity(max_severity, severity)

    # === RULE 7: Motion artifact ===
    motion_score = motion_features.get("score", 0.0)
    if motion_score > 0.5:
        severity = Severity.HIGH if motion_score > 0.8 else Severity.MODERATE
        rule = {
            "rule": "motion_artifact",
            "condition": f"motion_score={motion_score:.2f} > 0.5",
            "severity": severity,
        }
        triggered.append(rule)
        events.append(create_event(
            patient_id=patient_id, t_ms=t_ms,
            event_type=EventType.MOTION_ARTIFACT,
            severity=severity,
            details={"motion_score": motion_score},
        ))
        max_severity = _higher_severity(max_severity, severity)

    return {
        "triggered": [{"rule": r["rule"], "condition": r["condition"], "severity": r["severity"]} for r in triggered],
        "events": events,
        "severity": max_severity,
    }


# =============================================================================
# SEVERITY COMPARISON
# =============================================================================

_SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MODERATE: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def _higher_severity(a: str, b: str) -> str:
    return a if _SEVERITY_ORDER.get(a, 0) >= _SEVERITY_ORDER.get(b, 0) else b
