from __future__ import annotations

"""
Decision Engine — Smart Medical Mat
=======================================
Combines clinical rules + ML inference into a final decision (PDF §1.2f).

Severity levels (Table 6):
  LOW      → VERT  → surveillance standard
  MODERATE → JAUNE → vérification demandée
  HIGH     → ORANGE→ intervention recommandée
  CRITICAL → ROUGE → alerte immédiate
"""

from typing import Any, Dict, List

from common import SEVERITY_COLOR, Severity

_SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MODERATE: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}

_ACTION_MAP = {
    Severity.LOW: "surveillance_standard",
    Severity.MODERATE: "verification_demandee",
    Severity.HIGH: "intervention_recommandee",
    Severity.CRITICAL: "alerte_immediate",
}


def _higher_severity(a: str, b: str) -> str:
    return a if _SEVERITY_ORDER.get(a, 0) >= _SEVERITY_ORDER.get(b, 0) else b


def make_decision(
    rules_result: Dict[str, Any],
    ml_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fuse clinical rules and ML predictions into a final decision.

    Strategy:
      - Start with the highest severity from rules.
      - If ML risk_score > 0.7  → bump to at least HIGH.
      - If ML risk_score > 0.9  → bump to CRITICAL.
      - If ML deterioration_prob > 0.6 → bump to at least MODERATE.
      - ML event_class contributes context but rules always take priority.
    """
    # Start from rules severity
    severity = rules_result.get("severity", Severity.LOW)

    # ML risk score escalation
    risk_score = ml_result.get("risk_score")
    if risk_score is not None:
        if risk_score > 0.9:
            severity = _higher_severity(severity, Severity.CRITICAL)
        elif risk_score > 0.7:
            severity = _higher_severity(severity, Severity.HIGH)
        elif risk_score > 0.5:
            severity = _higher_severity(severity, Severity.MODERATE)

    # ML deterioration escalation
    deterioration_prob = ml_result.get("deterioration_prob")
    if deterioration_prob is not None and deterioration_prob > 0.6:
        severity = _higher_severity(severity, Severity.MODERATE)
        if deterioration_prob > 0.8:
            severity = _higher_severity(severity, Severity.HIGH)

    return {
        "severity": severity,
        "color": SEVERITY_COLOR.get(severity, "green"),
        "action": _ACTION_MAP.get(severity, "surveillance_standard"),
        "triggered_rules": [r["rule"] for r in rules_result.get("triggered", [])],
        "ml_event_class": ml_result.get("event_class", "normal"),
        "ml_risk_score": risk_score,
        "ml_deterioration_prob": deterioration_prob,
    }
