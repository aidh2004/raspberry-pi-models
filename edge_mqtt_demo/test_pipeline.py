"""
Integration test: runs the full Edge AI pipeline without MQTT.
Verifies: feature extraction → clinical rules → ML inference → decision engine.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from feature_extraction import (
    extract_ecg_features,
    extract_ppg_features,
    extract_spo2_features,
    extract_temp_features,
    extract_motion_features,
    extract_respiration_features,
)
from clinical_rules import evaluate_clinical_rules
from ml_inference import MLInferenceEngine
from decision_engine import make_decision


def generate_test_ecg(fs=250, duration=5, hr_bpm=75):
    t = np.arange(0, duration, 1.0 / fs)
    rr_sec = 60.0 / hr_bpm
    ecg = np.zeros_like(t)
    for beat_time in np.arange(0, duration, rr_sec):
        idx = int(beat_time * fs)
        if idx + 5 < len(ecg):
            ecg[idx:idx+2] = -0.1
            ecg[idx+2:idx+4] = 1.0
            ecg[idx+4:idx+5] = -0.3
    ecg += np.random.normal(0, 0.02, len(ecg))
    return ecg


def generate_test_ppg(fs=100, duration=5, hr_bpm=75):
    t = np.arange(0, duration, 1.0 / fs)
    rr_sec = 60.0 / hr_bpm
    ppg = 0.5 * np.sin(2 * np.pi * t / rr_sec) + np.random.normal(0, 0.01, len(t))
    return ppg


def generate_test_imu(fs=50, duration=5, still=True):
    n = int(fs * duration)
    if still:
        imu = np.column_stack([
            np.random.normal(0, 0.01, n),
            np.random.normal(0, 0.01, n),
            np.random.normal(1.0, 0.01, n),
        ])
    else:
        imu = np.column_stack([
            np.random.normal(0, 0.5, n),
            np.random.normal(0, 0.5, n),
            np.random.normal(1.0, 0.5, n),
        ])
    return imu


def test_scenario(name, ecg_hr, spo2_vals, temp_vals, moving=False):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")

    ecg_raw = generate_test_ecg(hr_bpm=ecg_hr)
    ppg_raw = generate_test_ppg(hr_bpm=ecg_hr)
    imu_raw = generate_test_imu(still=not moving)

    # Feature extraction
    motion_feat = extract_motion_features(imu_raw, 50)
    ecg_feat = extract_ecg_features(ecg_raw, 250, motion_score=motion_feat["score"])
    ecg_peaks = ecg_feat.pop("_peaks", np.array([]))
    ecg_filtered = ecg_feat.pop("_filtered", np.array([]))
    ppg_feat = extract_ppg_features(ppg_raw, 100, motion_score=motion_feat["score"],
                                     ecg_peaks=ecg_peaks, ecg_fs=250)
    spo2_feat, _ = extract_spo2_features(spo2_vals, 5.0, {})
    temp_feat = extract_temp_features(temp_vals)
    resp_feat = extract_respiration_features(ecg_filtered, 250)

    print(f"  ECG:  HR_mean={ecg_feat.get('hr_mean')} HRV_SDNN={ecg_feat.get('hrv_sdnn')} SQI={ecg_feat.get('sqi')}")
    print(f"  PPG:  pulse_rate={ppg_feat.get('pulse_rate')} PTT={ppg_feat.get('ptt_ms')} SQI={ppg_feat.get('sqi')}")
    print(f"  SpO2: mean={spo2_feat.get('mean')} min={spo2_feat.get('min')}")
    print(f"  Temp: mean={temp_feat.get('mean')} fever={temp_feat.get('fever')}")
    print(f"  Motion: score={motion_feat.get('score')} mvt_count={motion_feat.get('mvt_count')}")
    print(f"  Resp: rate={resp_feat.get('rate_bpm')}")

    # Clinical rules
    rules = evaluate_clinical_rules(
        patient_id="test", t_ms=0,
        ecg_features=ecg_feat, ppg_features=ppg_feat,
        spo2_features=spo2_feat, temp_features=temp_feat,
        motion_features=motion_feat, respiration_features=resp_feat,
    )
    print(f"  Rules: severity={rules['severity']} triggered={[r['rule'] for r in rules['triggered']]}")

    # ML inference
    ml = ml_engine.predict(ecg_feat, ppg_feat, spo2_feat, temp_feat, motion_feat, resp_feat)
    print(f"  ML: risk={ml['risk_score']} deterioration={ml['deterioration_prob']} class={ml['event_class']}")

    # Decision
    decision = make_decision(rules, ml)
    print(f"  DECISION: {decision['color'].upper()} → {decision['action']}")
    print(f"  Triggered rules: {decision.get('triggered_rules', [])}")

    return decision


if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    ml_engine = MLInferenceEngine(models_dir)

    # 1. Normal patient
    test_scenario("Normal patient",
                  ecg_hr=75,
                  spo2_vals=[97, 98, 96, 97, 98],
                  temp_vals=[36.8, 36.9, 36.7])

    # 2. Tachycardia
    test_scenario("Tachycardia (HR=130)",
                  ecg_hr=130,
                  spo2_vals=[96, 97, 95],
                  temp_vals=[37.1, 37.0])

    # 3. Bradycardia
    test_scenario("Bradycardia (HR=40)",
                  ecg_hr=40,
                  spo2_vals=[96, 95, 97],
                  temp_vals=[36.5, 36.6])

    # 4. Hypoxemia
    test_scenario("Hypoxemia (SpO2=85%)",
                  ecg_hr=90,
                  spo2_vals=[85, 87, 83, 86],
                  temp_vals=[37.0])

    # 5. Fever
    test_scenario("Fever (39°C)",
                  ecg_hr=95,
                  spo2_vals=[96, 97],
                  temp_vals=[39.2, 39.0, 38.8])

    # 6. Multi-alarm: hypoxemia + tachycardia + fever
    test_scenario("Multi-alarm: hypoxemia + tachy + fever",
                  ecg_hr=135,
                  spo2_vals=[84, 82, 86],
                  temp_vals=[39.5, 39.3])

    print(f"\n{'='*60}")
    print("ALL SCENARIOS COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
