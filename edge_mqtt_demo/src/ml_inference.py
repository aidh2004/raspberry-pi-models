from __future__ import annotations

"""
ML Inference Module — Smart Medical Mat (Edge AI)
====================================================
Three lightweight models running on Raspberry Pi (PDF Table 5):

1. Logistic Regression → risk_score        (real-time, every window)
2. Random Forest       → event_class       (classification)
3. XGBoost             → deterioration_prob (trend-based prediction)

On first run: trains on synthetic clinical data and saves .joblib files.
On subsequent runs: loads pre-trained models instantly.

All models use the same feature vector extracted from feature_extraction.py.
"""

import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Feature names (order matters — must match build_feature_vector)
FEATURE_NAMES = [
    "hr_mean", "hr_min", "hr_max",
    "hrv_sdnn", "hrv_rmssd",
    "rr_mean_ms", "qrs_duration_ms",
    "ecg_sqi",
    "pulse_rate", "ppg_amplitude", "ppg_sqi",
    "spo2_mean", "spo2_min", "desat_count",
    "temp_mean", "temp_variation",
    "motion_score", "agitation_index",
    "resp_rate",
]

# Event classes for classification
EVENT_CLASSES = ["normal", "tachycardia", "bradycardia", "hypoxemia", "fever", "apnea_risk"]


def build_feature_vector(
    ecg: Dict[str, Any],
    ppg: Dict[str, Any],
    spo2: Dict[str, Any],
    temp: Dict[str, Any],
    motion: Dict[str, Any],
    respiration: Dict[str, Any],
) -> np.ndarray:
    """Build a flat feature vector from extracted features. Missing → 0."""
    def safe(d: Dict, key: str, default: float = 0.0) -> float:
        v = d.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    return np.array([
        safe(ecg, "hr_mean", 75),
        safe(ecg, "hr_min", 70),
        safe(ecg, "hr_max", 80),
        safe(ecg, "hrv_sdnn", 50),
        safe(ecg, "hrv_rmssd", 30),
        safe(ecg, "rr_mean_ms", 800),
        safe(ecg, "qrs_duration_ms", 100),
        safe(ecg, "sqi", 0.8),
        safe(ppg, "pulse_rate", 75),
        safe(ppg, "amplitude", 0.5),
        safe(ppg, "sqi", 0.8),
        safe(spo2, "mean", 97),
        safe(spo2, "min", 95),
        safe(spo2, "desaturation_count", 0),
        safe(temp, "mean", 37),
        safe(temp, "variation", 0.2),
        safe(motion, "score", 0),
        safe(motion, "agitation_index", 0),
        safe(respiration, "rate_bpm", 16),
    ], dtype=float)


# =============================================================================
# SYNTHETIC TRAINING DATA GENERATOR
# =============================================================================

def _generate_synthetic_data(n_samples: int = 2000, seed: int = 42):
    """Generate clinically plausible synthetic data for training."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=float)
    y_class = np.zeros(n_samples, dtype=int)  # event class index
    y_risk = np.zeros(n_samples, dtype=float)  # risk score 0–1

    for i in range(n_samples):
        # Randomly assign a clinical scenario
        scenario = rng.choice(EVENT_CLASSES, p=[0.50, 0.12, 0.10, 0.12, 0.08, 0.08])
        class_idx = EVENT_CLASSES.index(scenario)
        y_class[i] = class_idx

        if scenario == "normal":
            hr = rng.normal(75, 8)
            spo2 = rng.normal(97, 1.5)
            temp = rng.normal(36.8, 0.3)
            risk = rng.uniform(0.0, 0.2)
        elif scenario == "tachycardia":
            hr = rng.normal(130, 15)
            spo2 = rng.normal(96, 2)
            temp = rng.normal(37.2, 0.5)
            risk = rng.uniform(0.4, 0.8)
        elif scenario == "bradycardia":
            hr = rng.normal(42, 5)
            spo2 = rng.normal(95, 2)
            temp = rng.normal(36.5, 0.4)
            risk = rng.uniform(0.4, 0.8)
        elif scenario == "hypoxemia":
            hr = rng.normal(90, 15)
            spo2 = rng.normal(85, 4)
            temp = rng.normal(37, 0.5)
            risk = rng.uniform(0.6, 1.0)
        elif scenario == "fever":
            hr = rng.normal(95, 10)
            spo2 = rng.normal(96, 2)
            temp = rng.normal(39, 0.8)
            risk = rng.uniform(0.3, 0.7)
        else:  # apnea_risk
            hr = rng.normal(70, 10)
            spo2 = rng.normal(89, 4)
            temp = rng.normal(36.8, 0.3)
            risk = rng.uniform(0.5, 0.9)

        y_risk[i] = np.clip(risk, 0, 1)

        X[i] = [
            hr,                              # hr_mean
            hr - rng.uniform(5, 15),         # hr_min
            hr + rng.uniform(5, 15),         # hr_max
            rng.uniform(20, 80),             # hrv_sdnn
            rng.uniform(15, 60),             # hrv_rmssd
            60000.0 / max(hr, 30),           # rr_mean_ms
            rng.uniform(80, 120),            # qrs_duration_ms
            rng.uniform(0.5, 1.0),           # ecg_sqi
            hr + rng.normal(0, 3),           # pulse_rate
            rng.uniform(0.2, 1.0),           # ppg_amplitude
            rng.uniform(0.5, 1.0),           # ppg_sqi
            spo2,                            # spo2_mean
            spo2 - rng.uniform(0, 3),        # spo2_min
            max(0, int(rng.normal(1, 2))) if spo2 < 92 else 0,  # desat_count
            temp,                            # temp_mean
            rng.uniform(0.1, 1.0),           # temp_variation
            rng.uniform(0, 0.5),             # motion_score
            rng.uniform(0, 2),               # agitation_index
            rng.normal(16, 3),               # resp_rate
        ]

    return X, y_class, y_risk


# =============================================================================
# MODEL TRAINING (runs once, saves .joblib)
# =============================================================================

def _train_and_save_models(models_dir: str) -> Dict[str, Any]:
    """Train all 3 models on synthetic data and save to disk."""
    print("[ml] Training models on synthetic clinical data...")
    X, y_class, y_risk = _generate_synthetic_data()

    models = {}

    # 1. Logistic Regression → risk score (binary: low risk vs high risk)
    y_risk_binary = (y_risk > 0.4).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X, y_risk_binary)
    models["logistic_regression"] = lr
    print(f"[ml]   Logistic Regression trained (risk score) — accuracy={lr.score(X, y_risk_binary):.3f}")

    # 2. Random Forest → event classification
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    rf.fit(X, y_class)
    models["random_forest"] = rf
    print(f"[ml]   Random Forest trained (event class) — accuracy={rf.score(X, y_class):.3f}")

    # 3. XGBoost → deterioration probability
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=50, max_depth=5, learning_rate=0.1,
            use_label_binarizer=True, random_state=42,
            verbosity=0,
        )
        y_deterioration = (y_risk > 0.5).astype(int)
        xgb.fit(X, y_deterioration)
        models["xgboost"] = xgb
        print(f"[ml]   XGBoost trained (deterioration) — accuracy={xgb.score(X, y_deterioration):.3f}")
    else:
        print("[ml]   XGBoost not available — skipping deterioration model")

    # Save models
    os.makedirs(models_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(model, path)
        print(f"[ml]   Saved {path}")

    return models


# =============================================================================
# MODEL LOADER
# =============================================================================

class MLInferenceEngine:
    """Loads and runs all 3 ML models."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.available = False

        if not SKLEARN_AVAILABLE:
            print("[ml] WARNING: scikit-learn not installed — ML inference disabled")
            return
        if not JOBLIB_AVAILABLE:
            print("[ml] WARNING: joblib not installed — ML inference disabled")
            return

        self._load_or_train()
        self.available = True

    def _load_or_train(self) -> None:
        """Load existing models or train new ones."""
        model_files = {
            "logistic_regression": os.path.join(self.models_dir, "logistic_regression.joblib"),
            "random_forest": os.path.join(self.models_dir, "random_forest.joblib"),
        }
        if XGBOOST_AVAILABLE:
            model_files["xgboost"] = os.path.join(self.models_dir, "xgboost.joblib")

        # Check if all models exist
        all_exist = all(os.path.isfile(p) for p in model_files.values())

        if all_exist:
            for name, path in model_files.items():
                self.models[name] = joblib.load(path)
                print(f"[ml] Loaded {path}")
        else:
            self.models = _train_and_save_models(self.models_dir)

    def predict(
        self,
        ecg: Dict[str, Any],
        ppg: Dict[str, Any],
        spo2: Dict[str, Any],
        temp: Dict[str, Any],
        motion: Dict[str, Any],
        respiration: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all models and return unified ML output."""
        if not self.available:
            return {
                "risk_score": None,
                "deterioration_prob": None,
                "event_class": "normal",
            }

        X = build_feature_vector(ecg, ppg, spo2, temp, motion, respiration).reshape(1, -1)

        # 1. Logistic Regression → risk score (probability of high risk)
        lr = self.models.get("logistic_regression")
        risk_score = None
        if lr is not None:
            prob = lr.predict_proba(X)[0]
            risk_score = round(float(prob[1]) if prob.shape[0] > 1 else float(prob[0]), 3)

        # 2. Random Forest → event classification
        rf = self.models.get("random_forest")
        event_class = "normal"
        if rf is not None:
            pred = int(rf.predict(X)[0])
            if 0 <= pred < len(EVENT_CLASSES):
                event_class = EVENT_CLASSES[pred]

        # 3. XGBoost → deterioration probability
        xgb = self.models.get("xgboost")
        deterioration_prob = None
        if xgb is not None:
            prob = xgb.predict_proba(X)[0]
            deterioration_prob = round(float(prob[1]) if prob.shape[0] > 1 else float(prob[0]), 3)

        return {
            "risk_score": risk_score,
            "deterioration_prob": deterioration_prob,
            "event_class": event_class,
        }
