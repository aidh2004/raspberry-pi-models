from __future__ import annotations

"""
Edge AI Patient Monitor — Single-Window Dashboard
====================================================
ONE optimised matplotlib window with 3 switchable pages:
  Page 1 — Raw Inputs:        ECG, PPG, SpO2, Temperature, IMU 3-axis, IMU |a|
  Page 2 — Processed Signals: Filtered ECG+R-peaks, Filtered PPG+peaks,
                               SpO2 trend, Temperature trend, Motion score, Respiration
  Page 3 — AI Features:       Vital-sign banner, HR trend, HRV, SQI,
                               SpO2+Temp, ML predictions, Decision/Rules

Performance optimisations:
  - Single figure → low GPU / memory usage
  - Only the active page is redrawn each tick
  - Update interval 600 ms (vs 500 for multiple windows)
  - Axes are cleared + redrawn only when visible
  - Single MQTT client for all topics
"""

import argparse
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

from common import (
    DEFAULT_DEVICE_ID,
    MqttConfig,
    get_input_topics,
    get_output_topics,
    parse_json,
    parse_topic,
    safe_float_list,
    safe_imu_triplet_list,
    validate_chunk_message,
    validate_imu_chunk_message,
    validate_sample_message,
)

try:
    from scipy.signal import butter, detrend, filtfilt, find_peaks, iirnotch
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ═════════════════════════════════════════════════════════════════════════════
# Signal processing (lightweight copies from feature_extraction)
# ═════════════════════════════════════════════════════════════════════════════

def _bp(x, fs, lo, hi):
    nyq = fs / 2.0
    b, a = butter(3, [lo / nyq, min(hi / nyq, 0.99)], btype="band")
    return filtfilt(b, a, x)

def preprocess_ecg(raw, fs):
    x = raw.astype(float)
    if SCIPY_OK:
        x = detrend(x, type="linear")
        x = _bp(x, fs, 0.5, 40.0)
        if 50 < fs / 2:
            b, a = iirnotch(50.0 / (fs / 2), 30)
            x = filtfilt(b, a, x)
    else:
        x -= np.mean(x)
    return x

def preprocess_ppg(raw, fs):
    x = raw.astype(float)
    if SCIPY_OK:
        x = detrend(x, type="linear")
        x = _bp(x, fs, 0.5, 8.0)
    else:
        x -= np.mean(x)
    return x

def detect_peaks(sig, fs, min_dist_s=0.25):
    if sig.size < fs:
        return np.array([], dtype=int)
    s = float(np.std(sig))
    if s < 1e-6:
        return np.array([], dtype=int)
    if SCIPY_OK:
        p, _ = find_peaks(sig, distance=int(min_dist_s * fs),
                          prominence=max(0.35 * s, 0.05))
        return p.astype(int)
    return np.array([], dtype=int)

def rate_bpm(peaks, fs):
    if len(peaks) < 2:
        return None
    ipi = np.diff(peaks) / float(fs)
    ipi = ipi[(ipi > 0.3) & (ipi < 2.0)]
    return 60.0 / np.mean(ipi) if len(ipi) else None


# ═════════════════════════════════════════════════════════════════════════════
# Shared sensor buffers (thread-safe)
# ═════════════════════════════════════════════════════════════════════════════

class Buffers:
    def __init__(self, win_sec=10.0):
        self.lock = threading.Lock()
        self.ecg_fs, self.ppg_fs, self.imu_fs = 250, 100, 50
        self.win_sec = win_sec
        # waveforms
        self._ecg_r: List[float] = []
        self._ppg_r: List[float] = []
        self._imu_x: List[float] = []
        self._imu_y: List[float] = []
        self._imu_z: List[float] = []
        # processed
        self._ecg_f: List[float] = []
        self._ecg_pk: List[int] = []
        self._ppg_f: List[float] = []
        self._ppg_pk: List[int] = []
        # scalars
        self.spo2_t: Deque[float] = deque(maxlen=300)
        self.spo2_v: Deque[float] = deque(maxlen=300)
        self.temp_t: Deque[float] = deque(maxlen=300)
        self.temp_v: Deque[float] = deque(maxlen=300)
        self._t0: Optional[float] = None
        # features
        self.feat_hist: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.feat_last: Optional[Dict[str, Any]] = None

    def _mx(self, fs): return int(self.win_sec * fs)
    def _rt(self, t_ms):
        if self._t0 is None: self._t0 = t_ms / 1000.0
        return t_ms / 1000.0 - self._t0

    def add_ecg(self, s, fs):
        with self.lock:
            self.ecg_fs = fs; self._ecg_r.extend(s)
            mx = self._mx(fs)
            if len(self._ecg_r) > mx: self._ecg_r[:] = self._ecg_r[-mx:]
            if len(self._ecg_r) >= fs:
                a = np.array(self._ecg_r, dtype=float)
                f = preprocess_ecg(a, fs)
                self._ecg_f[:] = f.tolist()
                self._ecg_pk[:] = detect_peaks(f, fs, 0.25).tolist()

    def add_ppg(self, s, fs):
        with self.lock:
            self.ppg_fs = fs; self._ppg_r.extend(s)
            mx = self._mx(fs)
            if len(self._ppg_r) > mx: self._ppg_r[:] = self._ppg_r[-mx:]
            if len(self._ppg_r) >= fs:
                a = np.array(self._ppg_r, dtype=float)
                f = preprocess_ppg(a, fs)
                self._ppg_f[:] = f.tolist()
                self._ppg_pk[:] = detect_peaks(f, fs, 0.4).tolist()

    def add_imu(self, trips, fs):
        with self.lock:
            self.imu_fs = fs
            for x, y, z in trips:
                self._imu_x.append(x); self._imu_y.append(y); self._imu_z.append(z)
            mx = self._mx(fs)
            if len(self._imu_x) > mx:
                self._imu_x[:] = self._imu_x[-mx:]
                self._imu_y[:] = self._imu_y[-mx:]
                self._imu_z[:] = self._imu_z[-mx:]

    def add_spo2(self, t_ms, v):
        with self.lock: self.spo2_t.append(self._rt(t_ms)); self.spo2_v.append(v)
    def add_temp(self, t_ms, v):
        with self.lock: self.temp_t.append(self._rt(t_ms)); self.temp_v.append(v)
    def add_feat(self, p):
        with self.lock: self.feat_hist.append(p); self.feat_last = p

    # snapshots
    def sn_ecg(self):
        with self.lock: return list(self._ecg_r), list(self._ecg_f), list(self._ecg_pk), self.ecg_fs
    def sn_ppg(self):
        with self.lock: return list(self._ppg_r), list(self._ppg_f), list(self._ppg_pk), self.ppg_fs
    def sn_imu(self):
        with self.lock: return list(self._imu_x), list(self._imu_y), list(self._imu_z), self.imu_fs
    def sn_spo2(self):
        with self.lock: return list(self.spo2_t), list(self.spo2_v)
    def sn_temp(self):
        with self.lock: return list(self.temp_t), list(self.temp_v)
    def sn_feat(self):
        with self.lock: return list(self.feat_hist), (dict(self.feat_last) if self.feat_last else None)


# ═════════════════════════════════════════════════════════════════════════════
# Single-window paged dashboard
# ═════════════════════════════════════════════════════════════════════════════

PAGE_NAMES = ["1 ▸ Raw Inputs", "2 ▸ Processed", "3 ▸ AI Features"]
DEC_COL = {"green": "#2ecc71", "yellow": "#f1c40f", "orange": "#e67e22", "red": "#e74c3c"}


class Dashboard:
    def __init__(self, buf: Buffers):
        self.buf = buf
        self.page = 0  # 0, 1, 2

        # Single figure
        self.fig = plt.figure(figsize=(19, 11), num="Edge AI Patient Monitor")
        self.fig.patch.set_facecolor("#f8f9fa")

        # Reserve bottom strip for buttons
        self.btn_axes = []
        self.buttons = []
        bw = 0.22
        for i, name in enumerate(PAGE_NAMES):
            ax = self.fig.add_axes([0.05 + i * (bw + 0.03), 0.01, bw, 0.045])
            btn = Button(ax, name, color="#dfe6e9", hovercolor="#74b9ff")
            btn.label.set_fontsize(10)
            btn.label.set_fontweight("bold")
            btn.on_clicked(self._make_switch(i))
            self.btn_axes.append(ax)
            self.buttons.append(btn)

        # 3x2 grid of axes for content (reused across pages)
        self.gs = GridSpec(3, 2, figure=self.fig,
                           left=0.06, right=0.97, top=0.93, bottom=0.09,
                           hspace=0.38, wspace=0.28)
        self.axes = [[self.fig.add_subplot(self.gs[r, c]) for c in range(2)] for r in range(3)]

        # Header
        self.title = self.fig.suptitle("", fontsize=14, fontweight="bold",
                                        color="#2c3e50", y=0.97)
        self._highlight_btn(0)

    def _make_switch(self, idx):
        def cb(_event):
            self.page = idx
            self._highlight_btn(idx)
        return cb

    def _highlight_btn(self, idx):
        for i, btn in enumerate(self.buttons):
            if i == idx:
                btn.ax.set_facecolor("#74b9ff")
                btn.color = "#74b9ff"
            else:
                btn.ax.set_facecolor("#dfe6e9")
                btn.color = "#dfe6e9"

    def _ax(self, r, c):
        return self.axes[r][c]

    # ── Page 1: raw inputs ──────────────────────────────────────────

    def _draw_raw(self):
        self.title.set_text("Page 1 — Raw Sensor Inputs (before preprocessing)")

        # ECG raw
        ax = self._ax(0, 0); ax.clear()
        raw, _, _, fs = self.buf.sn_ecg()
        if len(raw) > 10:
            t = np.arange(len(raw)) / fs
            ax.plot(t, raw, "#2980b9", lw=0.6)
        ax.set_title("ECG Raw (250 Hz)", fontsize=10, fontweight="bold")
        ax.set_ylabel("mV"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)

        # PPG raw
        ax = self._ax(0, 1); ax.clear()
        raw, _, _, fs = self.buf.sn_ppg()
        if len(raw) > 10:
            t = np.arange(len(raw)) / fs
            ax.plot(t, raw, "#e74c3c", lw=0.6)
        ax.set_title("PPG Raw (100 Hz)", fontsize=10, fontweight="bold")
        ax.set_ylabel("a.u."); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)

        # SpO2 raw
        ax = self._ax(1, 0); ax.clear()
        st, sv = self.buf.sn_spo2()
        if sv:
            ax.plot(st, sv, "#8e44ad", lw=1.5, marker="o", ms=3)
            ax.text(0.97, 0.92, f"{sv[-1]:.1f}%", transform=ax.transAxes,
                    fontsize=13, ha="right", va="top", fontweight="bold",
                    bbox=dict(boxstyle="round", fc="#ebdef0", alpha=0.9))
        ax.set_title("SpO2 Raw (1 Hz)", fontsize=10, fontweight="bold")
        ax.set_ylabel("SpO2 (%)"); ax.set_xlabel("Time (s)")
        ax.set_ylim(80, 102); ax.grid(True, alpha=0.3)

        # Temperature raw
        ax = self._ax(1, 1); ax.clear()
        tt, tv = self.buf.sn_temp()
        if tv:
            ax.plot(tt, tv, "#e67e22", lw=2, marker="s", ms=4)
            ax.text(0.97, 0.92, f"{tv[-1]:.1f}°C", transform=ax.transAxes,
                    fontsize=13, ha="right", va="top", fontweight="bold",
                    bbox=dict(boxstyle="round", fc="#fdebd0", alpha=0.9))
        ax.set_title("Temperature Raw (0.2 Hz)", fontsize=10, fontweight="bold")
        ax.set_ylabel("°C"); ax.set_xlabel("Time (s)")
        ax.set_ylim(34, 42); ax.grid(True, alpha=0.3)

        # IMU 3-axis
        ax = self._ax(2, 0); ax.clear()
        ix, iy, iz, ifs = self.buf.sn_imu()
        if len(ix) > 10:
            t = np.arange(len(ix)) / ifs
            ax.plot(t, ix, "r", lw=0.7, alpha=0.8, label="Ax")
            ax.plot(t, iy, "g", lw=0.7, alpha=0.8, label="Ay")
            ax.plot(t, iz, "b", lw=0.7, alpha=0.8, label="Az")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_title("IMU Accelerometer (50 Hz)", fontsize=10, fontweight="bold")
        ax.set_ylabel("g"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)

        # IMU magnitude
        ax = self._ax(2, 1); ax.clear()
        if len(ix) > 10:
            mag = np.sqrt(np.array(ix)**2 + np.array(iy)**2 + np.array(iz)**2)
            t = np.arange(len(mag)) / ifs
            ax.plot(t, mag, "#16a085", lw=0.8)
            ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="1g")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_title("IMU |a| Magnitude", fontsize=10, fontweight="bold")
        ax.set_ylabel("|a| (g)"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)

    # ── Page 2: processed signals ───────────────────────────────────

    def _draw_processed(self):
        self.title.set_text("Page 2 — Preprocessed Signals (after filtering & detection)")

        # Filtered ECG + R-peaks
        ax = self._ax(0, 0); ax.clear()
        _, filt, pks, fs = self.buf.sn_ecg()
        if len(filt) > 10:
            t = np.arange(len(filt)) / fs
            ax.plot(t, filt, "#27ae60", lw=0.8)
            pk = np.array(pks, dtype=int)
            v = pk[pk < len(filt)]
            if len(v):
                fa = np.array(filt)
                ax.plot(t[v], fa[v], "ro", ms=5, label=f"R-peaks ({len(v)})")
                hr = rate_bpm(v, fs)
                if hr:
                    ax.text(0.97, 0.92, f"HR: {hr:.0f} bpm", transform=ax.transAxes,
                            fontsize=12, ha="right", va="top", fontweight="bold",
                            bbox=dict(boxstyle="round", fc="#d5f5e3", alpha=0.9))
                ax.legend(loc="upper left", fontsize=7)
        ax.set_title("ECG Filtered (BP 0.5-40Hz + Notch 50Hz) + R-peaks",
                      fontsize=10, fontweight="bold")
        ax.set_ylabel("Amp"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)

        # Filtered PPG + peaks
        ax = self._ax(0, 1); ax.clear()
        _, filt, pks, fs = self.buf.sn_ppg()
        if len(filt) > 10:
            t = np.arange(len(filt)) / fs
            ax.plot(t, filt, "#e67e22", lw=0.8)
            pk = np.array(pks, dtype=int)
            v = pk[pk < len(filt)]
            if len(v):
                fa = np.array(filt)
                ax.plot(t[v], fa[v], "ro", ms=5, label=f"Peaks ({len(v)})")
                pr = rate_bpm(v, fs)
                if pr:
                    ax.text(0.97, 0.92, f"Pulse: {pr:.0f} bpm", transform=ax.transAxes,
                            fontsize=12, ha="right", va="top", fontweight="bold",
                            bbox=dict(boxstyle="round", fc="#fdebd0", alpha=0.9))
                ax.legend(loc="upper left", fontsize=7)
        ax.set_title("PPG Filtered (BP 0.5-8Hz) + Peaks",
                      fontsize=10, fontweight="bold")
        ax.set_ylabel("Amp"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)

        # SpO2 trend
        ax = self._ax(1, 0); ax.clear()
        st, sv = self.buf.sn_spo2()
        if sv:
            ax.plot(st, sv, "#8e44ad", lw=1.5, marker="o", ms=3)
            m = np.mean(sv)
            ax.axhline(m, color="#3498db", ls="-", alpha=0.4, label=f"Mean ({m:.1f}%)")
            ax.text(0.97, 0.92, f"Mean: {m:.1f}%\nMin: {min(sv):.1f}%",
                    transform=ax.transAxes, fontsize=10, ha="right", va="top",
                    fontweight="bold", bbox=dict(boxstyle="round", fc="#ebdef0", alpha=0.9))
        ax.axhline(90, color="red", ls="--", alpha=0.5, label="Critical (90%)")
        ax.axhline(94, color="orange", ls="--", alpha=0.3, label="Warning (94%)")
        ax.set_title("SpO2 Trend + Desaturation", fontsize=10, fontweight="bold")
        ax.set_ylabel("SpO2 (%)"); ax.set_xlabel("Time (s)")
        ax.set_ylim(80, 102); ax.legend(loc="lower right", fontsize=7); ax.grid(True, alpha=0.3)

        # Temperature trend
        ax = self._ax(1, 1); ax.clear()
        tt, tv = self.buf.sn_temp()
        if tv:
            ax.plot(tt, tv, "#e67e22", lw=2, marker="s", ms=4)
            last = tv[-1]
            st_txt = "NORMAL" if last < 38 else ("FEVER" if last < 39.5 else "HIGH FEVER")
            cl = "#27ae60" if last < 38 else ("#e67e22" if last < 39.5 else "#e74c3c")
            ax.text(0.97, 0.92, f"{last:.1f}°C  {st_txt}", transform=ax.transAxes,
                    fontsize=11, ha="right", va="top", fontweight="bold", color=cl,
                    bbox=dict(boxstyle="round", fc="#fef9e7", alpha=0.9))
        ax.axhline(38, color="red", ls="--", alpha=0.5, label="Fever")
        ax.axhline(36, color="blue", ls="--", alpha=0.3, label="Hypothermia")
        ax.set_title("Temperature Trend + Fever Detection", fontsize=10, fontweight="bold")
        ax.set_ylabel("°C"); ax.set_xlabel("Time (s)")
        ax.set_ylim(34, 42); ax.legend(loc="lower right", fontsize=7); ax.grid(True, alpha=0.3)

        # Motion score
        ax = self._ax(2, 0); ax.clear()
        ix, iy, iz, ifs = self.buf.sn_imu()
        if len(ix) > 10:
            mag = np.sqrt(np.array(ix)**2 + np.array(iy)**2 + np.array(iz)**2)
            mot = np.abs(mag - 1.0)
            t = np.arange(len(mot)) / ifs
            w = max(1, int(0.5 * ifs))
            sm = np.convolve(mot, np.ones(w)/w, "same") if len(mot) >= w else mot
            ax.fill_between(t, 0, sm, alpha=0.3, color="#e67e22")
            ax.plot(t, sm, "#e67e22", lw=1.2)
            sc = float(np.mean(mot[-min(len(mot), ifs):]))
            ax.text(0.97, 0.92, f"Motion: {sc:.3f}g", transform=ax.transAxes,
                    fontsize=11, ha="right", va="top", fontweight="bold",
                    bbox=dict(boxstyle="round", fc="#fdebd0", alpha=0.9))
            ax.axhline(0.5, color="red", ls="--", alpha=0.4, label="High motion")
        ax.set_title("IMU Motion Score (|a| − 1g)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Motion (g)"); ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left", fontsize=7); ax.grid(True, alpha=0.3)

        # Respiration rate
        ax = self._ax(2, 1); ax.clear()
        hist, _ = self.buf.sn_feat()
        if hist:
            rv = [f.get("respiration", {}).get("rate_bpm") for f in hist]
            vx = [i for i, v in enumerate(rv) if v is not None]
            vy = [v for v in rv if v is not None]
            if vy:
                ax.plot(vx, vy, "#16a085", lw=2, marker="o", ms=3)
                ax.text(0.97, 0.92, f"Resp: {vy[-1]:.0f} bpm", transform=ax.transAxes,
                        fontsize=11, ha="right", va="top", fontweight="bold",
                        bbox=dict(boxstyle="round", fc="#d1f2eb", alpha=0.9))
        ax.axhline(12, color="orange", ls="--", alpha=0.4, label="Low (12)")
        ax.axhline(20, color="orange", ls="--", alpha=0.4, label="High (20)")
        ax.set_title("Respiration Rate", fontsize=10, fontweight="bold")
        ax.set_ylabel("breaths/min"); ax.set_xlabel("Window #")
        ax.set_ylim(0, 40); ax.legend(loc="upper left", fontsize=7); ax.grid(True, alpha=0.3)

    # ── Page 3: AI features ─────────────────────────────────────────

    @staticmethod
    def _fs(hist, fn):
        vals = [fn(f) for f in hist]
        x = [i for i, v in enumerate(vals) if v is not None]
        y = [v for v in vals if v is not None]
        return x, y

    def _draw_features(self):
        self.title.set_text("Page 3 — Extracted Features & AI Pipeline Output")
        hist, lat = self.buf.sn_feat()
        if not lat:
            for r in range(3):
                for c in range(2):
                    ax = self._ax(r, c); ax.clear(); ax.axis("off")
            self._ax(1, 0).text(0.5, 0.5, "Waiting for edge features...",
                                 transform=self._ax(1, 0).transAxes,
                                 fontsize=14, ha="center", va="center", color="#aaa")
            return

        ecg = lat.get("ecg", {}); ppg = lat.get("ppg", {})
        spo2 = lat.get("spo2", {}); temp = lat.get("temp_c", {})
        motion = lat.get("motion", {}); resp = lat.get("respiration", {})
        ml = lat.get("ml", {}); dec = lat.get("decision", {}); rules = lat.get("rules", {})

        hr_e = ecg.get("hr_mean") or ecg.get("hr_bpm")
        hr_p = ppg.get("pulse_rate") or ppg.get("hr_bpm")
        sp = spo2.get("mean"); tp = temp.get("mean")
        rr = resp.get("rate_bpm"); mo = motion.get("score", 0)
        sq_e = ecg.get("sqi", 0); sq_p = ppg.get("sqi", 0)

        # ── Row 0, left: vital signs banner ──
        ax = self._ax(0, 0); ax.clear(); ax.axis("off")
        vitals = [
            ("HR-ECG", f"{hr_e:.0f}" if hr_e else "--", "bpm", "#3498db"),
            ("HR-PPG", f"{hr_p:.0f}" if hr_p else "--", "bpm", "#e74c3c"),
            ("SpO2", f"{sp:.1f}" if sp else "--", "%", "#8e44ad"),
            ("Temp", f"{tp:.1f}" if tp else "--", "°C", "#e67e22"),
        ]
        bw = 0.9 / len(vitals)
        for i, (lb, vl, un, co) in enumerate(vitals):
            x0 = 0.05 + i * bw
            ax.add_patch(plt.Rectangle((x0, 0.1), bw*0.88, 0.8,
                         transform=ax.transAxes, fc=co, alpha=0.12, ec=co, lw=2))
            ax.text(x0+bw*0.44, 0.65, vl, transform=ax.transAxes,
                    fontsize=20, fontweight="bold", ha="center", va="center", color=co)
            ax.text(x0+bw*0.44, 0.28, f"{lb} ({un})", transform=ax.transAxes,
                    fontsize=8, ha="center", va="center", color="#666")

        # ── Row 0, right: more vitals ──
        ax = self._ax(0, 1); ax.clear(); ax.axis("off")
        vitals2 = [
            ("Resp", f"{rr:.0f}" if rr else "--", "bpm", "#16a085"),
            ("Motion", f"{mo:.2f}", "g", "#f39c12"),
            ("SQI-E", f"{sq_e:.2f}", "", "#2980b9"),
            ("SQI-P", f"{sq_p:.2f}", "", "#c0392b"),
        ]
        bw = 0.9 / len(vitals2)
        for i, (lb, vl, un, co) in enumerate(vitals2):
            x0 = 0.05 + i * bw
            ax.add_patch(plt.Rectangle((x0, 0.1), bw*0.88, 0.8,
                         transform=ax.transAxes, fc=co, alpha=0.12, ec=co, lw=2))
            ax.text(x0+bw*0.44, 0.65, vl, transform=ax.transAxes,
                    fontsize=20, fontweight="bold", ha="center", va="center", color=co)
            ax.text(x0+bw*0.44, 0.28, f"{lb} ({un})" if un else lb, transform=ax.transAxes,
                    fontsize=8, ha="center", va="center", color="#666")

        # ── Row 1, left: HR + HRV trend ──
        ax = self._ax(1, 0); ax.clear()
        ex, ey = self._fs(hist, lambda f: f.get("ecg",{}).get("hr_mean") or f.get("ecg",{}).get("hr_bpm"))
        px, py = self._fs(hist, lambda f: f.get("ppg",{}).get("pulse_rate") or f.get("ppg",{}).get("hr_bpm"))
        if ey: ax.plot(ex, ey, "#3498db", lw=2, marker="o", ms=2, label=f"ECG ({ey[-1]:.0f})")
        if py: ax.plot(px, py, "#e74c3c", lw=2, marker="s", ms=2, label=f"PPG ({py[-1]:.0f})")
        ax.axhline(120, color="red", ls="--", alpha=0.3); ax.axhline(60, color="blue", ls="--", alpha=0.2)
        ax.set_title("Heart Rate Trend", fontsize=10, fontweight="bold")
        ax.set_ylabel("bpm"); ax.set_xlabel("Window #"); ax.set_ylim(40, 160)
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.3)

        # ── Row 1, right: HRV ──
        ax = self._ax(1, 1); ax.clear()
        sx, sy = self._fs(hist, lambda f: f.get("ecg",{}).get("hrv_sdnn"))
        rx, ry = self._fs(hist, lambda f: f.get("ecg",{}).get("hrv_rmssd"))
        if sy: ax.plot(sx, sy, "purple", lw=2, marker="o", ms=2, label=f"SDNN ({sy[-1]:.1f})")
        if ry: ax.plot(rx, ry, "teal", lw=2, marker="s", ms=2, label=f"RMSSD ({ry[-1]:.1f})")
        ax.set_title("HRV (Heart Rate Variability)", fontsize=10, fontweight="bold")
        ax.set_ylabel("ms"); ax.set_xlabel("Window #")
        mx_h = max(sy + ry, default=50) * 1.3
        ax.set_ylim(0, max(200, mx_h))
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.3)

        # ── Row 2, left: ML Predictions ──
        ax = self._ax(2, 0); ax.clear()
        rkx, rky = self._fs(hist, lambda f: f.get("ml",{}).get("risk_score"))
        dx, dy = self._fs(hist, lambda f: f.get("ml",{}).get("deterioration_prob"))
        if rky: ax.plot(rkx, rky, "#e74c3c", lw=2, marker="o", ms=2, label=f"Risk ({rky[-1]:.2f})")
        if dy: ax.plot(dx, dy, "#9b59b6", lw=2, marker="s", ms=2, label=f"Deter. ({dy[-1]:.2f})")
        ax.axhline(0.7, color="red", ls="--", alpha=0.3); ax.axhline(0.5, color="orange", ls="--", alpha=0.2)
        mc = ml.get("event_class", "")
        if mc:
            ax.text(0.02, 0.92, f"Class: {mc}", transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="top",
                    bbox=dict(boxstyle="round", fc="#fef9e7", alpha=0.9))
        ax.set_title("ML Predictions (LogReg + RF + XGB)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Prob (0-1)"); ax.set_xlabel("Window #"); ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.3)

        # ── Row 2, right: Decision ──
        ax = self._ax(2, 1); ax.clear(); ax.axis("off")
        cn = dec.get("color", "green"); sev = dec.get("severity", "")
        act = dec.get("action", ""); dc = DEC_COL.get(cn, "#95a5a6")

        ax.add_patch(plt.Rectangle((0.05, 0.5), 0.9, 0.45, transform=ax.transAxes,
                     fc=dc, alpha=0.22, ec=dc, lw=3))
        ax.text(0.5, 0.78, f"{cn.upper()} — {sev.upper()}", transform=ax.transAxes,
                fontsize=15, fontweight="bold", ha="center", va="center", color=dc)
        ax.text(0.5, 0.55, act.replace("_", " ").title(), transform=ax.transAxes,
                fontsize=9, ha="center", va="center", style="italic", color="#555")

        trig = rules.get("triggered", [])
        rt = ("Rules: " + ", ".join(trig)) if trig else "No clinical rules triggered"
        ax.text(0.5, 0.32, rt, transform=ax.transAxes, fontsize=8,
                ha="center", va="center",
                bbox=dict(boxstyle="round", fc="#eaecee", alpha=0.8))

        extras = []
        q = ecg.get("qrs_width_ms")
        if q is not None: extras.append(f"QRS:{q:.0f}ms")
        s1 = ecg.get("hrv_sdnn")
        if s1 is not None: extras.append(f"SDNN:{s1:.1f}")
        s2 = ecg.get("hrv_rmssd")
        if s2 is not None: extras.append(f"RMSSD:{s2:.1f}")
        di = spo2.get("desat_index")
        if di is not None: extras.append(f"Desat:{di:.2f}")
        if extras:
            ax.text(0.5, 0.1, " | ".join(extras), transform=ax.transAxes,
                    fontsize=8, ha="center", va="center", color="#7f8c8d", family="monospace")
        ax.set_title("Decision & Clinical Rules", fontsize=10, fontweight="bold")

    # ── Animation callback ──────────────────────────────────────────

    def update(self, _frame):
        if self.page == 0:
            self._draw_raw()
        elif self.page == 1:
            self._draw_processed()
        else:
            self._draw_features()


# ═════════════════════════════════════════════════════════════════════════════
# MQTT + main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Edge AI Patient Monitor — single window")
    ap.add_argument("--broker-host", default="localhost")
    ap.add_argument("--broker-port", type=int, default=1883)
    ap.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    ap.add_argument("--window-sec", type=float, default=10.0)
    args = ap.parse_args()

    buf = Buffers(win_sec=args.window_sec)

    inp = get_input_topics(args.device_id)
    out = get_output_topics(args.device_id)
    topics = [inp["ecg"], inp["ppg"], inp["imu"], inp["spo2"], inp["temp"], out["features"]]

    cfg = MqttConfig(host=args.broker_host, port=args.broker_port)
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                         client_id="edge-ai-monitor")

    def on_connect(c, _u, _f, rc, _p):
        if rc == 0:
            print(f"[monitor] connected to {cfg.host}:{cfg.port}")
            for t in topics:
                c.subscribe(t, qos=1)
                print(f"[monitor]   -> {t}")
        else:
            print(f"[monitor] connect failed rc={rc}")

    def on_disconnect(_c, _u, _f, rc, _p):
        print(f"[monitor] disconnected rc={rc}")

    def on_message(_c, _u, msg):
        try:
            p = parse_json(msg.payload)
        except Exception:
            return
        _, _, s = parse_topic(msg.topic)
        if s == "ecg" and not validate_chunk_message(p):
            buf.add_ecg(safe_float_list(p["samples"]), int(p["fs_hz"]))
        elif s == "ppg" and not validate_chunk_message(p):
            buf.add_ppg(safe_float_list(p["samples"]), int(p["fs_hz"]))
        elif s == "imu" and not validate_imu_chunk_message(p):
            buf.add_imu(safe_imu_triplet_list(p["samples"]), int(p["fs_hz"]))
        elif s == "spo2" and not validate_sample_message(p):
            buf.add_spo2(float(p["t_ms"]), float(p["value"]))
        elif s == "temp" and not validate_sample_message(p):
            buf.add_temp(float(p["t_ms"]), float(p["value"]))
        elif s == "features":
            buf.add_feat(p)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    print(f"[monitor] connecting to {cfg.host}:{cfg.port}...")
    while True:
        try:
            client.connect(cfg.host, cfg.port, cfg.keepalive)
            break
        except OSError as ex:
            print(f"[monitor] broker unreachable ({ex}); retry in 2s...")
            time.sleep(2)

    client.loop_start()

    print("[monitor] opening single-window dashboard...")
    print("[monitor] use buttons at bottom to switch pages:")
    print("[monitor]   Page 1: Raw Inputs  |  Page 2: Processed  |  Page 3: AI Features")
    plt.ion()

    dash = Dashboard(buf)
    ani = FuncAnimation(dash.fig, dash.update, interval=600, cache_frame_data=False)

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\n[monitor] stopping")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
