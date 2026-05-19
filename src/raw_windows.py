"""Raw windowed signals for the 1D-CNN.

Produces (3, 7500) float32 arrays per window for both local and WESAD data:
  channel 0: filtered ECG, 250 Hz
  channel 1: filtered Resp, upsampled 25 -> 250 Hz (linear)
  channel 2: skin temperature, upsampled to 250 Hz (linear)

Window size: 30 s @ 250 Hz = 7500 samples, 50% overlap (matches the existing
classical pipeline so LORO folds are directly comparable).

Per-recording robust z-score is applied per channel BEFORE windowing — the
single most important step for cross-device transfer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .features import OVERLAP, WINDOW_S, preprocess_recording
from .io import (
    Recording,
    list_recordings,
    load_recording,
    phase_boundaries,
    resample_uniform,
)
from .pipeline import assign_activity
from .preprocess import filter_br, filter_ecg
from .wesad_io import WESAD_LABEL_TO_ACTIVITY, label_runs, list_subjects, load_subject

TARGET_FS = 250.0
N_CHANNELS = 3       # ECG, Resp, Temp
ACTIVITY_TO_LABEL = {"baseline": 0, "meditation": 1, "stress": 2}
LABEL_TO_ACTIVITY = {v: k for k, v in ACTIVITY_TO_LABEL.items()}
WINDOW_N = int(WINDOW_S * TARGET_FS)


@dataclass
class RawWindow:
    signal: np.ndarray       # (3, WINDOW_N) float32
    label: int               # 0=baseline, 1=meditation, 2=stress
    activity: str
    source: str              # 'local' | 'wesad'
    subject: str             # e.g. 'mta', 'nvt', 'S7'
    rec_name: str            # e.g. 'mta-5-8-medi' or 'WESAD_S7_lbl2_run420000'
    t_start: float           # absolute seconds within the source recording


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Std-based z-score with outlier clipping at ±8 sigma.

    We use std (not MAD) here because the goal is CNN-input scaling, not
    QRS-peak emphasis. MAD-z under-weights baseline noise relative to QRS
    spikes, which means the local device (large QRS / baseline ratio) and
    WESAD's chest belt (smaller ratio) end up on very different output scales
    — a disaster for transfer. Std-z normalizes to unit variance regardless
    of device, then we clip extreme outliers so a single 60-Hz blip can't
    saturate the first conv layer.
    """
    x = x.astype(np.float64, copy=False)
    mean = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return (x - mean).astype(np.float32)
    z = (x - mean) / sd
    return np.clip(z, -8.0, 8.0).astype(np.float32)


def _upsample_linear(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x).astype(np.float32)


def _stack_and_window(
    ecg: np.ndarray,
    resp: np.ndarray,
    temp: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    rec_meta: dict,
) -> list[RawWindow]:
    """Slice ecg/resp/temp (all at TARGET_FS, aligned at t=0) into windows.

    `start_s` and `end_s` are in absolute recording seconds; we extract only
    windows whose [t, t+WINDOW_S) falls inside.
    """
    step = WINDOW_S * (1.0 - OVERLAP)
    out: list[RawWindow] = []

    n = min(len(ecg), len(resp), len(temp))
    i0 = max(0, int(round(start_s * TARGET_FS)))
    i1 = min(n, int(round(end_s * TARGET_FS)))

    t = start_s
    while t + WINDOW_S <= end_s + 1e-6:
        a = int(round(t * TARGET_FS))
        b = a + WINDOW_N
        if a < i0 or b > i1:
            t += step
            continue
        signal = np.stack([ecg[a:b], resp[a:b], temp[a:b]], axis=0).astype(np.float32)
        out.append(RawWindow(
            signal=signal,
            label=ACTIVITY_TO_LABEL[rec_meta["activity"]],
            activity=rec_meta["activity"],
            source=rec_meta["source"],
            subject=rec_meta["subject"],
            rec_name=rec_meta["rec_name"],
            t_start=t,
        ))
        t += step
    return out


# ---- local recordings -----------------------------------------------------

def windows_from_local_recording(rec: Recording) -> list[RawWindow]:
    """Build 3-channel windows from one local recording, labeled by phase/stressor."""
    pp = preprocess_recording(rec)   # ECG @ 250 Hz, BR @ 25 Hz, CPS @ 100 Hz

    # ECG already 250 Hz from preprocess_recording
    ecg_250 = _robust_z(pp.ecg)
    # Resp 25 Hz -> 250 Hz linear-interp upsample, then z-score
    resp_250 = _robust_z(_upsample_linear(pp.br, pp.fs["br"], TARGET_FS))
    # Temp: native ~1 Hz, irregular; resample onto a uniform 250 Hz grid spanning
    # the ECG window, then z-score
    temp_uniform = resample_uniform(rec.channels["temp"], TARGET_FS)
    # Align temp to ECG's t0 by padding/cropping
    n_target = len(ecg_250)
    if len(temp_uniform) < n_target:
        pad = np.full(n_target - len(temp_uniform), temp_uniform[-1] if len(temp_uniform) else 0.0)
        temp_uniform = np.concatenate([temp_uniform, pad])
    else:
        temp_uniform = temp_uniform[:n_target]
    temp_250 = _robust_z(temp_uniform.astype(np.float32))

    phases = phase_boundaries(rec)
    windows: list[RawWindow] = []
    for phase_name, (p_start, p_end) in phases.items():
        if p_end - p_start < WINDOW_S:
            continue
        activity = assign_activity(phase_name, rec.stressor)
        if activity not in ACTIVITY_TO_LABEL:
            # math or any future stressor type without a label slot -> skip
            continue
        # phase_boundaries are in absolute recording seconds (rec.channels["ecg"][0,0] offset).
        # preprocess_recording's ecg array starts at rec.channels["ecg"][0, 0] -> we shift to ecg local time.
        ecg_t0 = float(rec.channels["ecg"][0, 0])
        windows.extend(_stack_and_window(
            ecg_250, resp_250, temp_250,
            start_s=p_start - ecg_t0,
            end_s=p_end - ecg_t0,
            rec_meta=dict(
                activity=activity,
                source="local",
                subject=rec.subject,
                rec_name=rec.name,
            ),
        ))
    return windows


def local_raw_windows(data_dir: str = "data") -> list[RawWindow]:
    """All local recordings -> list of RawWindow."""
    out: list[RawWindow] = []
    for path in list_recordings(data_dir):
        rec = load_recording(path)
        ws = windows_from_local_recording(rec)
        print(f"  local  {os.path.basename(path):34s}  {len(ws):3d} windows")
        out.extend(ws)
    return out


# ---- WESAD ----------------------------------------------------------------

def windows_from_wesad_subject(subject_id: str, root: str = "WESAD") -> list[RawWindow]:
    sess = load_subject(subject_id, root=root)
    keep = set(WESAD_LABEL_TO_ACTIVITY.keys())
    runs = label_runs(sess.label, keep)
    if not runs:
        return []

    # Filter+resample the whole chest stream once. ECG at WESAD's 700 Hz ->
    # antialias via filter_ecg's 0.5-40 Hz band first (well below Nyquist) then
    # decimate to 250. For Resp, filter at 700 Hz BP 0.1-0.5 Hz first, then
    # decimate. For Temp, decimate directly.
    ecg_700 = sess.ecg - np.median(sess.ecg)
    ecg_700 = filter_ecg(ecg_700, sess.fs, mains=50.0)
    ecg_250 = _upsample_linear(ecg_700, sess.fs, TARGET_FS)
    ecg_250 = _robust_z(ecg_250)

    resp_700 = sess.resp - np.median(sess.resp)
    resp_700 = filter_br(resp_700, sess.fs)
    resp_250 = _upsample_linear(resp_700, sess.fs, TARGET_FS)
    resp_250 = _robust_z(resp_250)

    temp_250 = _upsample_linear(sess.temp, sess.fs, TARGET_FS)
    temp_250 = _robust_z(temp_250)

    windows: list[RawWindow] = []
    for run_start, run_end, lbl in runs:
        activity = WESAD_LABEL_TO_ACTIVITY[lbl]
        # convert WESAD-native indices to seconds, then to TARGET_FS index range
        start_s = run_start / sess.fs
        end_s = run_end / sess.fs
        if end_s - start_s < WINDOW_S:
            continue
        windows.extend(_stack_and_window(
            ecg_250, resp_250, temp_250,
            start_s=start_s,
            end_s=end_s,
            rec_meta=dict(
                activity=activity,
                source="wesad",
                subject=sess.subject_id,
                rec_name=f"WESAD_{sess.subject_id}_lbl{lbl}_run{run_start}",
            ),
        ))
    return windows


def wesad_raw_windows(root: str = "WESAD") -> list[RawWindow]:
    out: list[RawWindow] = []
    for sid in list_subjects(root):
        ws = windows_from_wesad_subject(sid, root=root)
        cnt = {a: 0 for a in ACTIVITY_TO_LABEL}
        for w in ws:
            cnt[w.activity] += 1
        print(f"  wesad  {sid:34s}  {len(ws):3d} windows  {cnt}")
        out.extend(ws)
    return out


# ---- helpers -------------------------------------------------------------

def stack_windows(windows: Iterable[RawWindow]) -> tuple[np.ndarray, np.ndarray]:
    """Stack a list of RawWindow into (N, 3, WINDOW_N) and (N,) arrays."""
    ws = list(windows)
    if not ws:
        return np.empty((0, N_CHANNELS, WINDOW_N), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.stack([w.signal for w in ws], axis=0).astype(np.float32)
    y = np.array([w.label for w in ws], dtype=np.int64)
    return X, y
