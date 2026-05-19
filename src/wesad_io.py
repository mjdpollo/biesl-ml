"""Load WESAD chest data + emit windows that match the local feature schema.

WESAD's chest RespiBAN gives ECG / Resp / Temp at 700 Hz with a parallel int32
`label` vector. We map WESAD's affective states onto this project's
{baseline, meditation, stress} taxonomy:

    WESAD label 1 (baseline reading)       -> 'baseline'
    WESAD label 4 (guided meditation)      -> 'meditation'
    WESAD label 2 (TSST stress)            -> 'stress'

WESAD label 3 (amusement) and 0/5/6/7 (transient) are dropped.

Caveat on the stress mapping: WESAD's TSST induces psychological stress
(public speaking + arithmetic) while the local 'stress' class is physical
(plank exercise). They share elevated sympathetic tone (higher HR, suppressed
HRV) but the breathing signatures differ — TSST resp is chopped by speech
artifacts, physical stress resp is effort-driven. We accept the modality
mismatch in exchange for substantially more labeled stress windows than
local-only provides.

WESAD has no microphone -> the CPS feature branch from src.features is *not*
computed here. Models trained on WESAD use HRV + BR + Temp features only.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import (
    MIN_PEAKS_PER_WINDOW,
    WINDOW_S,
    OVERLAP,
    br_features,
    hrv_features,
    temp_features,
)
from .preprocess import (
    detect_br_peaks,
    detect_ecg_rpeaks,
    filter_br,
    filter_ecg,
)


WESAD_LABEL_TO_ACTIVITY = {
    1: "baseline",
    4: "meditation",
    2: "stress",   # TSST stress -> 'stress' (physical-vs-psychological caveat in module docstring)
}
WESAD_FS = 700.0


@dataclass
class WesadSession:
    subject_id: str
    fs: float                          # 700 Hz
    ecg: np.ndarray                    # raw chest ECG, (N,)
    resp: np.ndarray                   # raw chest Resp, (N,)
    temp: np.ndarray                   # raw chest Temp, (N,)
    label: np.ndarray                  # int32, (N,)


def load_subject(subject_id: str, root: str = "WESAD") -> WesadSession:
    """Load S{N}.pkl, return chest signals + parallel label vector."""
    path = os.path.join(root, subject_id, f"{subject_id}.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest = d["signal"]["chest"]
    return WesadSession(
        subject_id=subject_id,
        fs=WESAD_FS,
        ecg=chest["ECG"].ravel().astype(np.float64),
        resp=chest["Resp"].ravel().astype(np.float64),
        temp=chest["Temp"].ravel().astype(np.float64),
        label=np.asarray(d["label"], dtype=np.int32),
    )


def list_subjects(root: str = "WESAD") -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if d.startswith("S") and os.path.isdir(os.path.join(root, d))
        and os.path.exists(os.path.join(root, d, f"{d}.pkl"))
    )


def label_runs(label: np.ndarray, keep: set[int]) -> list[tuple[int, int, int]]:
    """Find contiguous runs of label values in `keep`.

    Returns list of (start_idx, end_idx_exclusive, label_value).
    """
    mask = np.isin(label, list(keep))
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    runs: list[tuple[int, int, int]] = []
    for s, e in zip(starts, ends):
        # within each run, label is constant by construction
        runs.append((int(s), int(e), int(label[s])))
    return runs


def _resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Linear-interpolating resample. fs_out <= fs_in here, so antialiasing is
    handled by the downstream filter step (filter_ecg / filter_br)."""
    if fs_in == fs_out:
        return x
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x).astype(np.float64)


def _temp_channel_2xn(temp_uniform: np.ndarray, fs: float) -> np.ndarray:
    """Wrap a uniform temp array into the 2xN (time, value) form temp_features expects."""
    t = np.arange(len(temp_uniform)) / fs
    return np.vstack([t, temp_uniform])


def windows_for_subject(
    sess: WesadSession,
    *,
    ecg_fs_target: float = 250.0,
    br_fs_target: float = 25.0,
    temp_fs_target: float = 4.0,
    window_s: float = WINDOW_S,
    overlap: float = OVERLAP,
    min_run_s: float = 60.0,
) -> list[dict]:
    """Slide windows over labeled WESAD segments and emit feature dicts.

    Produces records with the same per-window features as the local pipeline's
    HRV + BR + Temp branches (no CPS). Each record carries the WESAD-mapped
    activity label so the result can be concatenated with the local feature table.
    """
    runs = label_runs(sess.label, set(WESAD_LABEL_TO_ACTIVITY.keys()))
    step = window_s * (1.0 - overlap)
    records: list[dict] = []

    for run_start, run_end, lbl in runs:
        # skip runs too short for even one window
        if (run_end - run_start) / sess.fs < max(min_run_s, window_s):
            continue
        activity = WESAD_LABEL_TO_ACTIVITY[lbl]

        # slice + resample each modality to its target rate for this run
        ecg_raw = sess.ecg[run_start:run_end]
        resp_raw = sess.resp[run_start:run_end]
        temp_raw = sess.temp[run_start:run_end]

        ecg_u = _resample_to(ecg_raw, sess.fs, ecg_fs_target)
        ecg_u = ecg_u - np.median(ecg_u)
        ecg_f = filter_ecg(ecg_u, ecg_fs_target, mains=50.0)   # WESAD recorded in DE -> 50 Hz mains
        # WESAD's ECG has the conventional polarity (R-peaks UP), unlike the local device.
        # detect_ecg_rpeaks flips internally, so we pre-flip here to cancel.
        rpeaks = detect_ecg_rpeaks(-ecg_f, ecg_fs_target)

        resp_u = (resp_raw - np.median(resp_raw))
        # robust z-score by MAD
        mad = np.median(np.abs(resp_u - np.median(resp_u)))
        scale = 1.4826 * mad if mad > 1e-12 else (np.std(resp_u) or 1.0)
        resp_u = resp_u / scale
        resp_u = _resample_to(resp_u, sess.fs, br_fs_target)
        resp_f = filter_br(resp_u, br_fs_target)
        br_peaks = detect_br_peaks(resp_f, br_fs_target)

        temp_u = _resample_to(temp_raw, sess.fs, temp_fs_target)

        # window iteration
        run_dur_s = (run_end - run_start) / sess.fs
        t = 0.0
        win_idx = 0
        while t + window_s <= run_dur_s + 1e-6:
            t_end = t + window_s

            # ECG window: slice peak indices that fall in [t, t_end)
            ecg_t = rpeaks / ecg_fs_target
            ecg_mask = (ecg_t >= t) & (ecg_t < t_end)
            ecg_window_peaks = rpeaks[ecg_mask]
            # rebase peak indices to window start so hrv_features sees them at correct offsets
            # (hrv_features only uses np.diff, so absolute offsets don't matter; pass as-is)

            # BR window slice
            i0_br = int(round(t * br_fs_target))
            i1_br = int(round(t_end * br_fs_target))
            br_window = resp_f[i0_br:i1_br]
            br_t = br_peaks / br_fs_target
            br_mask = (br_t >= t) & (br_t < t_end)
            br_window_peaks = br_peaks[br_mask] - i0_br
            br_window_peaks = br_window_peaks[
                (br_window_peaks >= 0) & (br_window_peaks < len(br_window))
            ]

            # temp window
            temp_ch = _temp_channel_2xn(temp_u, temp_fs_target)

            feats: dict = {}
            feats.update(hrv_features(ecg_window_peaks, ecg_fs_target))
            feats.update(br_features(br_window, br_window_peaks, br_fs_target))
            feats.update(temp_features(temp_ch, t, t_end))
            for k, v in list(feats.items()):
                if v is None or not np.isfinite(v):
                    feats[k] = float("nan")

            rec_name = f"WESAD_{sess.subject_id}_lbl{lbl}_run{run_start}_{win_idx}"
            row = {
                "rec_name":  f"WESAD_{sess.subject_id}_lbl{lbl}_run{run_start}",
                "subject":   sess.subject_id,
                "stressor":  "wesad",
                "phase":     activity,
                "activity":  activity,
                "t_start":   t,
                "t_end":     t_end,
            }
            row.update(feats)
            records.append(row)

            t += step
            win_idx += 1

    return records


def build_wesad_feature_table(root: str = "WESAD") -> pd.DataFrame:
    """Process every WESAD subject under `root` and return one DataFrame."""
    all_records: list[dict] = []
    for sid in list_subjects(root):
        sess = load_subject(sid, root=root)
        recs = windows_for_subject(sess)
        n_per_label = pd.Series([r["activity"] for r in recs]).value_counts().to_dict()
        print(f"  {sid}: {len(recs):4d} windows  {n_per_label}")
        all_records.extend(recs)
    return pd.DataFrame(all_records)
