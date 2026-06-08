"""Poincare-plot IMAGES for the 2D-CNN.

Per-window 64x64 single-channel images built from the ECG RR (NN) series:

    x = RR_n,  y = RR_{n+1}
    range   = 300 ms .. 1400 ms
    bins    = 64 x 64
    value   = log(1 + count)
    normalize = per-image max  (default)  |  global

Windowing:
    * window length  = 60 s   (user-chosen; the 2-min spec leaves plank with
      ~6 windows total, so a 60-s window keeps all four classes trainable)
    * stride         = 20 s
    * each window must lie fully inside ONE labelled phase (rest | stress) and
      must NOT overlap the boundary exclusion zones around the 5-min and
      10-min protocol marks:  [290, 310] s and [590, 610] s.
    * recovery is dropped (assign_activity returns "" -> skipped), matching the
      rest of the project.

Partial exclusions requested for the curated set:
    * smj_6_6_math_17  -> excluded entirely
    * oyj_6_6_math_11  -> rest phase excluded (keep the math/stress phase)

RR peaks are taken from the SAME pipeline the classical features use
(`preprocess_recording` -> neurokit R-peaks -> clean_nn_intervals), so the
Poincare images are consistent with the rest of the repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .features import (
    _slice_peaks_by_time,
    preprocess_recording,
)
from .io import Recording, list_recordings, load_recording, phase_boundaries
from .pipeline import PHASE_CLASSES, assign_activity
from .preprocess import clean_nn_intervals

# ---- image / window configuration ------------------------------------------

RANGE_MS: tuple[float, float] = (300.0, 1400.0)
BINS: int = 64
WINDOW_S: float = 60.0
STRIDE_S: float = 20.0
MIN_NN: int = 10                      # need >= MIN_NN intervals (>= MIN_NN-1 points)

# Asymmetric? No — the user asked for symmetric +/-10 s around the 5- and
# 10-minute boundaries.
BOUNDARY_ZONES: tuple[tuple[float, float], ...] = ((290.0, 310.0), (590.0, 610.0))

# Partial exclusions (by Recording.name).
EXCLUDE_RECORDINGS: frozenset[str] = frozenset({"smj_6_6_math_17"})
EXCLUDE_REST_PHASE: frozenset[str] = frozenset({"oyj_6_6_math_11"})

ACTIVITY_TO_LABEL = {name: i for i, name in enumerate(PHASE_CLASSES)}
LABEL_TO_ACTIVITY = {v: k for k, v in ACTIVITY_TO_LABEL.items()}
LABEL_NAMES = tuple(PHASE_CLASSES)


@dataclass
class PoincareWindow:
    image: np.ndarray        # (BINS, BINS) float32, channel added later
    label: int
    activity: str
    subject: str
    rec_name: str
    t_start: float
    n_nn: int


def _overlaps_boundary(lo: float, hi: float) -> bool:
    for zlo, zhi in BOUNDARY_ZONES:
        if lo < zhi and hi > zlo:
            return True
    return False


def poincare_image(
    nn_ms: np.ndarray,
    *,
    bins: int = BINS,
    rng: tuple[float, float] = RANGE_MS,
    norm: str = "per_image",
) -> np.ndarray:
    """Build a (bins, bins) log-count Poincare image from an NN series (ms).

    Rows correspond to RR_{n+1} (y), columns to RR_n (x); both ascending.
    `norm="per_image"` divides by the per-image max; `norm="none"` returns the
    raw log-count grid (use src.poincare_images.normalize_global afterwards for
    global normalization).
    """
    nn = np.asarray(nn_ms, dtype=np.float64)
    x = nn[:-1]
    y = nn[1:]
    H, _, _ = np.histogram2d(x, y, bins=bins, range=[list(rng), list(rng)])
    img = np.log1p(H).T                       # transpose so axis0 = y = RR_{n+1}
    if norm == "per_image":
        m = float(img.max())
        if m > 0:
            img = img / m
    return img.astype(np.float32)


def normalize_global(images: np.ndarray, denom: float | None = None) -> tuple[np.ndarray, float]:
    """Global max-normalize a stack of raw log-count images.

    Pass `denom` (e.g. the train-set max) to avoid val/test leakage; if None the
    stack's own max is used. Returns (normalized, denom).
    """
    if denom is None:
        denom = float(images.max())
    if denom <= 0:
        denom = 1.0
    return (images / denom).astype(np.float32), denom


def windows_from_recording(rec: Recording, *, norm: str = "per_image") -> list[PoincareWindow]:
    if rec.name in EXCLUDE_RECORDINGS:
        return []

    pp = preprocess_recording(rec)            # neurokit R-peaks, same as features
    phases = phase_boundaries(rec)
    out: list[PoincareWindow] = []

    for phase_name, (p_start, p_end) in phases.items():
        if phase_name == "recovery":
            continue
        activity = assign_activity(phase_name, rec.stressor)
        if activity not in ACTIVITY_TO_LABEL:
            continue
        if phase_name == "rest" and rec.name in EXCLUDE_REST_PHASE:
            continue

        start = p_start
        while start + WINDOW_S <= p_end + 1e-9:
            w_lo, w_hi = start, start + WINDOW_S
            if not _overlaps_boundary(w_lo, w_hi):
                rp = _slice_peaks_by_time(pp.rpeaks, pp.fs_ecg, pp.ecg_t0, w_lo, w_hi)
                nn = clean_nn_intervals(rp, pp.fs_ecg)
                if len(nn) >= MIN_NN:
                    out.append(PoincareWindow(
                        image=poincare_image(nn, norm=norm),
                        label=ACTIVITY_TO_LABEL[activity],
                        activity=activity,
                        subject=rec.subject,
                        rec_name=rec.name,
                        t_start=float(start),
                        n_nn=int(len(nn)),
                    ))
            start += STRIDE_S

    return out


def build_dataset(data_dir: str = "data", *, norm: str = "per_image") -> list[PoincareWindow]:
    out: list[PoincareWindow] = []
    for path in list_recordings(data_dir):
        rec = load_recording(path)
        ws = windows_from_recording(rec, norm=norm)
        from collections import Counter
        acts = dict(Counter(w.activity for w in ws))
        print(f"  {os.path.basename(path):30s}  {len(ws):3d} windows  {acts}")
        out.extend(ws)
    return out


def stack(windows: list[PoincareWindow]) -> dict:
    """Stack to arrays. Images become (N, 1, BINS, BINS) float32."""
    X = np.stack([w.image for w in windows], axis=0)[:, None, :, :].astype(np.float32)
    y = np.array([w.label for w in windows], dtype=np.int64)
    rec_names = np.array([w.rec_name for w in windows])
    subjects = np.array([w.subject for w in windows])
    activities = np.array([w.activity for w in windows])
    t_start = np.array([w.t_start for w in windows], dtype=np.float64)
    return dict(X=X, y=y, rec_names=rec_names, subjects=subjects,
                activities=activities, t_start=t_start)


def build_and_cache(
    data_dir: str = "data",
    cache_path: str = "outputs/poincare_dataset.npz",
    *,
    norm: str = "per_image",
) -> dict:
    print(f"building Poincare images (window={WINDOW_S:.0f}s stride={STRIDE_S:.0f}s "
          f"bins={BINS} range={RANGE_MS} norm={norm}) ...")
    ws = build_dataset(data_dir, norm=norm)
    data = stack(ws)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **data)
    from collections import Counter
    print(f"\n  X={data['X'].shape}  labels="
          f"{ {LABEL_NAMES[k]: v for k, v in sorted(Counter(data['y'].tolist()).items())} }")
    print(f"  -> wrote {cache_path}")
    return data


def load_cache(cache_path: str = "outputs/poincare_dataset.npz") -> dict:
    npz = np.load(cache_path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}
