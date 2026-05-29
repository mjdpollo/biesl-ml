"""Per-window feature extraction per features.pdf.

Exactly eight features per window — no others:
    csi         : S2/S1 ratio of Shannon-energy peaks paired to ECG R-peaks
    hr          : 60_000 / mean(NN_ms)
    hrv_rmssd   : root-mean-square of successive NN differences (ms)
    hrv_lf      : Welch power in 0.04-0.15 Hz on the interpolated tachogram
    hrv_hf      : Welch power in 0.15-0.40 Hz
    hrv_lf_hf   : hrv_lf / hrv_hf
    rr          : 60 / mean(breath interval, s)        — breaths per minute
    rrv         : std of the last 5 breath intervals (s)

Window length is **60 s** (PDF: HRV LF/HF needs ≥ 1 min). 50 % overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .io import Recording, channel_fs, phase_boundaries, resample_uniform
from .preprocess import (
    clean_nn_intervals,
    classify_s1_s2,
    detect_br_peaks_neurokit,
    detect_ecg_rpeaks,
    detect_ecg_rpeaks_per_phase,
    detect_pcg_peaks,
    filter_br,
    filter_ecg,
    filter_mic_pcg,
    shannon_energy,
)


WINDOW_S = 40.0           # per request; below the 60 s the HRV LF/HF Welch
                          # step ideally wants — LF resolution is coarser at
                          # 40 s, so treat hrv_lf / hrv_lf_hf as approximate.
OVERLAP = 0.5

FEATURE_NAMES = (
    "csi",
    "hr",
    "hrv_rmssd",
    "hrv_lf",
    "hrv_hf",
    "hrv_lf_hf",
    "rr",
    "rrv",
)

# Optional temperature features — NOT in features.pdf, kept only because
# legacy ablation runners may still reference them. New code should NOT
# include them (the project no longer uses temperature).
TEMP_FEATURE_NAMES = (
    "temp_mean_C",
    "temp_std_C",
    "temp_slope_Cps",
)

# Skip windows whose extent touches the 5-min or 10-min protocol transitions
# (rest → stress at 300 s, stress → recovery at 600 s). The patient reports
# discomfort around these transitions; ±BOUNDARY_BUFFER_S around each
# boundary is excluded. Buffer=0 means any window that strictly contains a
# boundary point (e.g. [240, 300] or [300, 360]) is dropped.
BOUNDARY_TIMES_S = (300.0, 600.0)
BOUNDARY_BUFFER_S = 0.0


def _window_touches_boundary(t_start: float, t_end: float) -> bool:
    """True iff the window [t_start, t_end] touches any protocol boundary
    (within ±BOUNDARY_BUFFER_S). Used to drop boundary-spanning windows.
    """
    for b in BOUNDARY_TIMES_S:
        if t_start <= b + BOUNDARY_BUFFER_S and t_end >= b - BOUNDARY_BUFFER_S:
            return True
    return False


@dataclass
class Window:
    rec_name: str
    subject: str
    stressor: str
    phase: str               # rest | stress | recovery (label)
    t_start: float
    t_end: float
    features: dict[str, float]


# ---- HRV time / frequency / non-linear helpers -----------------------------

def _hr_from_nn(nn_ms: np.ndarray) -> float:
    if len(nn_ms) == 0:
        return float("nan")
    mean_nn = float(np.mean(nn_ms))
    if mean_nn <= 0:
        return float("nan")
    return 60_000.0 / mean_nn


def _rmssd(nn_ms: np.ndarray) -> float:
    if len(nn_ms) < 2:
        return float("nan")
    d = np.diff(nn_ms)
    return float(np.sqrt(np.mean(d * d)))


def _lf_hf(nn_ms: np.ndarray, fs_interp: float = 4.0) -> tuple[float, float]:
    """Welch PSD on a 4 Hz interpolated tachogram. Returns (LF, HF) in ms²."""
    if len(nn_ms) < 8:
        return float("nan"), float("nan")
    # cumulative time of each NN endpoint, in seconds
    t = np.cumsum(nn_ms) / 1000.0
    if t[-1] - t[0] < 30.0:                # too short for stable spectrum
        return float("nan"), float("nan")
    grid = np.arange(t[0], t[-1], 1.0 / fs_interp)
    if len(grid) < 32:
        return float("nan"), float("nan")
    tach = np.interp(grid, t, nn_ms)
    tach = tach - tach.mean()
    nperseg = min(len(tach), int(60 * fs_interp))     # 60-s segments where possible
    f, pxx = signal.welch(tach, fs=fs_interp, nperseg=nperseg, window="hann")

    def band(lo: float, hi: float) -> float:
        m = (f >= lo) & (f < hi)
        if not m.any():
            return float("nan")
        return float(np.trapezoid(pxx[m], f[m]))

    return band(0.04, 0.15), band(0.15, 0.40)


# ---- per-window feature extractors ----------------------------------------

def hrv_window_features(rpeaks_idx_window: np.ndarray, fs_ecg: float) -> dict[str, float]:
    """Compute hr / hrv_rmssd / hrv_lf / hrv_hf / hrv_lf_hf from R-peaks
    falling inside the window. Returns NaN for features that cannot be
    computed (too few peaks)."""
    nn = clean_nn_intervals(rpeaks_idx_window, fs_ecg)
    if len(nn) < 2:
        return {k: float("nan") for k in ("hr", "hrv_rmssd", "hrv_lf", "hrv_hf", "hrv_lf_hf")}
    hr = _hr_from_nn(nn)
    rmssd = _rmssd(nn)
    lf, hf = _lf_hf(nn)
    lf_hf = float(lf / hf) if (np.isfinite(lf) and np.isfinite(hf) and hf > 0) else float("nan")
    return dict(hr=hr, hrv_rmssd=rmssd, hrv_lf=lf, hrv_hf=hf, hrv_lf_hf=lf_hf)


def rr_window_features(
    peaks_idx_in_window: np.ndarray, fs_br: float,
) -> dict[str, float]:
    """RR (breath rate, bpm) and RRV (std of last 5 breath intervals, seconds)."""
    if len(peaks_idx_in_window) < 2:
        return dict(rr=float("nan"), rrv=float("nan"))
    intervals_s = np.diff(peaks_idx_in_window) / fs_br
    intervals_s = intervals_s[(intervals_s > 1.0) & (intervals_s < 12.0)]
    if len(intervals_s) < 1:
        return dict(rr=float("nan"), rrv=float("nan"))
    rr_bpm = 60.0 / float(np.mean(intervals_s))
    if len(intervals_s) < 2:
        rrv = float("nan")
    else:
        last5 = intervals_s[-5:]
        rrv = float(np.std(last5, ddof=0))      # PDF: "deviations in these intervals"
    return dict(rr=rr_bpm, rrv=rrv)


def csi_window_features(s1_amps: np.ndarray, s2_amps: np.ndarray) -> dict[str, float]:
    """CSI = mean(S2 amplitude) / mean(S1 amplitude) over the window."""
    if len(s1_amps) == 0 or len(s2_amps) == 0:
        return dict(csi=float("nan"))
    s1_mean = float(np.mean(s1_amps))
    s2_mean = float(np.mean(s2_amps))
    csi = float(s2_mean / s1_mean) if s1_mean > 0 else float("nan")
    return dict(csi=csi)


def temp_features_window(temp_ch: np.ndarray, t_start: float, t_end: float) -> dict[str, float]:
    """Optional skin-temperature features for the ablation comparison.

    NOT part of features.pdf. Returns mean / std / linear-fit slope of the
    raw 2xN (time, value) temperature channel slice in [t_start, t_end).
    """
    t, v = temp_ch[0], temp_ch[1]
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 2:
        return {k: float("nan") for k in TEMP_FEATURE_NAMES}
    tt = t[mask]
    vv = v[mask]
    out: dict[str, float] = {
        "temp_mean_C": float(np.mean(vv)),
        "temp_std_C":  float(np.std(vv)),
        "temp_slope_Cps": float(np.polyfit(tt - tt[0], vv, 1)[0]) if len(tt) >= 3 else float("nan"),
    }
    return out


# ---- recording-level preprocessing ----------------------------------------

@dataclass
class Preprocessed:
    rec: Recording
    fs_ecg: float
    ecg: np.ndarray
    ecg_t0: float
    rpeaks: np.ndarray
    fs_br: float
    br: np.ndarray
    br_t0: float
    br_peaks: np.ndarray
    fs_mic: float
    se: np.ndarray              # Shannon-energy envelope at fs_mic
    pcg_peaks: np.ndarray       # indices into `se`
    mic_t0: float


def preprocess_recording(
    rec: Recording,
    ecg_fs_target: float = 500.0,
    br_fs_target: float = 100.0,
    mic_fs_target: float = 2000.0,
    *,
    phase_aware: bool = False,
    rpeak_method: str = "neurokit",
) -> Preprocessed:
    """Resample to common grids and run the PDF's per-modality pipelines.

    `phase_aware`: when True, run R-peak detection independently per protocol
    phase (rest / stress / recovery). Isolates adaptive detector state so
    plank-noise can't contaminate adjacent phases.
    `rpeak_method`: forwarded to neurokit2.ecg_peaks (default pantompkins1985).
    """
    # ECG @ 500 Hz (matches local native; gives good QRS resolution)
    ecg_t0 = float(rec.channels["ecg"][0, 0])
    ecg_u = resample_uniform(rec.channels["ecg"], ecg_fs_target).astype(np.float64)
    ecg_f = filter_ecg(ecg_u, ecg_fs_target)
    if phase_aware:
        phases = phase_boundaries(rec)
        rpeaks = detect_ecg_rpeaks_per_phase(
            ecg_f, ecg_fs_target, ecg_t0, phases, method=rpeak_method,
        )
    else:
        rpeaks = detect_ecg_rpeaks(
            ecg_f, ecg_fs_target, polarity=rec.ecg_polarity, method=rpeak_method,
        )

    # BR @ 100 Hz (≥ 4× the Cheby II stopband edge of 1 Hz, comfortable margin)
    br_t0 = float(rec.channels["br"][0, 0])
    br_u = resample_uniform(rec.channels["br"], br_fs_target).astype(np.float64)
    br_f = filter_br(br_u, br_fs_target)
    # neurokit2's rsp_peaks (biosppy method) wins the detector comparison —
    # 3-5x tighter per-phase IQR across recordings than either the global
    # prominence or the sliding-window adaptive detector. See
    # br-detector-comparison.md.
    br_peaks = detect_br_peaks_neurokit(br_f, br_fs_target)

    # Mic @ 2 kHz (matches native rate, Nyquist 1 kHz ≫ 200 Hz PCG band)
    mic_t0 = float(rec.channels["mic"][0, 0])
    mic_u = resample_uniform(rec.channels["mic"], mic_fs_target).astype(np.float64)
    mic_f = filter_mic_pcg(mic_u, mic_fs_target)
    se = shannon_energy(mic_f, smooth_ms=50.0, fs=mic_fs_target)
    pcg_peaks = detect_pcg_peaks(se, mic_fs_target)

    return Preprocessed(
        rec=rec,
        fs_ecg=ecg_fs_target, ecg=ecg_f, ecg_t0=ecg_t0, rpeaks=rpeaks,
        fs_br=br_fs_target, br=br_f, br_t0=br_t0, br_peaks=br_peaks,
        fs_mic=mic_fs_target, se=se, pcg_peaks=pcg_peaks, mic_t0=mic_t0,
    )


# ---- windowing ------------------------------------------------------------

def _slice_peaks_by_time(peaks_idx: np.ndarray, fs: float, t0: float,
                         t_start: float, t_end: float) -> np.ndarray:
    """Peak indices whose absolute time falls in [t_start, t_end)."""
    if len(peaks_idx) == 0:
        return peaks_idx
    t = t0 + peaks_idx / fs
    mask = (t >= t_start) & (t < t_end)
    return peaks_idx[mask]


def windows_for_recording(
    pp: Preprocessed,
    window_s: float = WINDOW_S,
    overlap: float = OVERLAP,
    *,
    include_temp: bool = False,
) -> list[Window]:
    rec = pp.rec
    phases = phase_boundaries(rec)
    step = window_s * (1.0 - overlap)
    windows: list[Window] = []

    for phase_name, (p_start, p_end) in phases.items():
        if p_end - p_start < window_s:
            continue
        # 'recovery' is no longer part of the taxonomy — drop those windows entirely.
        if phase_name == "recovery":
            continue
        t = p_start
        while t + window_s <= p_end + 1e-6:
            t_end = t + window_s

            # Skip windows that touch the 5-min / 10-min protocol boundaries
            # — patient reports discomfort around the transitions.
            if _window_touches_boundary(t, t_end):
                t += step
                continue

            # ECG: slice R-peaks, compute HR/HRV
            ecg_window_peaks = _slice_peaks_by_time(
                pp.rpeaks, pp.fs_ecg, pp.ecg_t0, t, t_end,
            )
            feats: dict[str, float] = {}
            feats.update(hrv_window_features(ecg_window_peaks, pp.fs_ecg))

            # BR: slice breath peaks within window, compute RR/RRV
            br_window_peaks = _slice_peaks_by_time(
                pp.br_peaks, pp.fs_br, pp.br_t0, t, t_end,
            )
            feats.update(rr_window_features(br_window_peaks, pp.fs_br))

            # Mic: take Shannon-energy envelope slice + paired S1/S2 detection
            # against the ECG R-peaks expressed at the SE sample rate.
            mic_t0 = pp.mic_t0
            t_to_idx = lambda t_abs: int(round((t_abs - mic_t0) * pp.fs_mic))
            i0_mic, i1_mic = max(0, t_to_idx(t)), min(len(pp.se), t_to_idx(t_end))
            se_slice = pp.se[i0_mic:i1_mic]
            pcg_slice = pp.pcg_peaks[
                (pp.pcg_peaks >= i0_mic) & (pp.pcg_peaks < i1_mic)
            ] - i0_mic
            # convert R-peak indices to SE-sample frame inside the window
            r_abs_t = pp.ecg_t0 + ecg_window_peaks / pp.fs_ecg
            r_se_idx = np.round((r_abs_t - t) * pp.fs_mic).astype(int)
            r_se_idx = r_se_idx[(r_se_idx >= 0) & (r_se_idx < len(se_slice))]
            s1_amps, s2_amps = classify_s1_s2(
                se_slice, pp.fs_mic, pcg_slice, r_se_idx,
            )
            feats.update(csi_window_features(s1_amps, s2_amps))

            # Optional temperature ablation features.
            if include_temp:
                feats.update(temp_features_window(rec.channels["temp"], t, t_end))

            # Sanitize inf/None -> NaN so downstream imputation handles them.
            for k, v in list(feats.items()):
                if v is None or not np.isfinite(v):
                    feats[k] = float("nan")

            # ENFORCE the feature schema — PDF features always; temp only if requested.
            keep = list(FEATURE_NAMES) + (list(TEMP_FEATURE_NAMES) if include_temp else [])
            feats = {k: feats.get(k, float("nan")) for k in keep}

            windows.append(Window(
                rec_name=rec.name, subject=rec.subject, stressor=rec.stressor,
                phase=phase_name, t_start=t, t_end=t_end, features=feats,
            ))
            t += step

    return windows
