"""Per-window feature extraction.

A "window" is a fixed-length segment (default 30 s, 50% overlap) of the
recording. For each window we compute:
    - HRV time/freq/nonlinear features from ECG R-peaks
    - Breath rate / variability from BR peaks
    - CPS cardiac (heart-rate band) and respiratory band features
    - Skin temperature mean / slope / std
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .io import Recording, channel_fs, phase_boundaries, resample_uniform
from .preprocess import (
    detect_br_peaks,
    detect_ecg_rpeaks,
    filter_br,
    filter_cps_cardiac,
    filter_cps_resp,
    filter_ecg,
)


WINDOW_S = 30.0
OVERLAP = 0.5
MIN_PEAKS_PER_WINDOW = 5    # drop windows with too few detected peaks for HRV
MIN_BR_PEAKS_PER_WINDOW = 3


@dataclass
class Window:
    rec_name: str
    subject: str
    stressor: str            # original stressor type for the recording
    phase: str               # rest | stress | recovery (label)
    t_start: float
    t_end: float
    features: dict[str, float]


# ---- helpers --------------------------------------------------------------

def _slice_uniform(values: np.ndarray, fs: float, t0_offset: float,
                   t_start: float, t_end: float) -> np.ndarray:
    """Slice a uniformly-sampled signal by absolute time.

    `values` was sampled at `fs` starting at absolute time `t0_offset`.
    """
    i0 = int(round((t_start - t0_offset) * fs))
    i1 = int(round((t_end - t0_offset) * fs))
    i0 = max(i0, 0)
    i1 = min(i1, len(values))
    return values[i0:i1]


def _slice_peaks(peaks_idx: np.ndarray, fs: float, t0_offset: float,
                 t_start: float, t_end: float) -> np.ndarray:
    """Return peak indices that fall in [t_start, t_end), relative to original array start."""
    t = t0_offset + peaks_idx / fs
    mask = (t >= t_start) & (t < t_end)
    return peaks_idx[mask]


# ---- per-window feature extractors ----------------------------------------

def hrv_features(rpeaks_idx: np.ndarray, fs_ecg: float) -> dict[str, float]:
    """HRV time/freq/nonlinear features from R-peak indices within a window."""
    out: dict[str, float] = {}
    if len(rpeaks_idx) < MIN_PEAKS_PER_WINDOW:
        return out

    rr = np.diff(rpeaks_idx) / fs_ecg                 # seconds
    rr = rr[(rr > 0.3) & (rr < 2.0)]                  # physiologic
    if len(rr) < 4:
        return out

    # time-domain
    out["hrv_meanRR_ms"] = float(np.mean(rr) * 1000)
    out["hrv_SDNN_ms"]   = float(np.std(rr, ddof=1) * 1000)
    out["hrv_RMSSD_ms"]  = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000) if len(rr) > 1 else 0.0
    out["hrv_pNN50"]     = float(np.mean(np.abs(np.diff(rr)) > 0.05)) if len(rr) > 1 else 0.0
    out["hrv_minRR_ms"]  = float(np.min(rr) * 1000)
    out["hrv_maxRR_ms"]  = float(np.max(rr) * 1000)
    out["hrv_HR_bpm"]    = float(60.0 / np.mean(rr))

    # frequency-domain via Welch on 4 Hz interpolated tachogram
    if len(rr) >= 8:
        t_rr = np.cumsum(rr)
        fs_tach = 4.0
        grid = np.arange(0, t_rr[-1], 1 / fs_tach)
        if len(grid) >= 16:
            interp = np.interp(grid, t_rr, rr)
            interp = interp - interp.mean()
            nperseg = min(len(interp), 256)
            f, pxx = signal.welch(interp, fs=fs_tach, nperseg=nperseg)
            def band_power(lo, hi):
                mask = (f >= lo) & (f < hi)
                if not mask.any():
                    return 0.0
                return float(np.trapezoid(pxx[mask], f[mask]))
            lf = band_power(0.04, 0.15)
            hf = band_power(0.15, 0.40)
            out["hrv_LF"] = lf
            out["hrv_HF"] = hf
            out["hrv_LFHF"] = float(lf / hf) if hf > 0 else 0.0
            out["hrv_totalpwr"] = float(np.trapezoid(pxx, f))

    # nonlinear: Poincare SD1/SD2
    if len(rr) > 2:
        d = np.diff(rr)
        sd1 = float(np.std(d, ddof=1) / np.sqrt(2))
        sd2 = float(np.sqrt(2 * np.var(rr, ddof=1) - 0.5 * np.var(d, ddof=1)))
        out["hrv_SD1"] = sd1
        out["hrv_SD2"] = sd2
        out["hrv_SD1SD2"] = float(sd1 / sd2) if sd2 > 0 else 0.0

    # sample entropy (approximate, m=2, r=0.2*std)
    if len(rr) >= 10:
        try:
            out["hrv_sampEn"] = _sample_entropy(rr, m=2, r=0.2 * np.std(rr))
        except Exception:
            pass

    return out


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return float("nan")

    def _phi(mm: int) -> float:
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        cnt = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            cnt += int(np.sum(d <= r)) - 1  # exclude self
        return cnt

    A = _phi(m + 1)
    B = _phi(m)
    if A == 0 or B == 0:
        return float("nan")
    return float(-np.log(A / B))


def br_features(br_window: np.ndarray, peaks_in_window_idx: np.ndarray, fs_br: float) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(peaks_in_window_idx) >= MIN_BR_PEAKS_PER_WINDOW:
        intervals = np.diff(peaks_in_window_idx) / fs_br  # seconds
        intervals = intervals[(intervals > 1.0) & (intervals < 12.0)]
        if len(intervals) >= 2:
            out["br_rate_bpm"] = float(60.0 / np.mean(intervals))
            out["br_RRV_std_s"] = float(np.std(intervals, ddof=1))
            amps = br_window[peaks_in_window_idx - peaks_in_window_idx.min()] \
                if peaks_in_window_idx.min() < len(br_window) else np.array([])
            if amps.size:
                out["br_amp_mean"] = float(np.mean(amps))
                out["br_amp_std"]  = float(np.std(amps))

    # dominant frequency via welch
    if len(br_window) >= int(fs_br * 5):
        nperseg = min(len(br_window), int(fs_br * 30))
        f, pxx = signal.welch(br_window, fs=fs_br, nperseg=nperseg)
        band = (f >= 0.1) & (f <= 0.5)
        if band.any() and pxx[band].max() > 0:
            out["br_dom_freq_Hz"] = float(f[band][np.argmax(pxx[band])])
            out["br_band_power"] = float(np.trapezoid(pxx[band], f[band]))
    return out


def cps_features(cardiac_band: np.ndarray, resp_band: np.ndarray, fs: float) -> dict[str, float]:
    out: dict[str, float] = {}
    # cardiac branch: peak rate as cardio rhythm proxy
    if len(cardiac_band) >= int(fs * 5):
        std = float(np.std(cardiac_band))
        if std > 0:
            peaks, _ = signal.find_peaks(
                cardiac_band, distance=int(fs * 0.4), prominence=0.3 * std,
            )
            if len(peaks) >= 4:
                ivl = np.diff(peaks) / fs
                ivl = ivl[(ivl > 0.3) & (ivl < 2.0)]
                if len(ivl) >= 2:
                    out["cps_card_rate_bpm"] = float(60.0 / np.mean(ivl))
                    out["cps_card_RR_std"]   = float(np.std(ivl))
        out["cps_card_rms"] = float(np.sqrt(np.mean(cardiac_band ** 2)))

    # respiratory branch
    if len(resp_band) >= int(fs * 5):
        std = float(np.std(resp_band))
        if std > 0:
            peaks, _ = signal.find_peaks(
                resp_band, distance=int(fs * 1.5), prominence=0.3 * std,
            )
            if len(peaks) >= 3:
                ivl = np.diff(peaks) / fs
                ivl = ivl[(ivl > 1.0) & (ivl < 12.0)]
                if len(ivl) >= 2:
                    out["cps_resp_rate_bpm"] = float(60.0 / np.mean(ivl))
                    out["cps_resp_RRV_std"]  = float(np.std(ivl))
        out["cps_resp_rms"] = float(np.sqrt(np.mean(resp_band ** 2)))
    return out


def temp_features(temp_ch: np.ndarray, t_start: float, t_end: float) -> dict[str, float]:
    t, v = temp_ch[0], temp_ch[1]
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 2:
        return {}
    tt = t[mask]
    vv = v[mask]
    out = {
        "temp_mean_C": float(np.mean(vv)),
        "temp_std_C":  float(np.std(vv)),
    }
    # slope (linear fit)
    if len(tt) >= 3:
        slope = float(np.polyfit(tt - tt[0], vv, 1)[0])
        out["temp_slope_Cps"] = slope
    return out


# ---- recording-level pipeline ---------------------------------------------

@dataclass
class Preprocessed:
    """Filtered signals + peak indices for one recording."""
    rec: Recording
    fs: dict[str, float]               # fs per uniform-resampled channel
    ecg: np.ndarray                    # bandpass-filtered, uniform fs
    ecg_t0: float                      # absolute time of ecg[0]
    rpeaks: np.ndarray                 # indices into ecg
    br: np.ndarray                     # filtered breathing signal
    br_t0: float
    br_peaks: np.ndarray
    cps_card: np.ndarray               # mic, cardiac band
    cps_resp: np.ndarray               # mic, respiratory band
    cps_t0: float                      # absolute time of cps[0]
    cps_fs: float                      # downsampled fs for cps branches


def _zscore(x: np.ndarray) -> np.ndarray:
    """Robust z-score: median + 1.4826*MAD. Resistant to outlier spikes from
    motion artifacts that would otherwise inflate std and crush the signal."""
    x = x.astype(np.float64, copy=False)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        # fall back to std for nearly-constant signals
        sd = float(np.std(x))
        return (x - med) / sd if sd > 1e-12 else x - med
    return (x - med) / scale


def preprocess_recording(
    rec: Recording,
    ecg_fs_target: float = 250.0,
    br_fs_target: float = 25.0,
    cps_fs_target: float = 100.0,
) -> Preprocessed:
    """Resample to common grids, normalize, filter, detect peaks.

    Targets follow the pipeline spec: 250 Hz ECG, 25 Hz BR. The mic is
    downsampled to `cps_fs_target` so the 0.1-3 Hz bands stay well within Nyquist.
    """
    ecg_t0 = float(rec.channels["ecg"][0, 0])
    # ECG: detrend only — the QRS spikes ARE the signal, so robust-zscore would
    # crush them. Pan-Tompkins doesn't care about absolute amplitude.
    ecg_u = resample_uniform(rec.channels["ecg"], ecg_fs_target).astype(np.float64)
    ecg_u = ecg_u - np.median(ecg_u)
    ecg_f = filter_ecg(ecg_u, ecg_fs_target)
    rpeaks = detect_ecg_rpeaks(ecg_f, ecg_fs_target)

    br_t0 = float(rec.channels["br"][0, 0])
    br_u = _zscore(resample_uniform(rec.channels["br"], br_fs_target))
    br_f = filter_br(br_u, br_fs_target)
    br_peaks = detect_br_peaks(br_f, br_fs_target)

    cps_t0 = float(rec.channels["mic"][0, 0])
    cps_u = _zscore(resample_uniform(rec.channels["mic"], cps_fs_target))
    cps_card = filter_cps_cardiac(cps_u, cps_fs_target)
    cps_resp = filter_cps_resp(cps_u, cps_fs_target)

    return Preprocessed(
        rec=rec,
        fs={"ecg": ecg_fs_target, "br": br_fs_target, "cps": cps_fs_target},
        ecg=ecg_f, ecg_t0=ecg_t0, rpeaks=rpeaks,
        br=br_f, br_t0=br_t0, br_peaks=br_peaks,
        cps_card=cps_card, cps_resp=cps_resp, cps_t0=cps_t0,
        cps_fs=cps_fs_target,
    )


def windows_for_recording(
    pp: Preprocessed,
    window_s: float = WINDOW_S,
    overlap: float = OVERLAP,
) -> list[Window]:
    rec = pp.rec
    phases = phase_boundaries(rec)
    step = window_s * (1.0 - overlap)
    windows: list[Window] = []

    for phase_name, (p_start, p_end) in phases.items():
        if p_end - p_start < window_s:
            # phase too short for one full window; skip
            continue
        t = p_start
        while t + window_s <= p_end + 1e-6:
            t_end = t + window_s
            # extract signal slices and peak subsets
            ecg_slice_peaks = _slice_peaks(
                pp.rpeaks, pp.fs["ecg"], pp.ecg_t0, t, t_end,
            )
            ecg_peaks_local = ecg_slice_peaks - int(round((t - pp.ecg_t0) * pp.fs["ecg"]))

            br_slice = _slice_uniform(pp.br, pp.fs["br"], pp.br_t0, t, t_end)
            br_idx_global = pp.br_peaks
            br_t = pp.br_t0 + br_idx_global / pp.fs["br"]
            br_mask = (br_t >= t) & (br_t < t_end)
            br_local = (br_idx_global[br_mask] - int(round((t - pp.br_t0) * pp.fs["br"])))
            br_local = br_local[(br_local >= 0) & (br_local < len(br_slice))]

            cps_card_slice = _slice_uniform(pp.cps_card, pp.cps_fs, pp.cps_t0, t, t_end)
            cps_resp_slice = _slice_uniform(pp.cps_resp, pp.cps_fs, pp.cps_t0, t, t_end)

            feats: dict[str, float] = {}
            feats.update(hrv_features(ecg_slice_peaks, pp.fs["ecg"]))
            feats.update(br_features(br_slice, br_local, pp.fs["br"]))
            feats.update(cps_features(cps_card_slice, cps_resp_slice, pp.cps_fs))
            feats.update(temp_features(rec.channels["temp"], t, t_end))
            # Sanitize: convert inf to NaN so downstream imputation can handle.
            for k, v in list(feats.items()):
                if v is None or not np.isfinite(v):
                    feats[k] = float("nan")

            windows.append(Window(
                rec_name=rec.name, subject=rec.subject, stressor=rec.stressor,
                phase=phase_name, t_start=t, t_end=t_end, features=feats,
            ))
            t += step

    return windows
