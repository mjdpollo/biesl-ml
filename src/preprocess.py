"""Filtering + peak detection per the agreed pipeline spec.

All bandpass / lowpass filters use scipy.signal.filtfilt for zero-phase response.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


# ---- filter design ---------------------------------------------------------

def _butter_bp_sos(low: float, high: float, fs: float, order: int):
    nyq = 0.5 * fs
    return signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def butter_bandpass(x: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass using SOS form (numerically stable at low cutoffs)."""
    sos = _butter_bp_sos(low, high, fs, order)
    return signal.sosfiltfilt(sos, x)


def notch(x: np.ndarray, fs: float, freq: float = 60.0, q: float = 30.0) -> np.ndarray:
    b, a = signal.iirnotch(freq / (fs / 2), q)
    return signal.filtfilt(b, a, x, padlen=min(len(x) - 1, 3 * max(len(b), len(a))))


# ---- per-signal filtering --------------------------------------------------

def filter_ecg(x: np.ndarray, fs: float, mains: float = 60.0) -> np.ndarray:
    """ECG: 0.5-40 Hz bandpass + mains notch."""
    y = butter_bandpass(x, 0.5, 40.0, fs, order=4)
    y = notch(y, fs, freq=mains, q=30.0)
    return y


def filter_ecg_for_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins prefilter: 5-15 Hz bandpass."""
    return butter_bandpass(x, 5.0, 15.0, fs, order=4)


def filter_br(x: np.ndarray, fs: float) -> np.ndarray:
    """Breathing band: 0.1-0.5 Hz BP order-3 + Savitzky-Golay smoothing."""
    y = butter_bandpass(x, 0.1, 0.5, fs, order=3)
    win = max(5, int(fs * 0.5))
    if win % 2 == 0:
        win += 1
    if win < 5:
        win = 5
    poly = min(3, win - 1)
    return signal.savgol_filter(y, window_length=win, polyorder=poly)


def filter_cps_cardiac(x: np.ndarray, fs: float) -> np.ndarray:
    """CPS cardiac branch: 0.8-3.0 Hz bandpass."""
    return butter_bandpass(x, 0.8, 3.0, fs, order=4)


def filter_cps_resp(x: np.ndarray, fs: float) -> np.ndarray:
    """CPS respiratory branch: 0.1-0.5 Hz bandpass."""
    return butter_bandpass(x, 0.1, 0.5, fs, order=4)


# ---- peak detection --------------------------------------------------------

def detect_ecg_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins via neurokit2, with kubios artefact correction.

    This device's R-peaks deflect NEGATIVE in the raw signal, so we negate
    before peak detection. After detection we snap each peak to the nearest
    local extremum of the flipped filtered signal within ±25 ms — the
    Pan-Tompkins output is offset a few samples from the true R-wave apex.
    """
    import neurokit2 as nk

    flipped = -ecg
    cleaned = nk.ecg_clean(flipped, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="pantompkins1985")
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if len(peaks) > 3:
        _, info2 = nk.signal_fixpeaks(peaks, sampling_rate=fs, method="kubios", iterative=True)
        if isinstance(info2, dict):
            peaks = np.asarray(info2.get("clean", peaks), dtype=int)
        else:
            peaks = np.asarray(info2, dtype=int)

    # snap to local maximum of flipped signal (= R-wave apex in original).
    # Pan-Tompkins reports peaks at the END of its integration window, so the
    # raw peak index lags the true apex by ~30-50 ms. ±60 ms covers it.
    half = int(round(0.060 * fs))
    snapped = []
    n = len(flipped)
    for p in peaks:
        lo = max(0, p - half)
        hi = min(n, p + half + 1)
        snapped.append(lo + int(np.argmax(flipped[lo:hi])))
    return np.asarray(snapped, dtype=int)


def detect_br_peaks(br: np.ndarray, fs: float) -> np.ndarray:
    """Breath peaks from a filtered respiratory signal.

    Tries neurokit2's rsp_peaks (BioSPPy method) first; falls back to a
    percentile-based find_peaks if NK fails or returns nothing.
    """
    try:
        import neurokit2 as nk
        _, info = nk.rsp_peaks(br, sampling_rate=int(round(fs)), method="biosppy")
        peaks = np.asarray(info.get("RSP_Peaks", []), dtype=int)
        if len(peaks) > 0:
            return peaks
    except Exception:
        pass

    std = float(np.std(br))
    if std < 1e-9:
        return np.array([], dtype=int)
    peaks, _ = signal.find_peaks(br, distance=max(1, int(fs * 1.5)), prominence=0.2 * std)
    return peaks


# ---- sanity check ----------------------------------------------------------

def sanity_report(name: str, fs: float, x: np.ndarray, *, flat_window_s: float = 2.0) -> dict:
    """Quick NaN / flat / saturation diagnostics. Returns a dict."""
    n = len(x)
    n_nan = int(np.isnan(x).sum())
    dur = n / fs
    # flat: rolling std < 1e-6 over flat_window_s
    win = max(5, int(flat_window_s * fs))
    if n >= win:
        s = np.convolve(np.abs(np.diff(x, prepend=x[0])), np.ones(win) / win, mode="same")
        flat_frac = float((s < 1e-6).mean())
    else:
        flat_frac = 0.0
    # naive saturation: fraction within 1% of min/max
    rng = x.max() - x.min() if n else 0.0
    sat_frac = 0.0
    if rng > 0:
        lo_thr = x.min() + 0.001 * rng
        hi_thr = x.max() - 0.001 * rng
        sat_frac = float(((x <= lo_thr) | (x >= hi_thr)).mean())
    return dict(
        name=name, n=n, fs=fs, duration_s=dur,
        n_nan=n_nan, flat_frac=flat_frac, sat_frac=sat_frac,
    )
