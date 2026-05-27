"""Signal preprocessing per features.pdf.

ECG  : low-pass 150 Hz + detrend, then Pan-Tompkins R-peak detection (peaks
       are negative in the raw local signal; pre-flipped before detection).
BR   : detrend -> 0.5 s moving-average -> 4th-order Cheby II low-pass
       (stopband edge 1 Hz) -> 2nd-order Butterworth high-pass at 0.12 Hz.
       Peak detection uses the PDF's slope-based algorithm with an adaptive
       1/3-of-recent-amplitude threshold.
Mic  : bandpass 20-200 Hz (PCG band), then Shannon-energy envelope
       SE = -x^2 log(x^2). S1/S2 peaks classified per cardiac cycle using
       the ECG R-peak timing.
NN   : intervals shorter than 300 ms or longer than 1500 ms are rejected;
       intervals that deviate more than 20 % from the median of the
       surrounding ~10 beats are also rejected; rejects are cubic-spline
       interpolated.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


# ---- ECG --------------------------------------------------------------------

def filter_ecg(x: np.ndarray, fs: float, mains: float | None = 60.0) -> np.ndarray:
    """HP 1 Hz -> notch `mains` Hz (Q=30) -> LP 150 Hz. Zero-phase (sosfiltfilt).

    HP removes baseline wander; notch removes mains interference (default 60 Hz
    for local Korean recordings, pass mains=50.0 for WESAD/EU); LP caps wideband
    noise. All sections chained in SOS form for numerical stability.
    """
    nyq = 0.5 * fs
    sos_hp = signal.butter(4, 1.0, btype="high", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos_hp, x)
    if mains and mains > 0 and mains < nyq:
        b, a = signal.iirnotch(w0=mains, Q=30.0, fs=fs)
        sos_notch = signal.tf2sos(b, a)
        y = signal.sosfiltfilt(sos_notch, y)
    cutoff = min(150.0, nyq * 0.95)
    sos_lp = signal.butter(4, cutoff, btype="low", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos_lp, y)
    return y


def filter_ecg_for_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins 5-15 Hz prefilter (kept for callers that want it explicit)."""
    sos = signal.butter(4, [5.0, 15.0], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def detect_ecg_rpeaks(ecg: np.ndarray, fs: float,
                      polarity: str = "negative",
                      method: str = "pantompkins1985") -> np.ndarray:
    """R-peak detection — ORIGINAL negative-polarity algorithm.

    Assumes the device's R-peaks deflect NEGATIVE: the signal is flipped
    before detection so R-peaks appear as upward maxima for the detector.
    Detected indices are then snapped to the local maximum of the flipped
    signal within ±60 ms (Pan-Tompkins reports peak indices at the end of
    its 150 ms integration window, which lags the true apex).

    `polarity` is accepted for API compatibility but **NOT honoured here**:
    this function applies the negative-polarity pipeline to every signal so
    you can directly compare against `posiECG` recordings (which will look
    visibly broken — that is the diagnostic).

    `method` is forwarded to `neurokit2.ecg_peaks(method=...)`. The default
    "pantompkins1985" matches the historical chain; "neurokit" is the
    library's own default (more robust on noisy short-window data).
    """
    import neurokit2 as nk
    del polarity  # intentionally ignored — always run negative-polarity pipeline

    flipped = -ecg
    cleaned = nk.ecg_clean(flipped, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=method)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if len(peaks) > 3:
        # iterative=False: kubios's iterative mode recomputes outlier thresholds
        # over the *whole* recording, which gets poisoned by plank-phase motion
        # artefact and then wrongly discards legitimate post-plank R-peaks.
        _, info2 = nk.signal_fixpeaks(peaks, sampling_rate=fs, method="kubios",
                                      iterative=False)
        if isinstance(info2, dict):
            peaks = np.asarray(info2.get("clean", peaks), dtype=int)
        else:
            peaks = np.asarray(info2, dtype=int)

    # Snap to the local maximum of the flipped signal (~true R-wave apex).
    half = int(round(0.060 * fs))
    snapped: list[int] = []
    n = len(flipped)
    for p in peaks:
        lo = max(0, p - half)
        hi = min(n, p + half + 1)
        snapped.append(lo + int(np.argmax(flipped[lo:hi])))
    return np.asarray(snapped, dtype=int)


def detect_ecg_rpeaks_per_phase(
    ecg: np.ndarray, fs: float, t0: float,
    phases: dict[str, tuple[float, float]],
    *,
    method: str = "pantompkins1985",
) -> np.ndarray:
    """Phase-aware R-peak detection.

    Slice the ECG by absolute-time `phases` dict (`{name: (t_start, t_end)}`),
    run `detect_ecg_rpeaks` independently on each slice, offset the indices
    back into the whole-signal frame, and concatenate. This isolates each
    phase's detector state (`ecg_clean` adaptive baseline + Pan-Tompkins
    adaptive threshold + kubios outlier stats) so high-noise phases (e.g.
    plank) cannot poison detection in adjacent phases (e.g. recovery).

    Returns a sorted, de-duplicated array of R-peak indices into `ecg`.
    """
    n = len(ecg)
    out: list[int] = []
    seen: set[int] = set()
    for _, (s, e) in phases.items():
        i_lo = max(0, int(round((s - t0) * fs)))
        i_hi = min(n, int(round((e - t0) * fs)))
        if i_hi - i_lo < int(round(2.0 * fs)):  # < 2 s — skip
            continue
        sub = ecg[i_lo:i_hi]
        sub_peaks = detect_ecg_rpeaks(sub, fs, method=method)
        for p in sub_peaks:
            idx = int(p + i_lo)
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                out.append(idx)
    out.sort()
    return np.asarray(out, dtype=int)


def clean_nn_intervals(
    rpeaks_idx: np.ndarray, fs: float,
    *, lo_ms: float = 300.0, hi_ms: float = 1500.0,
    median_window: int = 10, median_tol: float = 0.20,
) -> np.ndarray:
    """Return the cleaned NN series in milliseconds.

    Per features.pdf step 4: reject NN < `lo_ms` or > `hi_ms`; reject any NN
    that differs from the median of the surrounding `median_window` beats by
    more than `median_tol` (20 %). Rejected NNs are cubic-spline interpolated.
    """
    if len(rpeaks_idx) < 3:
        return np.array([], dtype=float)

    nn_ms = np.diff(rpeaks_idx) / fs * 1000.0
    n = len(nn_ms)
    valid = (nn_ms >= lo_ms) & (nn_ms <= hi_ms)

    # Median-deviation rejection (sliding window of `median_window` NN, excluding self).
    half = median_window // 2
    for i in range(n):
        if not valid[i]:
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        nearby = np.concatenate([nn_ms[lo:i], nn_ms[i + 1:hi]])
        nearby = nearby[(nearby >= lo_ms) & (nearby <= hi_ms)]
        if len(nearby) == 0:
            continue
        med = float(np.median(nearby))
        if med > 0 and abs(nn_ms[i] - med) / med > median_tol:
            valid[i] = False

    if valid.sum() < 4:
        return nn_ms[valid].astype(float)

    if valid.all():
        return nn_ms.astype(float)

    # Cubic-spline interpolation across the rejected positions.
    idx = np.arange(n, dtype=float)
    try:
        cs = CubicSpline(idx[valid], nn_ms[valid], extrapolate=True)
        out = nn_ms.astype(float).copy()
        out[~valid] = cs(idx[~valid])
        # clip the interpolated values back into the physiologic range
        out = np.clip(out, lo_ms, hi_ms)
        return out
    except Exception:
        # fall back to keeping only the valid intervals
        return nn_ms[valid].astype(float)


# ---- BR (respiration) ------------------------------------------------------

# Default median-filter window sizes (seconds). Both are adaptable — pass
# different values to filter_br() to retune for slower/faster breathing.
BR_BASELINE_WINDOW_S = 8.0     # long median → slow-drift baseline (subtracted)
BR_SMOOTH_WINDOW_S = 0.5       # short median → despike / smooth


def filter_br(
    x: np.ndarray, fs: float,
    *,
    smooth_window_s: float = BR_SMOOTH_WINDOW_S,
    baseline_window_s: float = BR_BASELINE_WINDOW_S,
) -> np.ndarray:
    """BR analysis via median filtering, with an adaptable window.

    Replaces the previous moving-average + Chebyshev-II + Butterworth chain.
    Two median-filter passes:

      1. **baseline removal** — `baseline = median_filter(x, baseline_window_s)`
         captures slow respiratory-baseline wander; subtracting it removes
         drift *without ringing*. A linear high-pass filter rings around the
         step transients produced by body motion at the phase boundaries; a
         median filter does not, which is the whole point of switching.
      2. **despike / smooth** — a short `median_filter(smooth_window_s)`
         removes residual motion spikes while preserving the breath waveform
         shape (median filters are edge-preserving, unlike a moving average).

    Both window lengths are parameters; the defaults (8 s baseline, 0.5 s
    smooth) suit ~6–20 breaths/min at the 100 Hz working rate. Widen
    `baseline_window_s` for very slow breathing; widen `smooth_window_s` for
    noisier mic-coupled respiration belts.
    """
    from scipy.ndimage import median_filter

    x = x.astype(np.float64, copy=False)
    base_w = max(1, int(round(baseline_window_s * fs)))
    baseline = median_filter(x, size=base_w, mode="nearest")
    detrended = x - baseline
    smooth_w = max(1, int(round(smooth_window_s * fs)))
    return median_filter(detrended, size=smooth_w, mode="nearest")


def detect_br_peaks(br: np.ndarray, fs: float,
                    clip_mad: float = 5.0,
                    prom_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """BR peak detector following features.pdf steps 4-5.

    Implementation note: the PDF specifies a 250 ms minimum rising/falling
    slope, but applied verbatim it over-rejects on real data because of
    micro-variations in the filtered breathing signal. We use
    `scipy.signal.find_peaks` with a 1.5 s minimum inter-peak distance as the
    slope surrogate (a breath cycle ≥ 1.5 s implies rising + falling segments
    each ≥ 250 ms in any reasonable filtered signal), then apply the PDF's
    adaptive 1/3 × mean-of-last-eight-amplitudes threshold on AC = peak
    height above the V1-V2 baseline midline.

    Robustness fix: before peak finding we clip the signal to ±clip_mad·MAD
    (median absolute deviation, scaled by 1.4826 to match σ for Gaussian
    data). Without this, a single motion-artefact transient at a phase
    boundary becomes the first accepted peak, sets the adaptive threshold to
    ~10⁵, and rejects every subsequent real breath. Setting clip_mad=0
    disables clipping.

    Returns (peak_indices, peak_amplitudes).
    """
    n = len(br)
    if n < int(fs * 2):
        return np.array([], dtype=int), np.array([], dtype=float)

    if clip_mad > 0:
        med = float(np.median(br))
        mad = float(np.median(np.abs(br - med)))
        sigma_robust = 1.4826 * mad
        if sigma_robust > 1e-12:
            limit = clip_mad * sigma_robust
            br = np.clip(br, med - limit, med + limit)

    # Global prominence floor: a fraction of the *active* breathing amplitude
    # (90th percentile of |signal|, robust to the flat rest/recovery stretches
    # that dominate the median). This suppresses the dense noise peaks the
    # median-filtered signal still carries where breathing is shallow — those
    # peaks would otherwise inflate the breath rate to ~25/min. prom_frac=0
    # disables the floor.
    min_dist = max(int(round(fs * 1.5)), 1)
    if prom_frac > 0:
        active = float(np.percentile(np.abs(br - np.median(br)), 90))
        prominence = prom_frac * active if active > 1e-9 else None
    else:
        prominence = None
    cand, _ = signal.find_peaks(br, distance=min_dist, prominence=prominence)
    if len(cand) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    accepted: list[int] = []
    amps: list[float] = []
    for i, p in enumerate(cand):
        # V1 = local minimum between previous candidate and p
        left = cand[i - 1] if i > 0 else max(0, p - int(fs * 4))
        right = cand[i + 1] if i < len(cand) - 1 else min(n, p + int(fs * 4))
        v1_val = float(np.min(br[left:p])) if p > left else float(br[p])
        v2_val = float(np.min(br[p + 1:right])) if (p + 1) < right else float(br[p])

        ac = float(br[p]) - 0.5 * (v1_val + v2_val)
        if ac <= 0:
            continue

        threshold = (float(np.mean(amps[-8:])) / 3.0) if len(amps) > 0 else 0.0
        if ac < threshold:
            continue

        accepted.append(int(p))
        amps.append(ac)

    return np.asarray(accepted, dtype=int), np.asarray(amps, dtype=float)


# ---- Microphone / PCG ------------------------------------------------------

def filter_mic_pcg(x: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass 20-200 Hz for heart sound extraction (features.pdf step 1 of CSI)."""
    nyq = 0.5 * fs
    high = min(200.0, nyq * 0.95)
    sos = signal.butter(4, [20.0, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def shannon_energy(x: np.ndarray, smooth_ms: float = 50.0, fs: float = 2000.0) -> np.ndarray:
    """Shannon-energy envelope: SE = -x^2 log(x^2), with smoothing.

    PDF wrote "S = C log(C)"; the standard PCG envelope (Liang et al. 1997)
    that yields the clear S1/S2 peaks the figure shows is the Shannon energy
    of the normalized signal, -x^2 log(x^2). We follow the standard.
    """
    x = x.astype(np.float64, copy=False)
    m = float(np.max(np.abs(x)))
    if m < 1e-12:
        return np.zeros_like(x)
    xn = x / m
    x2 = xn * xn
    x2 = np.where(x2 < 1e-12, 1e-12, x2)
    se = -x2 * np.log(x2)
    win = max(1, int(round(smooth_ms / 1000.0 * fs)))
    if win > 1:
        se = np.convolve(se, np.ones(win) / win, mode="same")
    return se


def detect_pcg_peaks(se: np.ndarray, fs: float) -> np.ndarray:
    """All cardiac-sound peaks in the Shannon-energy envelope."""
    # heart sounds are ~10/sec at most -> min distance 80 ms; prominence
    # adapted to the signal scale.
    std = float(np.std(se))
    if std < 1e-12:
        return np.array([], dtype=int)
    min_dist = max(int(round(0.08 * fs)), 1)
    peaks, _ = signal.find_peaks(se, distance=min_dist, prominence=0.2 * std)
    return peaks


def classify_s1_s2(
    se: np.ndarray, se_fs: float, pcg_peaks: np.ndarray,
    rpeaks_se_idx: np.ndarray,
    *,
    s1_window_s: tuple[float, float] = (0.0, 0.20),
    s2_window_s: tuple[float, float] = (0.20, 0.50),
) -> tuple[np.ndarray, np.ndarray]:
    """For each R-peak, pick the PCG peak closest to expected S1 / S2 timing.

    Returns (s1_amplitudes, s2_amplitudes) in Shannon-energy units, in
    chronological order. R-peaks without a matching S1 or S2 within the
    timing windows are skipped.
    """
    if len(pcg_peaks) == 0 or len(rpeaks_se_idx) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    s1_lo, s1_hi = int(round(s1_window_s[0] * se_fs)), int(round(s1_window_s[1] * se_fs))
    s2_lo, s2_hi = int(round(s2_window_s[0] * se_fs)), int(round(s2_window_s[1] * se_fs))

    s1_amps: list[float] = []
    s2_amps: list[float] = []
    for r in rpeaks_se_idx:
        s1_band = pcg_peaks[(pcg_peaks > r + s1_lo) & (pcg_peaks <= r + s1_hi)]
        s2_band = pcg_peaks[(pcg_peaks > r + s2_lo) & (pcg_peaks <= r + s2_hi)]
        s1_idx = s1_band[np.argmax(se[s1_band])] if len(s1_band) else None
        s2_idx = s2_band[np.argmax(se[s2_band])] if len(s2_band) else None
        if s1_idx is not None and s2_idx is not None:
            s1_amps.append(float(se[s1_idx]))
            s2_amps.append(float(se[s2_idx]))
    return np.asarray(s1_amps, dtype=float), np.asarray(s2_amps, dtype=float)


# ---- back-compat shims (kept so old imports don't break callers we missed) -

def filter_cps_cardiac(x: np.ndarray, fs: float) -> np.ndarray:  # pragma: no cover
    """Deprecated — features.pdf replaces the CPS cardiac-band branch with
    Shannon-energy CSI. Returns a 0.8-3 Hz bandpass for any straggling callers."""
    sos = signal.butter(4, [0.8, 3.0], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def filter_cps_resp(x: np.ndarray, fs: float) -> np.ndarray:  # pragma: no cover
    """Deprecated — see filter_cps_cardiac. Returns a 0.1-0.5 Hz bandpass."""
    sos = signal.butter(4, [0.1, 0.5], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


# ---- diagnostics -----------------------------------------------------------

def sanity_report(name: str, fs: float, x: np.ndarray, *, flat_window_s: float = 2.0) -> dict:
    n = len(x)
    n_nan = int(np.isnan(x).sum())
    dur = n / fs
    win = max(5, int(flat_window_s * fs))
    if n >= win:
        s = np.convolve(np.abs(np.diff(x, prepend=x[0])), np.ones(win) / win, mode="same")
        flat_frac = float((s < 1e-6).mean())
    else:
        flat_frac = 0.0
    rng = (x.max() - x.min()) if n else 0.0
    sat_frac = 0.0
    if rng > 0:
        lo_thr = x.min() + 0.001 * rng
        hi_thr = x.max() - 0.001 * rng
        sat_frac = float(((x <= lo_thr) | (x >= hi_thr)).mean())
    return dict(name=name, n=n, fs=fs, duration_s=dur,
                n_nan=n_nan, flat_frac=flat_frac, sat_frac=sat_frac)
