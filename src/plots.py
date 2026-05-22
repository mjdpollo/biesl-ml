"""Diagnostic plots called out in the pipeline spec.

(a) raw vs filtered signal
(b) detected peaks overlaid
(c) confusion matrix
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .features import preprocess_recording
from .io import load_recording, list_recordings, channel_fs, phase_boundaries


def plot_raw_vs_filtered(rec_path: str, out_path: str, window_s: float = 10.0,
                         **pp_kwargs):
    rec = load_recording(rec_path)
    pp = preprocess_recording(rec, **pp_kwargs)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)

    # ECG: first `window_s` of filtered + R peaks (signal flipped to put R-peaks UP)
    fs = pp.fs_ecg
    n = int(window_s * fs)
    ax = axes[0]
    raw_ecg = rec.channels["ecg"][1, :n]
    raw_t = rec.channels["ecg"][0, :n] - rec.channels["ecg"][0, 0]
    filt_t = np.arange(n) / fs
    ax.plot(raw_t, -(raw_ecg - np.median(raw_ecg)), color="0.7", lw=0.5,
            label="raw (flipped, detrended)")
    ax2 = ax.twinx()
    ax2.plot(filt_t, -pp.ecg[:n], color="C0", lw=0.9, label="filtered 0.5-40 Hz (flipped)")
    rpks = pp.rpeaks[pp.rpeaks < n]
    ax2.plot(rpks / fs, -pp.ecg[rpks], "rv", ms=7, label="R peaks")
    ax.set_title(f"ECG — {rec.name} (R-peaks negative in raw → flipped)")
    ax.set_xlabel("s"); ax.set_ylabel("raw counts (detrended)")
    ax2.set_ylabel("filtered (z-units)")
    l1, lbl1 = ax.get_legend_handles_labels()
    l2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lbl1 + lbl2, loc="upper right", fontsize=8)

    # BR — 30 s window
    fs = pp.fs_br
    n = int(window_s * 3 * fs)
    ax = axes[1]
    raw_n = int(window_s * 3 * channel_fs(rec.channels["br"]))
    raw_br = rec.channels["br"][1, :raw_n]
    raw_t = rec.channels["br"][0, :raw_n] - rec.channels["br"][0, 0]
    filt_t = np.arange(n) / fs
    ax.plot(raw_t, (raw_br - raw_br.mean()) / (raw_br.std() or 1.0),
            color="0.6", lw=0.6, label="raw (z-scored)")
    ax.plot(filt_t, pp.br[:n], color="C2", lw=0.9, label="filtered 0.1-0.5 Hz")
    bpks = pp.br_peaks[pp.br_peaks < n]
    ax.plot(bpks / fs, pp.br[bpks], "rv", ms=4, label="breath peaks")
    ax.set_title("Breathing")
    ax.set_xlabel("s"); ax.set_ylabel("z-units"); ax.legend(loc="upper right", fontsize=8)

    # PCG Shannon-energy envelope (replaces old CPS bands)
    fs = pp.fs_mic
    n = int(window_s * fs)
    ax = axes[2]
    ax.plot(np.arange(n) / fs, pp.se[:n], color="C3", lw=0.7,
            label="PCG Shannon energy (20-150 Hz)")
    pcg_in = pp.pcg_peaks[pp.pcg_peaks < n]
    if len(pcg_in):
        ax.plot(pcg_in / fs, pp.se[pcg_in], "rv", ms=4,
                label=f"S1/S2 peaks (n={len(pcg_in)})")
    ax.set_title("Mic — PCG Shannon-energy envelope")
    ax.set_xlabel("s"); ax.set_ylabel("envelope"); ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_br_full(rec_path: str, out_path: str, **pp_kwargs) -> None:
    """Plot the full-duration filtered BR signal with detected breath peaks.

    Uses the exact preprocessed signal + peaks that feed the feature extractor:
        BR resampled to 25 Hz -> robust-zscore -> 0.1-0.5 Hz BP + Sav-Gol ->
        neurokit2 rsp_peaks (biosppy backend, ±0.2*std fallback).
    Phase regions (rest / stress / recovery) are shaded for context.
    `pp_kwargs` is forwarded to preprocess_recording (e.g. phase_aware=True).
    """
    rec = load_recording(rec_path)
    pp = preprocess_recording(rec, **pp_kwargs)

    fs = pp.fs_br
    t0 = pp.br_t0
    t = t0 + np.arange(len(pp.br)) / fs
    peaks = pp.br_peaks

    fig, ax = plt.subplots(figsize=(14, 3.5))
    phases = phase_boundaries(rec)
    phase_colors = {"rest": "#e5f0ff", "stress": "#ffe1d6", "recovery": "#e9f5e3"}
    for ph, (s, e) in phases.items():
        ax.axvspan(s, e, color=phase_colors[ph], alpha=0.6, zorder=0)
        ax.text((s + e) / 2, 0.97, ph, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="0.3")

    ax.plot(t, pp.br, color="C2", lw=0.7, zorder=1, label="BR filtered")
    if len(peaks):
        ax.plot(t0 + peaks / fs, pp.br[peaks], "rv", ms=4, zorder=2,
                label=f"breath peaks (n={len(peaks)})")

    # Auto-zoom y-axis to the 2-98 percentile of the signal so motion-artefact
    # transients go off-chart instead of crushing the breath waveform onto
    # zero. Peak markers above/below that range will also crop — that's fine,
    # the count + rate in the title come from detection on the full signal.
    if len(pp.br):
        lo, hi = np.percentile(pp.br, [2.0, 98.0])
        pad = 0.25 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)

    dur = rec.duration_s
    rate = 60.0 * len(peaks) / dur if dur > 0 else 0
    ax.set_title(f"BR — {rec.name}    duration={dur:.0f}s   peaks={len(peaks)}   "
                 f"avg rate={rate:.1f} breaths/min")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (filtered)")
    ax.set_xlim(0, dur)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_ecg_full(rec_path: str, out_path: str,
                  t_start: float | None = None, t_end: float | None = None,
                  **pp_kwargs) -> None:
    """Plot the full-duration ECG (negated so R-peaks point up) with detected
    R-peaks overlaid. Uses the exact preprocessing + detection that the
    feature extractor uses. Phase regions are shaded for context.
    `pp_kwargs` is forwarded to preprocess_recording (e.g. phase_aware=True).
    """
    rec = load_recording(rec_path)
    pp = preprocess_recording(rec, **pp_kwargs)

    fs = pp.fs_ecg
    t0 = pp.ecg_t0
    # Always show the flipped signal (negative-polarity algorithm view), so
    # negative- and positive-polarity recordings can be compared under the
    # SAME algorithm. For posiECG inputs the R-peaks will look like downward
    # spikes on this view — that is the diagnostic, not a bug.
    polarity = getattr(rec, "ecg_polarity", "negative")
    sig = -pp.ecg
    t = t0 + np.arange(len(sig)) / fs
    peaks = pp.rpeaks

    dur = rec.duration_s
    xlo = 0.0 if t_start is None else float(t_start)
    xhi = dur if t_end is None else float(t_end)
    # peak times for the window, in seconds
    peak_t = (t0 + peaks / fs) if len(peaks) else np.array([])
    peak_mask = (peak_t >= xlo) & (peak_t <= xhi)
    peaks_in = peaks[peak_mask]
    peak_t_in = peak_t[peak_mask]
    # sample slice for the window (used for axis autozoom only)
    i_lo = max(0, int((xlo - t0) * fs))
    i_hi = min(len(sig), int((xhi - t0) * fs) + 1)
    sig_in = sig[i_lo:i_hi]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    phases = phase_boundaries(rec)
    phase_colors = {"rest": "#e5f0ff", "stress": "#ffe1d6", "recovery": "#e9f5e3"}
    for ph, (s, e) in phases.items():
        ax.axvspan(s, e, color=phase_colors[ph], alpha=0.6, zorder=0)
        ax.text((s + e) / 2, 0.97, ph, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="0.3")

    sig_label = ("ECG filtered (flipped — negative-polarity algorithm view)"
                 + ("  [posiECG: expect inverted]" if polarity == "positive" else ""))
    ax.plot(t, sig, color="C0", lw=0.6, zorder=1, label=sig_label)
    if len(peaks_in):
        ax.plot(peak_t_in, sig[peaks_in], "rv", ms=6, zorder=2,
                label=f"R peaks (n={len(peaks_in)} in window)")

    # zoom y-axis to the bulk of the visible window (so artefacts outside the
    # window don't drag the limits)
    if len(sig_in):
        lo, hi = np.percentile(sig_in, [1.0, 99.0])
        pad = 0.3 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)

    rate = 60.0 * len(peaks) / dur if dur > 0 else 0
    polarity_tag = "  [posiECG]" if polarity == "positive" else ""
    win_tag = f"   window {xlo:.0f}-{xhi:.0f} s" if (t_start is not None or t_end is not None) else ""
    ax.set_title(f"ECG — {rec.name}{polarity_tag}    duration={dur:.0f}s   "
                 f"R peaks total={len(peaks)}   avg HR={rate:.1f} bpm{win_tag}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (filtered)")
    ax.set_xlim(xlo, xhi)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list[str], out_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def make_all(data_dir: str = "data", out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    files = list_recordings(data_dir)
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0].replace(" ", "_").replace("'", "_")
        try:
            plot_raw_vs_filtered(f, os.path.join(out_dir, f"signals_{basename}.png"))
            print(f"  signals_{basename}.png")
        except Exception as e:
            print(f"  skipped (signals) {f}: {e}")
        try:
            plot_br_full(f, os.path.join(out_dir, f"br_full_{basename}.png"))
            print(f"  br_full_{basename}.png")
        except Exception as e:
            print(f"  skipped (br_full) {f}: {e}")
        try:
            plot_ecg_full(f, os.path.join(out_dir, f"ecg_full_{basename}.png"))
            print(f"  ecg_full_{basename}.png")
        except Exception as e:
            print(f"  skipped (ecg_full) {f}: {e}")

    # confusion matrix from LOSO results — one per model
    results_path = os.path.join(out_dir, "loso_results.json")
    if os.path.exists(results_path):
        with open(results_path) as fh:
            r = json.load(fh)
        classes = r["classes"]
        # Support new (multi-model) and legacy (single-model) layouts.
        results = r.get("results")
        if results is None:
            results = {"knn": {
                "confusion_total": r["confusion_total"],
                "mean_accuracy":   r["mean_accuracy"],
                "mean_macro_f1":   r["mean_macro_f1"],
            }}
        for name, rr in results.items():
            cm = np.array(rr["confusion_total"])
            out_path = os.path.join(out_dir, f"confusion_loso_{name}.png")
            plot_confusion(cm, classes, out_path,
                           title=f"{name}  LOSO  acc={rr['mean_accuracy']:.2f}  "
                                 f"macro-F1={rr['mean_macro_f1']:.2f}")
            print(f"  confusion_loso_{name}.png")


if __name__ == "__main__":
    make_all()
