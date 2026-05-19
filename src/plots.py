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
from .io import load_recording, list_recordings, channel_fs


def plot_raw_vs_filtered(rec_path: str, out_path: str, window_s: float = 10.0):
    rec = load_recording(rec_path)
    pp = preprocess_recording(rec)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)

    # ECG: first `window_s` of filtered + R peaks (signal flipped to put R-peaks UP)
    fs = pp.fs["ecg"]
    n = int(window_s * fs)
    ax = axes[0]
    raw_ecg = rec.channels["ecg"][1, :n]
    raw_t = rec.channels["ecg"][0, :n] - rec.channels["ecg"][0, 0]
    filt_t = np.arange(n) / fs
    # plot raw on a twin axis so its big-amplitude QRS doesn't dwarf the filtered trace
    ax.plot(raw_t, -(raw_ecg - np.median(raw_ecg)), color="0.7", lw=0.5,
            label="raw (flipped, detrended)")
    ax2 = ax.twinx()
    ax2.plot(filt_t, -pp.ecg[:n], color="C0", lw=0.9, label="filtered 0.5-40 Hz (flipped)")
    rpks = pp.rpeaks[pp.rpeaks < n]
    ax2.plot(rpks / fs, -pp.ecg[rpks], "rv", ms=7, label="R peaks")
    ax.set_title(f"ECG — {rec.name} (R-peaks negative in raw → flipped)")
    ax.set_xlabel("s"); ax.set_ylabel("raw counts (detrended)")
    ax2.set_ylabel("filtered (z-units)")
    # combine legends
    l1, lbl1 = ax.get_legend_handles_labels()
    l2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lbl1 + lbl2, loc="upper right", fontsize=8)

    # BR
    fs = pp.fs["br"]
    n = int(window_s * 3 * fs)        # 30 s for breathing
    ax = axes[1]
    raw_br = rec.channels["br"][1, :int(window_s * 3 * channel_fs(rec.channels["br"]))]
    raw_t = rec.channels["br"][0, :len(raw_br)] - rec.channels["br"][0, 0]
    filt_t = np.arange(n) / fs
    ax.plot(raw_t, (raw_br - raw_br.mean()) / (raw_br.std() or 1.0),
            color="0.6", lw=0.6, label="raw (z-scored)")
    ax.plot(filt_t, pp.br[:n], color="C2", lw=0.9, label="filtered 0.1-0.5 Hz")
    bpks = pp.br_peaks[pp.br_peaks < n]
    ax.plot(bpks / fs, pp.br[bpks], "rv", ms=4, label="breath peaks")
    ax.set_title("Breathing")
    ax.set_xlabel("s"); ax.set_ylabel("z-units"); ax.legend(loc="upper right", fontsize=8)

    # CPS resp branch
    fs = pp.cps_fs
    n = int(window_s * 3 * fs)
    ax = axes[2]
    ax.plot(np.arange(n) / fs, pp.cps_card[:n], color="C3", lw=0.7, label="cps cardiac 0.8-3 Hz")
    ax.plot(np.arange(n) / fs, pp.cps_resp[:n], color="C4", lw=0.9, label="cps resp 0.1-0.5 Hz")
    ax.set_title("CPS (mic) bands")
    ax.set_xlabel("s"); ax.set_ylabel("z-units"); ax.legend(loc="upper right", fontsize=8)

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
    # one medi + one pla, if present
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0].replace(" ", "_").replace("'", "_")
        try:
            plot_raw_vs_filtered(f, os.path.join(out_dir, f"signals_{basename}.png"))
            print(f"  signals_{basename}.png")
        except Exception as e:
            print(f"  skipped {f}: {e}")

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
