#!/usr/bin/env python
"""Compare three BR peak detectors on every recording, split into phases.

For each recording in data/:
  * filter BR via the median-filter chain (src.preprocess.filter_br)
  * run THREE peak detectors:
      - "global"   : the current detector with a single global p90 prominence
                     floor (src.preprocess.detect_br_peaks).
      - "sliding"  : sliding-window detector with a LOCAL p90 floor per
                     window (src.preprocess.detect_br_peaks_sliding).
      - "neurokit" : neurokit2's nk.rsp_peaks (biosppy method).
  * split each detector's peaks into the three protocol phases
    (rest 0-5 min, stress 5 - 5+stressor min, recovery to end of recording)
    and tabulate breath count + mean RR (bpm) per phase.
  * plot the median-filtered BR signal with each method's peaks on a stacked
    3-row panel, with phase boundaries marked.

Outputs:
  figures/br_compare/<rec>.png           one comparison figure per recording
  outputs/br_detector_compare.json       per-recording, per-phase counts/rates

Usage:
    uv run python scripts/compare_br_detectors.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402

from src.io import load_recording, list_recordings, phase_boundaries, resample_uniform   # noqa: E402
from src.preprocess import (              # noqa: E402
    BR_BASELINE_WINDOW_S, BR_SMOOTH_WINDOW_S,
    detect_br_peaks, detect_br_peaks_sliding, detect_br_peaks_neurokit,
    filter_br,
)

warnings.filterwarnings("ignore")
FIG_DIR = Path("figures") / "br_compare"
FIG_DIR.mkdir(parents=True, exist_ok=True)
BR_FS = 100.0
METHODS = ("global", "sliding", "neurokit")
METHOD_COLORS = {"global": "C3", "sliding": "C1", "neurokit": "C2"}


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("'", "_").replace("(", "_").replace(")", "_")


def _detect(method: str, filt: np.ndarray) -> np.ndarray:
    if method == "global":
        peaks, _ = detect_br_peaks(filt, BR_FS)
        return peaks
    if method == "sliding":
        return detect_br_peaks_sliding(filt, BR_FS)
    if method == "neurokit":
        return detect_br_peaks_neurokit(filt, BR_FS)
    raise ValueError(method)


def _per_phase_stats(peaks_idx: np.ndarray, fs: float, phases: dict, t0: float) -> dict:
    """Return {phase: {n_peaks, rate_bpm}}.

    `t0` is the absolute time of sample 0 (the BR channel's first timestamp).
    """
    out = {}
    if len(peaks_idx) == 0:
        return {ph: dict(n_peaks=0, rate_bpm=float("nan")) for ph in phases}
    t_peaks = t0 + peaks_idx / fs                       # absolute seconds
    for ph, (a, b) in phases.items():
        mask = (t_peaks >= a) & (t_peaks < b)
        n = int(mask.sum())
        if n >= 2:
            sub = t_peaks[mask]
            iv = np.diff(sub)
            iv = iv[(iv > 1.0) & (iv < 12.0)]
            rate = float(60.0 / np.mean(iv)) if len(iv) else float("nan")
        else:
            rate = float("nan")
        out[ph] = dict(n_peaks=n, rate_bpm=rate, phase_s=float(b - a))
    return out


def _plot_compare(rec_name: str, t: np.ndarray, filt: np.ndarray,
                  results: dict, phases: dict, out_path: Path,
                  ecg_polarity: str, stressor: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True)
    for ax, method in zip(axes, METHODS):
        peaks = results[method]["peaks"]
        n_total = len(peaks)
        ax.plot(t, filt, color="0.30", lw=0.6)
        if n_total:
            ax.plot(t[peaks], filt[peaks], "v", color=METHOD_COLORS[method], ms=4,
                    label=f"{method}  (n={n_total})")
        # phase boundaries + shading
        for ph, (a, b) in phases.items():
            ax.axvspan(a, b, color="gray", alpha=0.06 if ph == "rest" else 0.10 if ph == "stress" else 0.04, lw=0)
            ax.axvline(a, color="k", ls="--", lw=0.5, alpha=0.4)
        per = results[method]["per_phase"]
        ph_str = "  ".join(f"{ph}: n={per[ph]['n_peaks']} ({per[ph]['rate_bpm']:.1f}/min)" for ph in phases)
        ax.set_ylabel(f"{method}\nBR (filt)", fontsize=9)
        ax.set_title(f"{method}   {ph_str}", fontsize=8, loc="left")
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_title(f"{rec_name}   stressor={stressor}   ecg={ecg_polarity}   "
                      f"(median win: base {BR_BASELINE_WINDOW_S}s / smooth {BR_SMOOTH_WINDOW_S}s)",
                      fontsize=10, loc="center")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def analyze_recording(path: str) -> dict:
    rec = load_recording(path)
    raw = resample_uniform(rec.channels["br"], BR_FS).astype(np.float64)
    filt = filter_br(raw, BR_FS)
    t0 = float(rec.channels["br"][0, 0])
    t = t0 + np.arange(len(filt)) / BR_FS
    phases = phase_boundaries(rec)

    results = {}
    for m in METHODS:
        peaks = _detect(m, filt)
        results[m] = dict(
            peaks=peaks,
            per_phase=_per_phase_stats(peaks, BR_FS, phases, t0),
            n_total=int(len(peaks)),
        )

    out_path = FIG_DIR / f"{_slug(rec.name)}.png"
    _plot_compare(rec.name, t, filt, results, phases, out_path,
                  rec.ecg_polarity, rec.stressor)

    return dict(
        name=rec.name, slug=_slug(rec.name), stressor=rec.stressor,
        ecg_polarity=rec.ecg_polarity, duration_s=float(rec.duration_s),
        phases={ph: dict(start=float(a), end=float(b)) for ph, (a, b) in phases.items()},
        per_method={m: dict(n_total=results[m]["n_total"],
                            per_phase=results[m]["per_phase"]) for m in METHODS},
    )


def main() -> None:
    files = list_recordings("data")
    print(f"Comparing 3 detectors on {len(files)} recordings\n")
    all_rows = []
    for p in files:
        try:
            row = analyze_recording(p)
            all_rows.append(row)
            g = row["per_method"]["global"]["n_total"]
            s = row["per_method"]["sliding"]["n_total"]
            k = row["per_method"]["neurokit"]["n_total"]
            print(f"  {row['name']:34s}  global={g:3d}  sliding={s:3d}  neurokit={k:3d}")
        except Exception as e:
            print(f"  !! {p}: {e}", file=sys.stderr)
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/br_detector_compare.json", "w") as fh:
        json.dump(all_rows, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
    print(f"\nSaved {len(all_rows)} comparison plots under {FIG_DIR}/")
    print(f"Wrote outputs/br_detector_compare.json")


if __name__ == "__main__":
    main()
