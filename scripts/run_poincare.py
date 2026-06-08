#!/usr/bin/env python
"""End-to-end Poincare-image 2D-CNN experiment.

  1. build 64x64 log-count Poincare images (window/stride configurable)
  2. save a sample montage per class to figures/poincare_images/
  3. LORO 2D-CNN training on CUDA
  4. write report/poincare-cnn-report{tag}.md + confusion heatmap

Run:
    uv run python scripts/run_poincare.py                       # 60-s window
    uv run python scripts/run_poincare.py --window 120 --tag _2min
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import PHASE_CLASSES                                  # noqa: E402
from src.poincare_images import (                                      # noqa: E402
    BINS, RANGE_MS, LABEL_NAMES, build_and_cache,
)
from src.poincare_train import (                                       # noqa: E402
    EPOCHS, _json_default, run_loro,
)

# Everything for this experiment lives in one self-contained, downloadable
# folder under report/. Reports use local `figures/...` links.
BUNDLE = ROOT / "report" / "poincare-cnn"
FIG_DIR = BUNDLE / "figures"


def plot_samples(data: dict, *, window_s: float, stride_s: float,
                 out_path: Path, n_per_class: int = 6) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, y = data["X"], data["y"]
    fig, axes = plt.subplots(len(PHASE_CLASSES), n_per_class,
                             figsize=(2 * n_per_class, 2 * len(PHASE_CLASSES)))
    for r, cls in enumerate(PHASE_CLASSES):
        lbl = LABEL_NAMES.index(cls)
        idx = np.where(y == lbl)[0]
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False) if len(idx) else []
        for c in range(n_per_class):
            ax = axes[r, c] if len(PHASE_CLASSES) > 1 else axes[c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(pick):
                ax.imshow(X[pick[c], 0], origin="lower", aspect="equal",
                          cmap="magma", extent=[RANGE_MS[0], RANGE_MS[1],
                                                RANGE_MS[0], RANGE_MS[1]])
            if c == 0:
                ax.set_ylabel(cls, fontsize=11)
    fig.suptitle(f"Poincare images  ({BINS}x{BINS}, {RANGE_MS[0]:.0f}-{RANGE_MS[1]:.0f} ms, "
                 f"log(1+count), per-image max)  —  window={window_s:.0f}s stride={stride_s:.0f}s",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out_path}")


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    safe = np.where(rs == 0, 1.0, rs)
    return cm / safe * 100.0


def save_confusion_png(cm: np.ndarray, out_path: Path, title: str) -> None:
    """Row-normalized confusion heatmap, matching scripts/show_confusion_matrices.py."""
    cm_pct = _row_normalize(cm)
    supports = cm.sum(axis=1).astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm_pct, cmap="Blues", aspect="equal", vmin=0, vmax=100)
    ax.set_xticks(range(len(PHASE_CLASSES)))
    ax.set_yticks(range(len(PHASE_CLASSES)))
    ax.set_xticklabels(PHASE_CLASSES, rotation=30, ha="right")
    ax.set_yticklabels([f"{c}\n(n={n})" for c, n in zip(PHASE_CLASSES, supports)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            v = cm_pct[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color="white" if v >= 50 else "black", fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("% of true class")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


def write_report(result: dict, data: dict, *, window_s: float, stride_s: float,
                 report_path: Path, samples_png: str, confusion_png: str) -> None:
    s = result["summary"]
    y = data["y"]
    rec_names = data["rec_names"]
    counts = {LABEL_NAMES[k]: int(v) for k, v in sorted(Counter(y.tolist()).items())}
    n_rec = len(set(rec_names.tolist()))
    n_subj = len(set(r.split("_")[0] for r in rec_names.tolist()))
    cm = np.asarray(s["confusion_total"], dtype=int)
    cm_pct = _row_normalize(cm)

    L: list[str] = []
    L.append(f"# Poincaré-image 2D-CNN — {window_s:.0f}s window\n")
    L.append("> 4 classes: **rest / meditation / plank / math**. ECG RR (NN) "
             "Poincaré plots rendered as 64×64 log-count **images** and classified "
             "with a small 2D-CNN.")
    L.append("> Companion to the feature-based [`ml-report.md`](../ml-report.md), the "
             "60s-vs-2min comparison in "
             "[`poincare-cnn-window-comparison.md`](poincare-cnn-window-comparison.md), "
             "and the Poincaré diagnostics in [`poincare-report.md`](../poincare-report.md).\n")

    L.append("## Setup\n")
    L.append(f"- **Image.** x = RRₙ, y = RRₙ₊₁; range {RANGE_MS[0]:.0f}–{RANGE_MS[1]:.0f} ms; "
             f"**{BINS}×{BINS}** bins; value = **log(1+count)**; **per-image max** "
             "normalization. Single channel (64×64×1).")
    L.append(f"- **Windowing.** **{window_s:.0f}-s window, {stride_s:.0f}-s stride**, each "
             "window fully inside one phase (no cross-phase mixing).")
    L.append("- **Boundary exclusion.** Windows overlapping the 5-min cue "
             "**[290, 310] s** or the 10-min mark **[590, 610] s** are dropped. "
             "Recovery phase dropped.")
    L.append("- **Partial exclusions (curator review).** `smj_6_6_math_17` removed "
             "entirely; `oyj_6_6_math_11` rest phase removed (math kept).")
    L.append("- **RR source.** Wavelet 5–45 Hz ECG filter → neurokit R-peaks → NN "
             "cleaning (300–1500 ms reject + 20 % median-deviation reject + "
             "cubic-spline interpolation).")
    L.append("- **Model.** Conv2D(16,3×3,same)→BN→ReLU→MaxPool · "
             "Conv2D(32,…)→BN→ReLU→MaxPool · Conv2D(64,…)→BN→ReLU→GAP · "
             "Dense(64)→ReLU→Dropout(0.3)→Output(4). ~28 k params.")
    L.append("- **Training.** Class-weighted CE, AdamW (lr 1e-3, wd 1e-4), cosine "
             "schedule, AMP on **CUDA (RTX 5090)**, early stopping on inner-val "
             "macro-F1.")
    L.append("- **Protocol — LORO** (leave-one-recording-out).\n")

    L.append("## Dataset\n")
    L.append(f"- **{len(y)} windows**, **{n_rec} recordings**, {n_subj} subjects "
             f"(LORO = {len(result['folds'])} folds).")
    L.append("- Class counts:\n")
    L.append("  | rest | meditation | plank | math | total |")
    L.append("  |---:|---:|---:|---:|---:|")
    L.append(f"  | {counts.get('rest',0)} | {counts.get('meditation',0)} | "
             f"{counts.get('plank',0)} | {counts.get('math',0)} | {len(y)} |\n")

    L.append("## Headline — pooled-LORO\n")
    L.append("| Model | acc | macro-F1 | F1[rest] | F1[medi] | F1[plank] | F1[math] |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    pc = s["per_class_f1_mean"]
    L.append(f"| **Poincaré 2D-CNN ({window_s:.0f}s)** | {s['mean_accuracy']:.3f} | "
             f"{s['mean_macro_f1']:.3f} | {pc['rest']:.3f} | {pc['meditation']:.3f} | "
             f"{pc['plank']:.3f} | {pc['math']:.3f} |")
    L.append("")
    L.append(f"Accuracy **{s['mean_accuracy']:.3f} ± {s['std_accuracy']:.3f}**, "
             f"macro-F1 **{s['mean_macro_f1']:.3f} ± {s['std_macro_f1']:.3f}** "
             "(mean ± std across folds).\n")

    L.append("## Confusion matrix (LORO, summed across folds)\n")
    L.append(f"![confusion](figures/{confusion_png})\n")
    L.append("Row-normalized (rows = true class, % of that class):\n")
    L.append("| true \\ pred | " + " | ".join(PHASE_CLASSES) + " | support |")
    L.append("|---|" + "|".join("---:" for _ in PHASE_CLASSES) + "|---:|")
    for i, c in enumerate(PHASE_CLASSES):
        cells = " | ".join(f"{cm_pct[i, j]:.1f}%" for j in range(cm.shape[1]))
        L.append(f"| **{c}** | {cells} | {int(cm[i].sum())} |")
    L.append("")
    L.append("Raw counts:\n")
    L.append("| true \\ pred | " + " | ".join(PHASE_CLASSES) + " |")
    L.append("|---|" + "|".join("---:" for _ in PHASE_CLASSES) + "|")
    for i, c in enumerate(PHASE_CLASSES):
        L.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |")
    L.append("")

    L.append("## Per-fold results\n")
    L.append("| recording | test_n | macro-F1 | acc |")
    L.append("|---|---:|---:|---:|")
    for f in result["folds"]:
        L.append(f"| `{f['recording']}` | {f['test_n']} | "
                 f"{f['macro_f1']:.3f} | {f['accuracy']:.3f} |")
    L.append("")

    L.append("## Samples\n")
    L.append(f"![samples](figures/{samples_png})\n")

    L.append("## Reproduce\n")
    L.append("```bash")
    if abs(window_s - 60.0) < 1e-6:
        L.append("uv run python scripts/run_poincare.py")
    else:
        L.append(f"uv run python scripts/run_poincare.py --window {window_s:.0f} "
                 f"--stride {stride_s:.0f} --tag _{window_s/60:.0f}min")
    L.append("```")
    L.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L))
    print(f"  -> wrote {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=float, default=60.0, help="window length (s)")
    ap.add_argument("--stride", type=float, default=20.0, help="stride (s)")
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _2min)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    tag = args.tag
    cache = ROOT / "outputs" / f"poincare_dataset{tag}.npz"
    json_path = ROOT / "outputs" / f"poincare_loro{tag}.json"
    report_path = BUNDLE / f"poincare-cnn-report{tag}.md"
    samples_png = f"samples_by_class{tag}.png"
    confusion_png = f"confusion_loro{tag}.png"

    data = build_and_cache(cache_path=str(cache), norm="per_image",
                           window_s=args.window, stride_s=args.stride)
    plot_samples(data, window_s=args.window, stride_s=args.stride,
                 out_path=FIG_DIR / samples_png)

    result = run_loro(data, epochs=args.epochs)
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    print(f"  -> wrote {json_path}")

    save_confusion_png(np.asarray(result["summary"]["confusion_total"], dtype=int),
                       FIG_DIR / confusion_png,
                       title=f"Poincare 2D-CNN  LORO  ({args.window:.0f}s)")
    write_report(result, data, window_s=args.window, stride_s=args.stride,
                 report_path=report_path, samples_png=samples_png,
                 confusion_png=confusion_png)


if __name__ == "__main__":
    main()
