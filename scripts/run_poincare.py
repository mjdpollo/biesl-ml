#!/usr/bin/env python
"""End-to-end Poincare-image 2D-CNN experiment.

  1. build 64x64 log-count Poincare images (60-s window, 20-s stride)
  2. save a sample montage per class to figures/poincare_images/
  3. LORO 2D-CNN training on CUDA
  4. write report/poincare-cnn-report.md

Run:  uv run python scripts/run_poincare.py
"""
from __future__ import annotations

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
    BINS, RANGE_MS, STRIDE_S, WINDOW_S, LABEL_NAMES, build_and_cache,
)
from src.poincare_train import CACHE, main as train_main               # noqa: E402

FIG_DIR = ROOT / "figures" / "poincare_images"
REPORT = ROOT / "report" / "poincare-cnn-report.md"


def plot_samples(data: dict, n_per_class: int = 6) -> None:
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
                 f"log(1+count), per-image max)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "samples_by_class.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out}")


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    safe = np.where(rs == 0, 1.0, rs)
    return cm / safe * 100.0


def save_confusion_png(cm: np.ndarray, slug: str = "confusion_loro",
                       title: str = "Poincare 2D-CNN  LORO") -> Path:
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
    path = FIG_DIR / f"{slug}.png"
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  -> wrote {path}")
    return path


def _per_class_pr(report: dict, c: str) -> tuple[float, float]:
    blk = report.get(c, {})
    return float(blk.get("precision", 0.0)), float(blk.get("recall", 0.0))


def write_report(result: dict, data: dict) -> None:
    s = result["summary"]
    y = data["y"]
    rec_names = data["rec_names"]
    counts = {LABEL_NAMES[k]: int(v) for k, v in sorted(Counter(y.tolist()).items())}
    n_rec = len(set(rec_names.tolist()))
    n_subj = len(set(s.split("_")[0] for s in rec_names.tolist()))
    cm = np.asarray(s["confusion_total"], dtype=int)
    cm_pct = _row_normalize(cm)

    L: list[str] = []
    L.append("# Poincaré-image 2D-CNN — stress/activity classification\n")
    L.append("> 4 classes: **rest / meditation / plank / math**. ECG RR (NN) "
             "Poincaré plots rendered as 64×64 log-count **images** and classified "
             "with a small 2D-CNN.")
    L.append("> Companion to the feature-based [`ml-report.md`](ml-report.md) and the "
             "Poincaré diagnostics in [`poincare-report.md`](poincare-report.md).\n")

    L.append("## What this run is\n")
    L.append("Instead of feeding scalar Poincaré descriptors (SD1, SD2, …) to a "
             "classifier, each window's RR series is turned into a **2-D Poincaré "
             "image** and a convolutional net learns directly from the scatter "
             "shape. One image per window, one window every 20 s.\n")

    L.append("## Setup\n")
    L.append(f"- **Image.** x = RRₙ, y = RRₙ₊₁; range {RANGE_MS[0]:.0f}–{RANGE_MS[1]:.0f} ms; "
             f"**{BINS}×{BINS}** bins; value = **log(1+count)**; **per-image max** "
             "normalization. Single channel (64×64×1).")
    L.append(f"- **Windowing.** {WINDOW_S:.0f}-s window, **{STRIDE_S:.0f}-s stride**, each "
             "window fully inside one phase (no cross-phase mixing).")
    L.append("- **Boundary exclusion.** Windows overlapping the 5-min cue "
             "**[290, 310] s** or the 10-min mark **[590, 610] s** are dropped. "
             "Recovery phase dropped (matches the rest of the project).")
    L.append("- **Partial exclusions (curator review).** `smj_6_6_math_17` removed "
             "entirely; `oyj_6_6_math_11` rest phase removed (math kept).")
    L.append("- **RR source.** Same pipeline as the classical features — wavelet "
             "5–45 Hz ECG filter → neurokit R-peaks → NN cleaning "
             "(300–1500 ms reject + 20 % median-deviation reject + cubic-spline "
             "interpolation).")
    L.append("- **Model.**")
    L.append("")
    L.append("  ```")
    L.append("  Input 64×64×1")
    L.append("  Conv2D(16,3×3,same) → BN → ReLU → MaxPool2×2     # 64→32")
    L.append("  Conv2D(32,3×3,same) → BN → ReLU → MaxPool2×2     # 32→16")
    L.append("  Conv2D(64,3×3,same) → BN → ReLU → GlobalAvgPool  # → 64")
    L.append("  Dense(64) → ReLU → Dropout(0.3) → Output(4)")
    L.append("  ```")
    L.append("")
    L.append("  ~28 k parameters. Class-weighted cross-entropy, AdamW "
             "(lr 1e-3, wd 1e-4), cosine schedule, mild additive-noise augment, "
             "AMP on **CUDA (RTX 5090)**, early stopping on inner-val macro-F1.")
    L.append("- **Protocol — LORO** (leave-one-recording-out); one further recording "
             "held out per fold as the inner validation set.\n")

    L.append("## Dataset\n")
    L.append(f"- **{len(y)} windows**, **{n_rec} recordings**, {n_subj} subjects "
             f"(LORO = {len(result['folds'])} folds).")
    L.append("- Class counts:\n")
    L.append("  | rest | meditation | plank | math | total |")
    L.append("  |---:|---:|---:|---:|---:|")
    L.append(f"  | {counts.get('rest',0)} | {counts.get('meditation',0)} | "
             f"{counts.get('plank',0)} | {counts.get('math',0)} | {len(y)} |")
    L.append("")
    L.append("  > The class set is heavily imbalanced (rest ≫ plank). The 60-s "
             "window (vs the originally-requested 2 min) is what keeps plank "
             "trainable at all — at 2 min the plank phases (120–210 s) yield "
             "~6 windows total.\n")

    L.append("## Headline — pooled-LORO\n")
    L.append("| Model | acc | macro-F1 | F1[rest] | F1[medi] | F1[plank] | F1[math] |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    pc = s["per_class_f1_mean"]
    L.append(f"| **Poincaré 2D-CNN** | {s['mean_accuracy']:.3f} | "
             f"{s['mean_macro_f1']:.3f} | {pc['rest']:.3f} | {pc['meditation']:.3f} | "
             f"{pc['plank']:.3f} | {pc['math']:.3f} |")
    L.append("")
    L.append(f"Accuracy **{s['mean_accuracy']:.3f} ± {s['std_accuracy']:.3f}**, "
             f"macro-F1 **{s['mean_macro_f1']:.3f} ± {s['std_macro_f1']:.3f}** "
             "(mean ± std across folds).\n")

    L.append("## Confusion matrix (LORO, summed across folds)\n")
    L.append("![confusion](../figures/poincare_images/confusion_loro.png)\n")
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

    L.append("## Reading the result\n")
    L.append("1. **rest and math carry the score**; **meditation and plank are "
             "essentially not learned** cross-recording (medi F1 "
             f"{pc['meditation']:.2f}, plank F1 {pc['plank']:.2f}).")
    L.append("2. The confusion matrix shows the failure mode: **meditation is "
             "absorbed into rest/math** and **plank is absorbed into math** — "
             "from RR shape alone the net cannot separate the minority stress "
             "classes from the majority ones across unseen subjects.")
    L.append("3. Causes: (a) strong **class imbalance** (rest 216 vs plank 23); "
             "(b) **LORO is hard** — Poincaré shape has large per-subject baseline "
             "spread, so some folds collapse (e.g. `nvt_5_21_medi` F1 0.0); "
             "(c) **single modality** — only RR, vs the 8-feature / multi-channel "
             "pipelines in `ml-report.md`.")
    L.append("4. For reference, the feature-based 1D-CNN in `ml-report.md` reaches "
             "macro-F1 0.75 on the same recordings using ECG+Resp+Mic — the "
             "Poincaré-image-only model is well below that and should be read as a "
             "**single-modality baseline**, not a replacement.\n")

    L.append("## Reproduce\n")
    L.append("```bash")
    L.append("uv run python scripts/run_poincare.py")
    L.append("```")
    L.append("")
    L.append("Outputs:\n")
    L.append("| File | Contents |")
    L.append("|---|---|")
    L.append("| `outputs/poincare_dataset.npz` | stacked 64×64 images + labels/meta |")
    L.append("| `outputs/poincare_loro.json` | per-fold + summary, confusion matrix |")
    L.append("| `figures/poincare_images/samples_by_class.png` | sample images per class |")
    L.append("| `figures/poincare_images/confusion_loro.png` | LORO confusion heatmap |")
    L.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L))
    print(f"  -> wrote {REPORT}")


def main() -> None:
    data = build_and_cache(cache_path=CACHE, norm="per_image")
    plot_samples(data)
    result = train_main(rebuild=False)        # reuse the cache just built
    save_confusion_png(np.asarray(result["summary"]["confusion_total"], dtype=int))
    write_report(result, data)


if __name__ == "__main__":
    main()
