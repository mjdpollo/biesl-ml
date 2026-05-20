#!/usr/bin/env python
"""Print and plot confusion matrices for all (model × protocol × feature-config)
combinations from the four result JSONs in outputs/.

Outputs to stdout (markdown tables) and to outputs/confusion_*.png.

Usage:
    uv run python scripts/show_confusion_matrices.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from src.pipeline import PHASE_CLASSES   # noqa: E402


OUT = Path("outputs")
FIG_DIR = Path("figures") / "confusion"   # tracked in git (outputs/ is gitignored)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"!! missing: {path}", file=sys.stderr)
        return None
    with open(path) as fh:
        return json.load(fh)


def _sum_seed_confusions(seed_results: dict, *, dl: bool, model: str | None = None) -> np.ndarray:
    """Sum confusion matrices across the 5 random-split seeds."""
    total = np.zeros((len(PHASE_CLASSES), len(PHASE_CLASSES)), dtype=int)
    for seed, payload in seed_results.items():
        if dl:
            cm = np.asarray(payload["test"]["confusion"], dtype=int)
        else:
            cm = np.asarray(payload["models"][model]["test"]["confusion"], dtype=int)
        total += cm
    return total


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix to percentages (true-class recall view).

    Each row sums to 100 %. Rows with zero support stay all-zero.
    """
    cm = cm.astype(np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    safe = np.where(rs == 0, 1.0, rs)
    return cm / safe * 100.0


def _print_markdown(name: str, cm: np.ndarray, n: int | None = None) -> None:
    """Print a row-normalized (percent) confusion matrix as a markdown table.

    Each row sums to 100 % (row-wise recall view). A `support` column shows
    the number of true samples per class — essential context when `stress`
    has only 4 windows across the whole experiment.
    """
    cm_pct = _row_normalize(cm)
    supports = cm.sum(axis=1)
    print(f"\n#### {name}" + (f"  *(n_test={n})*" if n is not None else ""))
    head = "| true \\\\ pred | " + " | ".join(PHASE_CLASSES) + " | support |"
    sep = "|---|" + "|".join("---" for _ in PHASE_CLASSES) + "|---|"
    print(head)
    print(sep)
    for i, c in enumerate(PHASE_CLASSES):
        cells = " | ".join(f"{cm_pct[i, j]:.1f}%" for j in range(cm_pct.shape[1]))
        print(f"| **{c}** | {cells} | {int(supports[i])} |")


def _save_png(name: str, cm: np.ndarray, slug: str) -> None:
    """Save a row-normalized (percent) heatmap of one confusion matrix.

    Each row sums to 100 % (true-class recall view). The colour scale is
    fixed at 0-100 % so heatmaps are directly comparable across (model ×
    protocol × config) — a row that's "all blue" means perfect recall on
    that class, regardless of how many samples it contained. True-class
    support is annotated on the y-axis tick labels.
    """
    cm_pct = _row_normalize(cm)
    supports = cm.sum(axis=1).astype(int)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm_pct, cmap="Blues", aspect="equal", vmin=0, vmax=100)
    ax.set_xticks(range(len(PHASE_CLASSES)))
    ax.set_yticks(range(len(PHASE_CLASSES)))
    ax.set_xticklabels(PHASE_CLASSES)
    # Annotate y-tick labels with the per-row support count so a viewer can
    # tell whether a row is meaningful (105 baseline samples) or near-empty
    # (4 stress samples).
    ax.set_yticklabels(
        [f"{c}\n(n={n})" for c, n in zip(PHASE_CLASSES, supports)]
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(name, fontsize=9)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            v = cm_pct[i, j]
            # white on dark cells (>= 50 %), black on light
            color = "white" if v >= 50 else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color=color, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("% of true class")
    plt.tight_layout()
    path = FIG_DIR / f"{slug}.png"
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  -> {path}", file=sys.stderr)


# ---- protocol 1: LORO ------------------------------------------------------

def render_loro() -> None:
    print("\n## LORO confusion matrices (sum across 7 folds)\n")

    cls = _load(OUT / "local_loro_temp_ablation.json")
    if cls is not None:
        for cfg_key, cfg_lbl in (("pdf_only", "PDF features only"),
                                 ("with_temp", "PDF + temperature")):
            print(f"\n### Classical — {cfg_lbl}")
            for model in ("knn", "randomforest", "xgboost"):
                cm = np.asarray(cls[cfg_key]["summary"][model]["confusion_total"], dtype=int)
                title = f"{model.upper():<14s}  LORO  {cfg_lbl}"
                slug = f"loro__classical_{model}_{cfg_key}"
                _print_markdown(f"{model.upper()} — LORO — {cfg_lbl}", cm, int(cm.sum()))
                _save_png(title, cm, slug)

    dl = _load(OUT / "dl_local_loro_temp_ablation.json")
    if dl is not None:
        print(f"\n### 1D-CNN — LORO")
        for cfg_key, cfg_lbl in (("pdf_only", "PDF channels (3 ch)"),
                                 ("with_temp", "PDF + temperature (4 ch)")):
            cm = np.asarray(dl[cfg_key]["summary"]["confusion_total"], dtype=int)
            title = f"1D-CNN          LORO  {cfg_lbl}"
            slug = f"loro__cnn_{cfg_key}"
            _print_markdown(f"1D-CNN — LORO — {cfg_lbl}", cm, int(cm.sum()))
            _save_png(title, cm, slug)


# ---- protocol 2: random 70:15:15 ------------------------------------------

def render_random_split() -> None:
    print("\n\n## Random-split confusion matrices (sum across 5 seeds)\n")

    cls = _load(OUT / "local_randomsplit_temp_ablation.json")
    if cls is not None:
        for cfg_key, cfg_lbl in (("pdf_only", "PDF features only"),
                                 ("with_temp", "PDF + temperature")):
            print(f"\n### Classical — {cfg_lbl}")
            for model in ("knn", "randomforest", "xgboost"):
                cm = _sum_seed_confusions(cls[cfg_key]["per_seed"], dl=False, model=model)
                title = f"{model.upper():<14s}  random  {cfg_lbl}"
                slug = f"randomsplit__classical_{model}_{cfg_key}"
                _print_markdown(f"{model.upper()} — random 70:15:15 — {cfg_lbl}", cm, int(cm.sum()))
                _save_png(title, cm, slug)

    dl = _load(OUT / "dl_local_randomsplit_temp_ablation.json")
    if dl is not None:
        print(f"\n### 1D-CNN — random 70:15:15")
        for cfg_key, cfg_lbl in (("pdf_only", "PDF channels (3 ch)"),
                                 ("with_temp", "PDF + temperature (4 ch)")):
            cm = _sum_seed_confusions(dl[cfg_key]["per_seed"], dl=True)
            title = f"1D-CNN          random  {cfg_lbl}"
            slug = f"randomsplit__cnn_{cfg_key}"
            _print_markdown(f"1D-CNN — random 70:15:15 — {cfg_lbl}", cm, int(cm.sum()))
            _save_png(title, cm, slug)


def main() -> None:
    render_loro()
    render_random_split()
    print(f"\nAll PNGs saved under {FIG_DIR}/", file=sys.stderr)


if __name__ == "__main__":
    main()
