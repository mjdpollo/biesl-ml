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


def _print_markdown(name: str, cm: np.ndarray, n: int | None = None) -> None:
    """Print a confusion matrix as a markdown table on stdout."""
    print(f"\n#### {name}" + (f"  *(n_test={n})*" if n is not None else ""))
    head = "| true \\\\ pred | " + " | ".join(PHASE_CLASSES) + " |"
    sep = "|---|" + "|".join("---" for _ in PHASE_CLASSES) + "|"
    print(head)
    print(sep)
    for i, c in enumerate(PHASE_CLASSES):
        row = " | ".join(str(int(v)) for v in cm[i])
        print(f"| **{c}** | {row} |")


def _save_png(name: str, cm: np.ndarray, slug: str) -> None:
    """Save a heatmap PNG of one confusion matrix."""
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(PHASE_CLASSES)))
    ax.set_yticks(range(len(PHASE_CLASSES)))
    ax.set_xticklabels(PHASE_CLASSES)
    ax.set_yticklabels(PHASE_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(name, fontsize=9)
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.045)
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
