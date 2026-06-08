#!/usr/bin/env python
"""Assemble a self-contained, downloadable Poincaré-CNN report bundle.

Gathers the three Poincaré-image reports and every figure into a single
folder `report/poincare-cnn/` with local (same-folder) image links, plus an
index README. Download that one folder and everything renders offline.

Run after the experiments + comparison:
    uv run python scripts/run_poincare.py --window 60  --stride 20 --tag ""
    uv run python scripts/run_poincare.py --window 120 --stride 20 --tag _2min
    uv run python scripts/compare_poincare_windows.py
    uv run python scripts/bundle_poincare_report.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_FIG = ROOT / "figures" / "poincare_images"
BUNDLE = ROOT / "report" / "poincare-cnn"
BUNDLE_FIG = BUNDLE / "figures"

REPORTS = [
    "poincare-cnn-report.md",
    "poincare-cnn-report_2min.md",
    "poincare-cnn-window-comparison.md",
]
FIGURES = [
    "confusion_loro.png",
    "confusion_loro_2min.png",
    "samples_by_class.png",
    "samples_by_class_2min.png",
    "window_comparison.png",
]


def _rewrite_links(text: str) -> str:
    # local figures live in ./figures/ next to the markdown
    text = text.replace("../figures/poincare_images/", "figures/")
    # cross-references to reports that stay in the parent report/ dir
    text = text.replace("(ml-report.md)", "(../ml-report.md)")
    text = text.replace("(poincare-report.md)", "(../poincare-report.md)")
    return text


def main() -> None:
    BUNDLE_FIG.mkdir(parents=True, exist_ok=True)

    for fig in FIGURES:
        src = SRC_FIG / fig
        if src.exists():
            shutil.copy2(src, BUNDLE_FIG / fig)
            print(f"  fig  {fig}")
        else:
            print(f"  !! missing figure {src}")

    for rep in REPORTS:
        src = ROOT / "report" / rep
        if src.exists():
            (BUNDLE / rep).write_text(_rewrite_links(src.read_text()))
            print(f"  doc  {rep}")
        else:
            print(f"  !! missing report {src}")

    index = [
        "# Poincaré-image 2D-CNN — report bundle\n",
        "Self-contained bundle: ECG RR (NN) Poincaré plots → 64×64 log-count "
        "images → small 2D-CNN, evaluated with leave-one-recording-out (LORO).\n",
        "## Contents\n",
        "| Document | What |",
        "|---|---|",
        "| [poincare-cnn-window-comparison.md](poincare-cnn-window-comparison.md) "
        "| **Start here** — 60 s vs 2 min head-to-head + takeaway |",
        "| [poincare-cnn-report.md](poincare-cnn-report.md) | Full 60-s run "
        "(recommended setting) |",
        "| [poincare-cnn-report_2min.md](poincare-cnn-report_2min.md) | Full 2-min "
        "run (original spec; plank collapses) |",
        "\n## Headline (LORO macro-F1)\n",
        "| window | windows | acc | macro-F1 | F1[plank] |",
        "|---|---:|---:|---:|---:|",
        "| **60 s** | 382 | 0.562 | **0.249** | 0.090 |",
        "| 2 min | 271 | 0.571 | 0.199 | 0.000 |",
        "\n60 s is the better setting; the 2-min window leaves plank with only "
        "5 windows. See the comparison doc for details.\n",
        "## Figures\n",
        "![comparison](figures/window_comparison.png)\n",
        "| 60 s confusion | 2 min confusion |",
        "|---|---|",
        "| ![](figures/confusion_loro.png) | ![](figures/confusion_loro_2min.png) |",
        "\n### Sample Poincaré images (60 s)\n",
        "![samples](figures/samples_by_class.png)\n",
    ]
    (BUNDLE / "README.md").write_text("\n".join(index))
    print(f"  idx  README.md")
    print(f"\n  -> bundle ready: {BUNDLE}")


if __name__ == "__main__":
    main()
