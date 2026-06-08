#!/usr/bin/env python
"""Write the index README for the self-contained Poincaré-CNN report bundle.

All reports and figures are generated directly into `report/poincare-cnn/`
(by run_poincare.py and compare_poincare_windows.py), with local `figures/...`
links — so this script only emits the index README that ties them together.

Run after the experiments + comparison:
    uv run python scripts/run_poincare.py --window 60  --stride 20 --tag ""
    uv run python scripts/run_poincare.py --window 120 --stride 20 --tag _2min
    uv run python scripts/compare_poincare_windows.py
    uv run python scripts/bundle_poincare_report.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "report" / "poincare-cnn"


def main() -> None:
    (BUNDLE / "figures").mkdir(parents=True, exist_ok=True)
    expected = [
        "poincare-cnn-report.md", "poincare-cnn-report_2min.md",
        "poincare-cnn-window-comparison.md",
        "figures/window_comparison.png",
        "figures/confusion_loro.png", "figures/confusion_loro_2min.png",
        "figures/samples_by_class.png", "figures/samples_by_class_2min.png",
    ]
    for rel in expected:
        if not (BUNDLE / rel).exists():
            print(f"  !! missing {BUNDLE / rel} — run the experiment scripts first")

    index = [
        "# Poincaré-image 2D-CNN — report bundle\n",
        "Self-contained, downloadable bundle: ECG RR (NN) Poincaré plots → "
        "64×64 log-count images → small 2D-CNN, evaluated with "
        "leave-one-recording-out (LORO). All figures are local to this folder.\n",
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
    print(f"  -> wrote {BUNDLE / 'README.md'}")


if __name__ == "__main__":
    main()
