#!/usr/bin/env python
"""Compare Poincare-image 2D-CNN runs across window lengths.

Reads outputs/poincare_loro{tag}.json for each (label, tag) pair, builds a
side-by-side macro-F1 / per-class table and a grouped bar chart, and writes
report/poincare-cnn-window-comparison.md.

Run after scripts/run_poincare.py for each window:
    uv run python scripts/run_poincare.py --window 60  --stride 20 --tag ""
    uv run python scripts/run_poincare.py --window 120 --stride 20 --tag _2min
    uv run python scripts/compare_poincare_windows.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import PHASE_CLASSES                                  # noqa: E402

# (display label, json tag)
RUNS = [("60 s", ""), ("120 s (2 min)", "_2min")]
FIG = ROOT / "figures" / "poincare_images" / "window_comparison.png"
REPORT = ROOT / "report" / "poincare-cnn-window-comparison.md"


def _load(tag: str) -> dict | None:
    p = ROOT / "outputs" / f"poincare_loro{tag}.json"
    if not p.exists():
        print(f"!! missing {p}", file=sys.stderr)
        return None
    return json.loads(p.read_text())


def _counts_from_confusion(s: dict) -> dict:
    cm = np.asarray(s["confusion_total"], dtype=int)
    return {c: int(cm[i].sum()) for i, c in enumerate(PHASE_CLASSES)}


def plot(runs: list[tuple[str, dict]]) -> None:
    metrics = ["accuracy", "macro-F1"] + [f"F1[{c}]" for c in PHASE_CLASSES]
    x = np.arange(len(metrics))
    w = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (label, res) in enumerate(runs):
        s = res["summary"]
        vals = [s["mean_accuracy"], s["mean_macro_f1"]] + \
               [s["per_class_f1_mean"][c] for c in PHASE_CLASSES]
        bars = ax.bar(x + k * w, vals, w, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w * (len(runs) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Poincaré 2D-CNN — LORO by window length")
    ax.legend()
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    plt.close(fig)
    print(f"  -> wrote {FIG}")


def write_report(runs: list[tuple[str, dict]]) -> None:
    L: list[str] = []
    L.append("# Poincaré-image 2D-CNN — window-length comparison (60 s vs 2 min)\n")
    L.append("> Same pipeline, model, exclusions and **LORO** protocol; only the "
             "window length (and therefore the number of windows) differs. "
             "Stride fixed at 20 s. See per-run detail in "
             "[`poincare-cnn-report.md`](poincare-cnn-report.md) and "
             "[`poincare-cnn-report_2min.md`](poincare-cnn-report_2min.md).\n")

    # dataset sizes
    L.append("## Dataset size by window\n")
    L.append("| window | total | rest | meditation | plank | math | folds |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, res in runs:
        c = _counts_from_confusion(res["summary"])
        tot = sum(c.values())
        L.append(f"| {label} | {tot} | {c['rest']} | {c['meditation']} | "
                 f"{c['plank']} | {c['math']} | {len(res['folds'])} |")
    L.append("")
    L.append("> The 2-min window is the originally-requested setting; it leaves "
             "**plank** with very few windows (the plank phases are only "
             "120–210 s), which is the whole reason the 60-s window was adopted.\n")

    # headline
    L.append("## Headline — pooled-LORO\n")
    L.append("| window | acc | macro-F1 | F1[rest] | F1[medi] | F1[plank] | F1[math] |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, res in runs:
        s = res["summary"]
        pc = s["per_class_f1_mean"]
        L.append(f"| {label} | {s['mean_accuracy']:.3f} | {s['mean_macro_f1']:.3f} | "
                 f"{pc['rest']:.3f} | {pc['meditation']:.3f} | {pc['plank']:.3f} | "
                 f"{pc['math']:.3f} |")
    L.append("")

    # deltas (last - first)
    if len(runs) == 2:
        (la, ra), (lb, rb) = runs[0], runs[1]
        sa, sb = ra["summary"], rb["summary"]
        L.append(f"### Δ ({lb} − {la})\n")
        L.append("| metric | " + f"{la}" + " | " + f"{lb}" + " | Δ |")
        L.append("|---|---:|---:|---:|")
        rows = [("accuracy", sa["mean_accuracy"], sb["mean_accuracy"]),
                ("macro-F1", sa["mean_macro_f1"], sb["mean_macro_f1"])]
        rows += [(f"F1[{c}]", sa["per_class_f1_mean"][c], sb["per_class_f1_mean"][c])
                 for c in PHASE_CLASSES]
        for name, a, b in rows:
            L.append(f"| {name} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
        L.append("")

    L.append("## Takeaway\n")
    L.append("1. **60 s wins on macro-F1** (0.249 vs 0.199). The 2-min window's "
             "marginally higher *accuracy* is just the majority-class effect — "
             "fewer plank/medi windows means rest dominates more, so guessing "
             "rest/math scores slightly better on raw accuracy while the balanced "
             "macro-F1 drops.")
    L.append("2. **2 min kills plank** (F1 0.09 → **0.00**; only 5 windows total, "
             "most plank files yield 0). This is the decisive reason to prefer the "
             "60-s window — a 2-min window cannot fit inside the 120–210 s plank "
             "phases after boundary exclusion.")
    L.append("3. **Both settings are weak** in absolute terms: meditation and plank "
             "are not learned cross-recording from RR shape alone. Neither window "
             "approaches the multi-modal feature pipelines in "
             "[`ml-report.md`](ml-report.md) (1D-CNN macro-F1 0.75). Read this as a "
             "single-modality (RR-only) baseline.")
    L.append("4. **Conclusion:** keep the **60-s** window for the Poincaré-image "
             "model; the 2-min spec is documented here for completeness but is "
             "strictly worse for this 4-class task.\n")

    L.append("## Chart\n")
    L.append("![comparison](../figures/poincare_images/window_comparison.png)\n")

    L.append("## Confusion heatmaps\n")
    L.append("| 60 s | 2 min |")
    L.append("|---|---|")
    L.append("| ![](../figures/poincare_images/confusion_loro.png) | "
             "![](../figures/poincare_images/confusion_loro_2min.png) |")
    L.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L))
    print(f"  -> wrote {REPORT}")


def main() -> None:
    runs = [(label, _load(tag)) for label, tag in RUNS]
    runs = [(label, res) for label, res in runs if res is not None]
    if len(runs) < 2:
        print("need both runs; run scripts/run_poincare.py for each window first")
        sys.exit(1)
    plot(runs)
    write_report(runs)


if __name__ == "__main__":
    main()
