#!/usr/bin/env python
"""Render Poincaré scatter plots from outputs/preprocessed_nn.json.

For every recording, draw a 1×2 figure: (rest | stressor) Poincaré scatter
plus a per-panel SD1/SD2 ellipse and label.

Also draws one aggregate figure per stressor (medi/pla/math) on the same
axes scale so cross-subject patterns are visible.

Run after scripts/dump_preprocessed_nn.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import _poincare                                 # noqa: E402

IN_PATH = ROOT / "outputs" / "preprocessed_nn.json"
OUT_DIR = ROOT / "figures" / "poincare"
PHASE_COLORS = {"rest": "#3a86ff", "meditation": "#43aa8b",
                "plank": "#f3722c", "math": "#d62828"}


def _ellipse(ax, nn_ms, color):
    nn = np.asarray(nn_ms, dtype=float)
    if len(nn) < 3:
        return float("nan"), float("nan")
    poin = _poincare(nn)
    sd1, sd2 = poin["sd1"], poin["sd2"]
    if not (np.isfinite(sd1) and np.isfinite(sd2)):
        return sd1, sd2
    mean = float(np.mean(nn))
    # Poincaré ellipse: axes along ±45° (identity / anti-identity lines).
    ell = patches.Ellipse(
        (mean, mean), width=2 * sd2, height=2 * sd1, angle=45.0,
        facecolor="none", edgecolor=color, linewidth=1.8, alpha=0.9,
    )
    ax.add_patch(ell)
    return sd1, sd2


def plot_recording(rec_name: str, info: dict, out_dir: Path) -> None:
    phases = info["phases"]
    activities = [a for a in ("rest", "meditation", "plank", "math") if a in phases]
    if len(activities) != 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, act in zip(axes, activities):
        nn = np.asarray(phases[act]["nn_ms"], dtype=float)
        if len(nn) < 4:
            ax.set_title(f"{act}\n(too few NN)")
            continue
        color = PHASE_COLORS.get(act, "#666666")
        ax.scatter(nn[:-1], nn[1:], s=12, alpha=0.45, color=color, edgecolor="none")
        sd1, sd2 = _ellipse(ax, nn, color)
        ax.plot([nn.min(), nn.max()], [nn.min(), nn.max()],
                color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("NNᵢ (ms)")
        ax.set_ylabel("NNᵢ₊₁ (ms)")
        ax.set_title(f"{act}    SD1={sd1:.1f} ms · SD2={sd2:.1f} ms · "
                     f"SS={1000.0 / sd2:.1f}" if np.isfinite(sd2) and sd2 > 0
                     else f"{act}")
        ax.set_aspect("equal")
    fig.suptitle(rec_name, fontsize=11)
    fig.tight_layout()
    out = out_dir / f"{rec_name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(records: dict, out_dir: Path) -> None:
    by_stressor: dict[str, dict] = {}
    for rec_name, info in records.items():
        st = info["stressor"]
        if st not in by_stressor:
            by_stressor[st] = {"rest": [], "stress": []}
        for act, blk in info["phases"].items():
            target = "rest" if act == "rest" else "stress"
            by_stressor[st][target].extend(blk["nn_ms"])
    for st, blk in by_stressor.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        # consistent axes across the two panels
        all_nn = np.concatenate([np.asarray(blk["rest"]), np.asarray(blk["stress"])])
        if len(all_nn) < 4:
            continue
        lo, hi = np.percentile(all_nn, 1), np.percentile(all_nn, 99)
        for ax, phase_name in zip(axes, ("rest", "stress")):
            nn = np.asarray(blk[phase_name], dtype=float)
            color = PHASE_COLORS["rest"] if phase_name == "rest" else PHASE_COLORS.get(
                {"medi": "meditation", "pla": "plank", "math": "math"}.get(st, "math"),
                "#666666",
            )
            ax.scatter(nn[:-1], nn[1:], s=4, alpha=0.18, color=color, edgecolor="none")
            sd1, sd2 = _ellipse(ax, nn, color)
            ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("NNᵢ (ms)")
            ax.set_ylabel("NNᵢ₊₁ (ms)")
            ax.set_title(f"{st} · {phase_name}    "
                         f"SD1={sd1:.1f} · SD2={sd2:.1f}" if np.isfinite(sd2)
                         else f"{st} · {phase_name}")
            ax.set_aspect("equal")
        fig.suptitle(f"Aggregate Poincaré — stressor: {st}", fontsize=11)
        fig.tight_layout()
        out = out_dir / f"_aggregate__{st}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  aggregate {st}: rest={len(blk['rest'])}  stress={len(blk['stress'])}")


def main() -> None:
    if not IN_PATH.exists():
        print(f"missing {IN_PATH}; run scripts/dump_preprocessed_nn.py first")
        sys.exit(1)
    records = json.loads(IN_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rec_name, info in sorted(records.items()):
        plot_recording(rec_name, info, OUT_DIR)
        print(f"  {rec_name}")
    plot_aggregate(records, OUT_DIR)
    print(f"\nwrote {len(list(OUT_DIR.glob('*.png')))} PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
