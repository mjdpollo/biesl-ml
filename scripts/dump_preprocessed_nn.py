#!/usr/bin/env python
"""Dump per-recording × per-phase NN-interval series + BR-peak intervals to
outputs/preprocessed_nn.json so we can render Poincaré plots and ad-hoc
diagnostics without rerunning the full preprocessing pipeline.

Output schema:
    {
      "<rec_name>": {
        "subject": "<subject>",
        "stressor": "medi" | "pla" | "math",
        "phases": {
          "rest":       {"nn_ms": [...], "br_intervals_s": [...]},
          "<stress>":   {"nn_ms": [...], "br_intervals_s": [...]}
          # 'stress' phase name is the same string assign_activity uses
          # (meditation / plank / math).
        }
      },
      ...
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exclusions import PARTIAL_EXCLUSIONS                          # noqa: E402
from src.features import preprocess_recording                          # noqa: E402
from src.io import list_recordings, load_recording, phase_boundaries    # noqa: E402
from src.pipeline import assign_activity                                # noqa: E402
from src.preprocess import clean_nn_intervals                           # noqa: E402

OUT_PATH = ROOT / "outputs" / "preprocessed_nn.json"


def _partial_drop_phase(rec_name: str, phase_label: str) -> bool:
    """Apply src.exclusions PARTIAL_EXCLUSIONS at the dump level so the Poincaré
    plot matches what the classifier sees."""
    for r, kind, val in PARTIAL_EXCLUSIONS:
        if r != rec_name:
            continue
        if kind == "drop_activity" and val == phase_label:
            return True
    return False


def dump() -> dict:
    out: dict[str, dict] = {}
    for path in list_recordings(str(ROOT / "data")):
        rec = load_recording(path)
        try:
            pp = preprocess_recording(rec)
        except Exception as e:                  # noqa: BLE001
            print(f"  skipped {rec.name}: {e}")
            continue
        phases = phase_boundaries(rec)
        per_phase: dict[str, dict] = {}
        for phase_name, (p_start, p_end) in phases.items():
            if phase_name == "recovery":
                continue
            activity = assign_activity(phase_name, rec.stressor)
            if _partial_drop_phase(rec.name, activity):
                continue
            # NN intervals — slice R-peaks to the phase, clean, store ms.
            t_abs = pp.ecg_t0 + pp.rpeaks / pp.fs_ecg
            mask = (t_abs >= p_start) & (t_abs < p_end)
            nn_ms = clean_nn_intervals(pp.rpeaks[mask], pp.fs_ecg)
            # BR breath intervals (seconds) within the phase.
            t_br = pp.br_t0 + pp.br_peaks / pp.fs_br
            mb = (t_br >= p_start) & (t_br < p_end)
            br_peaks_phase = pp.br_peaks[mb]
            iv_br = np.diff(br_peaks_phase) / pp.fs_br if len(br_peaks_phase) >= 2 else np.empty(0)
            iv_br = iv_br[(iv_br > 1.0) & (iv_br < 12.0)]
            per_phase[activity] = {
                "nn_ms": [float(x) for x in nn_ms],
                "br_intervals_s": [float(x) for x in iv_br],
            }
        out[rec.name] = {
            "subject": rec.subject,
            "stressor": rec.stressor,
            "phases": per_phase,
        }
        n_rest = len(per_phase.get("rest", {}).get("nn_ms", []))
        stress_act = next((a for a in per_phase if a != "rest"), None)
        n_stress = len(per_phase.get(stress_act, {}).get("nn_ms", [])) if stress_act else 0
        print(f"  {rec.name:40s}  rest={n_rest:4d} NN  {stress_act or '-'}={n_stress:4d} NN")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out))
    print(f"\nwrote {OUT_PATH}  ({len(out)} recordings)")
    return out


if __name__ == "__main__":
    dump()
