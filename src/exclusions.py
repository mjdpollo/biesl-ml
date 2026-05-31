"""Hardware / data-quality exclusion rules for the running notes.

* Full-recording exclusions are handled by physically moving the files to
  `data/_excluded/` — the pipeline never sees them.
* This module records the *partial* exclusions that have to be applied at the
  window level after build (`build_feature_table()` /  `local_raw_windows()`).

The two filter helpers can be applied uniformly to the classical feature
DataFrame and to the list of CNN raw windows.
"""
from __future__ import annotations

from typing import Iterable


# (rec_name, kind, value)
# kind = "t_lt"          → drop windows whose t_start < value
# kind = "drop_activity" → drop windows whose activity equals value
PARTIAL_EXCLUSIONS: tuple[tuple[str, str, object], ...] = (
    # 0-150 s contains motion artefact / sensor settling
    ("mta_5_21_medi",            "t_lt",          150.0),
    # 0-140 s contains motion artefact / sensor settling
    ("mta_5_26_math_8_12",       "t_lt",          140.0),
    # plank phase has a hardware glitch; rest period is fine
    ("oyj_5_22_pla_2'15_posiECG", "drop_activity", "plank"),
    ("oyj_5_22_pla_1'50_posiECG", "drop_activity", "plank"),
    # rest phase noisy; math phase is fine
    ("tnq_5_29_math_7_12",        "drop_activity", "rest"),
)


def filter_feature_df(df):
    """Apply PARTIAL_EXCLUSIONS to a `build_feature_table()` output."""
    if df is None or len(df) == 0:
        return df
    out = df
    for rec, kind, val in PARTIAL_EXCLUSIONS:
        mask = out["rec_name"] == rec
        if not mask.any():
            continue
        if kind == "t_lt":
            drop = mask & (out["t_start"] < float(val))
        elif kind == "drop_activity":
            drop = mask & (out["activity"] == val)
        else:
            raise ValueError(f"unknown exclusion kind: {kind}")
        out = out[~drop]
    return out.reset_index(drop=True)


def filter_raw_windows(windows: Iterable):
    """Apply PARTIAL_EXCLUSIONS to a list of `RawWindow`."""
    rules = PARTIAL_EXCLUSIONS

    def keep(w) -> bool:
        for rec, kind, val in rules:
            if w.rec_name != rec:
                continue
            if kind == "t_lt" and w.t_start < float(val):
                return False
            if kind == "drop_activity" and w.activity == val:
                return False
        return True

    return [w for w in windows if keep(w)]
