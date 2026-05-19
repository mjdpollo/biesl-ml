"""Three-way comparison: local-only vs WESAD-only vs (WESAD + local) -> local test.

Test set is ALWAYS local data, held out one recording at a time
(leave-one-recording-out). Reports per-condition: accuracy, macro-F1,
per-class F1, confusion matrix.

Conditions:
    A) local-only        train on local \\ {R}                test on R
    A_all) local-only-with-mic train on local \\ {R}, full feats   test on R
    B) wesad-only        train on WESAD                       test on R
    C) wesad + local     train on WESAD U (local \\ {R})       test on R

All training in A/B/C uses the SHARED feature set (HRV + BR + Temp) — WESAD
has no microphone. A_all is reported separately to show how much the mic
adds on local-only.

Default model: XGBoost (handles NaN natively, no scaling needed).
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .pipeline import PHASE_CLASSES, build_feature_table
from .wesad_io import build_wesad_feature_table


META_COLS = {"rec_name", "subject", "stressor", "phase", "activity", "t_start", "t_end"}
CPS_PREFIXES = ("cps_",)        # microphone-derived features (no WESAD analogue)


def _shared_feature_cols(df_local: pd.DataFrame, df_wesad: pd.DataFrame) -> list[str]:
    """Columns present in BOTH tables, with CPS (mic) features excluded."""
    local_cols = set(df_local.columns) - META_COLS
    wesad_cols = set(df_wesad.columns) - META_COLS
    shared = sorted(local_cols & wesad_cols)
    return [c for c in shared if not c.startswith(CPS_PREFIXES)]


def _local_feature_cols(df_local: pd.DataFrame) -> list[str]:
    return sorted(set(df_local.columns) - META_COLS)


def _xy(df: pd.DataFrame, feat_cols: list[str], le: LabelEncoder) -> tuple[np.ndarray, np.ndarray]:
    # XGBoost is fine with NaN — let it handle missing values via its default split logic.
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = le.transform(df["activity"].to_numpy())
    return X, y


def _make_xgb() -> XGBClassifier:
    use_gpu = os.environ.get("BIESL_USE_GPU", "0") == "1"
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=0, n_jobs=-1, eval_metric="mlogloss",
        tree_method="hist", device=("cuda" if use_gpu else "cpu"),
    )


def _metrics(y_true_lbl: np.ndarray, y_pred_lbl: np.ndarray) -> dict:
    return dict(
        accuracy=float(accuracy_score(y_true_lbl, y_pred_lbl)),
        macro_f1=float(f1_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)),
        per_class_f1={
            c: float(f1_score(
                y_true_lbl, y_pred_lbl, labels=[c], average="macro", zero_division=0,
            ))
            for c in PHASE_CLASSES
        },
        confusion=confusion_matrix(y_true_lbl, y_pred_lbl, labels=PHASE_CLASSES).tolist(),
        report=classification_report(
            y_true_lbl, y_pred_lbl, labels=PHASE_CLASSES, zero_division=0, output_dict=True,
        ),
    )


def _fit_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str], le: LabelEncoder,
) -> dict:
    if len(train_df) == 0:
        raise ValueError("empty training set")
    # XGBoost needs ALL classes present in y_train. Pad with 1 NaN-only row per
    # missing class so the encoder shape stays consistent (rare, only happens
    # for tiny folds).
    present = set(train_df["activity"].unique())
    missing = set(PHASE_CLASSES) - present
    if missing:
        pad = pd.DataFrame([
            {c: np.nan for c in feat_cols} | {
                "rec_name": f"_pad_{m}", "subject": "_pad", "stressor": "_pad",
                "phase": m, "activity": m, "t_start": 0.0, "t_end": 0.0,
            }
            for m in missing
        ])
        train_df = pd.concat([train_df, pad], ignore_index=True)

    X_tr, y_tr = _xy(train_df, feat_cols, le)
    X_te, y_te = _xy(test_df, feat_cols, le)
    clf = _make_xgb()
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return _metrics(le.inverse_transform(y_te), le.inverse_transform(y_pred))


def run_three_way(
    df_local: pd.DataFrame | None = None,
    df_wesad: pd.DataFrame | None = None,
    *,
    out_dir: str = "outputs",
    save: bool = True,
) -> dict:
    """Leave-one-recording-out comparison: A / A_all / B / C, all tested on local."""
    os.makedirs(out_dir, exist_ok=True)
    if df_local is None:
        print("[local] building feature table ...")
        df_local = build_feature_table()
    if df_wesad is None:
        print("[wesad] building feature table ...")
        df_wesad = build_wesad_feature_table()

    if df_wesad.empty:
        raise RuntimeError(
            "No WESAD windows produced — is the WESAD/ directory populated?"
        )

    le = LabelEncoder().fit(PHASE_CLASSES)
    shared_cols = _shared_feature_cols(df_local, df_wesad)
    full_cols = _local_feature_cols(df_local)
    print(f"shared feature count (HRV+BR+Temp): {len(shared_cols)}")
    print(f"full local feature count (+CPS):    {len(full_cols)}")

    recordings = sorted(df_local["rec_name"].unique())
    print(f"local recordings ({len(recordings)}): {recordings}")
    print(f"WESAD subjects ({df_wesad['subject'].nunique()}): "
          f"{sorted(df_wesad['subject'].unique())}")
    print(f"WESAD windows by activity: {dict(Counter(df_wesad['activity']))}")

    folds: list[dict] = []
    for rec in recordings:
        test_df = df_local[df_local["rec_name"] == rec]
        train_local = df_local[df_local["rec_name"] != rec]
        if test_df.empty or train_local.empty:
            continue
        if test_df["activity"].nunique() < 2:
            # Single-class test fold — macro-F1 isn't meaningful, but we still
            # report so the user can see which recording lacks variety.
            pass

        # A: local-only, shared features (apples-to-apples vs B/C)
        m_local = _fit_predict(train_local, test_df, shared_cols, le)
        # A_all: local-only, all features (shows what the mic adds)
        m_local_all = _fit_predict(train_local, test_df, full_cols, le)
        # B: WESAD-only -> local test
        m_wesad = _fit_predict(df_wesad, test_df, shared_cols, le)
        # C: WESAD + local-train -> local test
        train_combined = pd.concat([df_wesad, train_local], ignore_index=True)
        m_combined = _fit_predict(train_combined, test_df, shared_cols, le)

        folds.append(dict(
            recording=rec,
            test_n=int(len(test_df)),
            train_local_n=int(len(train_local)),
            train_wesad_n=int(len(df_wesad)),
            train_combined_n=int(len(train_combined)),
            test_activities=dict(Counter(test_df["activity"])),
            A_local_shared=m_local,
            A_local_full=m_local_all,
            B_wesad=m_wesad,
            C_combined=m_combined,
        ))
        print(
            f"  [{rec:<28s}] "
            f"A(shared)={m_local['macro_f1']:.3f}  "
            f"A(+mic)={m_local_all['macro_f1']:.3f}  "
            f"B(wesad)={m_wesad['macro_f1']:.3f}  "
            f"C(combo)={m_combined['macro_f1']:.3f}"
        )

    summary = _aggregate(folds)
    out = dict(
        classes=PHASE_CLASSES,
        shared_features=shared_cols,
        full_features=full_cols,
        folds=folds,
        summary=summary,
    )
    if save:
        path = os.path.join(out_dir, "transfer_results.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  -> wrote {path}")

    _print_summary(summary)
    return out


def _aggregate(folds: list[dict]) -> dict:
    keys = ("A_local_shared", "A_local_full", "B_wesad", "C_combined")
    summary: dict = {}
    for k in keys:
        acc = [f[k]["accuracy"] for f in folds]
        f1m = [f[k]["macro_f1"] for f in folds]
        per_class = {
            c: [f[k]["per_class_f1"][c] for f in folds] for c in PHASE_CLASSES
        }
        cm = np.sum([f[k]["confusion"] for f in folds], axis=0).tolist()
        summary[k] = dict(
            mean_accuracy=float(np.mean(acc)),
            std_accuracy=float(np.std(acc)),
            mean_macro_f1=float(np.mean(f1m)),
            std_macro_f1=float(np.std(f1m)),
            per_class_f1_mean={c: float(np.mean(per_class[c])) for c in PHASE_CLASSES},
            confusion_total=cm,
        )
    return summary


def _print_summary(summary: dict) -> None:
    print("\n========== Three-way comparison (LORO, test=local) ==========")
    pretty = {
        "A_local_shared": "A  local-only (shared feats)",
        "A_local_full":   "A+ local-only (+ mic CPS)",
        "B_wesad":        "B  WESAD-only -> local",
        "C_combined":     "C  WESAD + local-train",
    }
    header_classes = "  ".join(f"F1[{c}]" for c in PHASE_CLASSES)
    print(f"{'condition':<32s} {'acc':>6s} {'macroF1':>9s}  {header_classes}")
    for k, label in pretty.items():
        s = summary[k]
        per_cls = "  ".join(f"{s['per_class_f1_mean'][c]:6.3f}" for c in PHASE_CLASSES)
        print(
            f"{label:<32s} {s['mean_accuracy']:>6.3f} {s['mean_macro_f1']:>9.3f}  {per_cls}"
        )


def main() -> None:
    run_three_way()


if __name__ == "__main__":
    main()
