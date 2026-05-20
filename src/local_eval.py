"""Local-only LORO evaluation with the features.pdf feature set.

For each local recording R in turn:
    train on (local \ {R}), test on R, repeat for KNN / RF / XGBoost.

Two configurations are reported side by side:
    (i)  PDF features only            — csi, hr, hrv_rmssd, hrv_lf, hrv_hf,
                                        hrv_lf_hf, rr, rrv  (8 features)
    (ii) PDF + temperature ablation   — above + temp_mean_C, temp_std_C,
                                        temp_slope_Cps      (11 features)

This is the answer to "compare results with temperature vs without".
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import FEATURE_NAMES, TEMP_FEATURE_NAMES
from .pipeline import PHASE_CLASSES, build_feature_table


META_COLS = {"rec_name", "subject", "stressor", "phase", "activity", "t_start", "t_end"}


def _make_knn() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean")),
    ])


def _make_rf() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            max_features="sqrt", random_state=0, n_jobs=-1,
        )),
    ])


def _make_xgb() -> XGBClassifier:
    use_gpu = os.environ.get("BIESL_USE_GPU", "0") == "1"
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=0, n_jobs=-1, eval_metric="mlogloss",
        tree_method="hist", device=("cuda" if use_gpu else "cpu"),
    )


MODEL_FACTORIES = {
    "knn":          _make_knn,
    "randomforest": _make_rf,
    "xgboost":      _make_xgb,
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        per_class_f1={
            c: float(f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0))
            for c in PHASE_CLASSES
        },
        confusion=confusion_matrix(y_true, y_pred, labels=PHASE_CLASSES).tolist(),
        report=classification_report(
            y_true, y_pred, labels=PHASE_CLASSES, zero_division=0, output_dict=True,
        ),
    )


def _pad_missing_classes(train_df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    present = set(train_df["activity"].unique())
    missing = set(PHASE_CLASSES) - present
    if not missing:
        return train_df
    pad = pd.DataFrame([
        {c: np.nan for c in feat_cols} | {
            "rec_name": f"_pad_{m}", "subject": "_pad", "stressor": "_pad",
            "phase": m, "activity": m, "t_start": 0.0, "t_end": 0.0,
        }
        for m in missing
    ])
    return pd.concat([train_df, pad], ignore_index=True)


def _fit_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str],
    le: LabelEncoder, *, model: str,
) -> dict:
    train_df = _pad_missing_classes(train_df, feat_cols)
    X_tr = train_df[feat_cols].to_numpy(dtype=np.float64)
    y_tr = le.transform(train_df["activity"].to_numpy())
    X_te = test_df[feat_cols].to_numpy(dtype=np.float64)
    y_te = le.transform(test_df["activity"].to_numpy())
    clf = MODEL_FACTORIES[model]()
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return _metrics(le.inverse_transform(y_te), le.inverse_transform(y_pred))


def run_loro_local(df: pd.DataFrame, *, models: tuple[str, ...], label: str) -> dict:
    """LORO over local recordings, all models."""
    le = LabelEncoder().fit(PHASE_CLASSES)
    feat_cols = sorted(set(df.columns) - META_COLS)
    print(f"\n[{label}] feature columns ({len(feat_cols)}): {feat_cols}")

    recordings = sorted(df["rec_name"].unique())
    print(f"[{label}] recordings ({len(recordings)}): {recordings}")
    print(f"[{label}] activity counts: {dict(Counter(df['activity']))}")

    per_model: dict[str, list[dict]] = {m: [] for m in models}
    for rec in recordings:
        test_df = df[df["rec_name"] == rec]
        train_df = df[df["rec_name"] != rec]
        for m in models:
            res = _fit_predict(train_df, test_df, feat_cols, le, model=m)
            per_model[m].append(dict(recording=rec, **res))
        f1s = "  ".join(
            f"{m}={per_model[m][-1]['macro_f1']:.3f}" for m in models
        )
        print(f"  [{rec:<28s}] {f1s}")

    # Aggregate.
    summary: dict[str, dict] = {}
    for m, folds in per_model.items():
        acc = [f["accuracy"] for f in folds]
        f1m = [f["macro_f1"] for f in folds]
        per_cls = {c: [f["per_class_f1"][c] for f in folds] for c in PHASE_CLASSES}
        cm = np.sum([f["confusion"] for f in folds], axis=0).tolist()
        summary[m] = dict(
            mean_accuracy=float(np.mean(acc)),
            std_accuracy=float(np.std(acc)),
            mean_macro_f1=float(np.mean(f1m)),
            std_macro_f1=float(np.std(f1m)),
            per_class_f1_mean={c: float(np.mean(per_cls[c])) for c in PHASE_CLASSES},
            confusion_total=cm,
        )

    return dict(
        label=label,
        feature_cols=feat_cols,
        recordings=recordings,
        per_fold=per_model,
        summary=summary,
    )


def _stratified_three_way_split(
    n: int, y: np.ndarray, *,
    val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split returning index arrays.

    With only 4 stress-class windows the stratified `train_test_split` falls
    back to non-stratified once any class has < 2 samples in a target split,
    so we trap that and split unstratified in those cases.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    rest_frac = val_frac + test_frac
    try:
        idx_tr, idx_rest = train_test_split(
            idx, test_size=rest_frac, stratify=y, random_state=seed,
        )
        y_rest = y[idx_rest]
        # ratio of val within (val+test)
        val_within = val_frac / rest_frac
        idx_va, idx_te = train_test_split(
            idx_rest, test_size=1 - val_within, stratify=y_rest, random_state=seed,
        )
    except ValueError:
        idx_tr, idx_rest = train_test_split(idx, test_size=rest_frac, random_state=seed)
        idx_va, idx_te = train_test_split(idx_rest, test_size=test_frac / rest_frac,
                                          random_state=seed)
    return idx_tr, idx_va, idx_te


def run_random_split_local(
    df: pd.DataFrame, *, label: str,
    models: tuple[str, ...] = ("knn", "randomforest", "xgboost"),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    val_frac: float = 0.15, test_frac: float = 0.15,
) -> dict:
    """7:1.5:1.5 stratified random window split, repeated over `seeds`.

    Returns per-seed metrics and a mean±std summary per model.
    """
    le = LabelEncoder().fit(PHASE_CLASSES)
    feat_cols = sorted(set(df.columns) - META_COLS)
    print(f"\n[{label}] feature columns ({len(feat_cols)}): {feat_cols}")
    print(f"[{label}] total windows: {len(df)}, "
          f"label counts: {dict(Counter(df['activity']))}")
    print(f"[{label}] split fractions: train={1 - val_frac - test_frac:.2f} / "
          f"val={val_frac:.2f} / test={test_frac:.2f}, seeds={seeds}")

    per_seed: dict[int, dict] = {}
    y_full = df["activity"].to_numpy()
    for seed in seeds:
        idx_tr, idx_va, idx_te = _stratified_three_way_split(
            len(df), y_full, val_frac=val_frac, test_frac=test_frac, seed=seed,
        )
        train_df = df.iloc[idx_tr]
        val_df = df.iloc[idx_va]
        test_df = df.iloc[idx_te]
        seed_results: dict[str, dict] = {}
        for m in models:
            test_m = _fit_predict(train_df, test_df, feat_cols, le, model=m)
            val_m = _fit_predict(train_df, val_df, feat_cols, le, model=m)
            seed_results[m] = dict(val=val_m, test=test_m)
        per_seed[seed] = dict(
            counts=dict(
                train=dict(Counter(train_df["activity"])),
                val=dict(Counter(val_df["activity"])),
                test=dict(Counter(test_df["activity"])),
            ),
            models=seed_results,
        )
        f1s = "  ".join(
            f"{m}={seed_results[m]['test']['macro_f1']:.3f}" for m in models
        )
        print(f"  [seed {seed}]  test_n={len(test_df)} "
              f"({dict(Counter(test_df['activity']))})  {f1s}")

    # Aggregate per model.
    summary: dict[str, dict] = {}
    for m in models:
        acc = [per_seed[s]["models"][m]["test"]["accuracy"] for s in seeds]
        f1m = [per_seed[s]["models"][m]["test"]["macro_f1"] for s in seeds]
        per_cls: dict[str, list[float]] = {c: [] for c in PHASE_CLASSES}
        for s in seeds:
            for c in PHASE_CLASSES:
                per_cls[c].append(per_seed[s]["models"][m]["test"]["per_class_f1"][c])
        summary[m] = dict(
            mean_accuracy=float(np.mean(acc)),
            std_accuracy=float(np.std(acc)),
            mean_macro_f1=float(np.mean(f1m)),
            std_macro_f1=float(np.std(f1m)),
            per_class_f1_mean={c: float(np.mean(per_cls[c])) for c in PHASE_CLASSES},
            per_class_f1_std={c: float(np.std(per_cls[c])) for c in PHASE_CLASSES},
        )

    return dict(
        label=label,
        feature_cols=feat_cols,
        seeds=list(seeds),
        per_seed=per_seed,
        summary=summary,
    )


def run_random_split_temp_ablation(
    *, out_dir: str = "outputs",
    models: tuple[str, ...] = ("knn", "randomforest", "xgboost"),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 78)
    print(" Random 70:15:15 window split  —  classical models, with/without temperature")
    print("=" * 78)

    df_pdf = build_feature_table(include_temp=False)
    df_temp = build_feature_table(include_temp=True)

    results = dict(
        pdf_only=run_random_split_local(
            df_pdf, label="PDF features only", models=models, seeds=seeds,
        ),
        with_temp=run_random_split_local(
            df_temp, label="PDF + temperature", models=models, seeds=seeds,
        ),
    )
    path = os.path.join(out_dir, "local_randomsplit_temp_ablation.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    print(f"\n  -> wrote {path}")

    print("\n" + "=" * 78)
    print(" Classical random-split LORO  —  mean ± std macro-F1 over seeds")
    print("=" * 78)
    header_cls = "  ".join(f"F1[{c}]" for c in PHASE_CLASSES)
    print(f"{'config':<24s} {'model':<14s} {'acc':>10s} {'macroF1':>14s}  {header_cls}")
    for key, lbl in (("pdf_only", "PDF only (8 feat)"),
                     ("with_temp", "+temp (11 feat)")):
        for m in models:
            s = results[key]["summary"][m]
            acc = f"{s['mean_accuracy']:.3f}±{s['std_accuracy']:.3f}"
            f1m = f"{s['mean_macro_f1']:.3f}±{s['std_macro_f1']:.3f}"
            per_cls = "  ".join(
                f"{s['per_class_f1_mean'][c]:.3f}" for c in PHASE_CLASSES
            )
            print(f"{lbl:<24s} {m:<14s} {acc:>10s} {f1m:>14s}  {per_cls}")
    return results


def run_local_only_temp_ablation(
    *,
    out_dir: str = "outputs",
    models: tuple[str, ...] = ("knn", "randomforest", "xgboost"),
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print(" Building feature tables (PDF features + temperature ablation)")
    print("=" * 78)

    t0 = time.time()
    df_pdf = build_feature_table(include_temp=False)
    print(f"  PDF-only table built in {time.time() - t0:.1f}s, "
          f"{df_pdf.shape[0]} windows × {df_pdf.shape[1]} cols")

    t0 = time.time()
    df_with_temp = build_feature_table(include_temp=True)
    print(f"  With-temp table built in {time.time() - t0:.1f}s, "
          f"{df_with_temp.shape[0]} windows × {df_with_temp.shape[1]} cols")

    # Sanity: same row count.
    assert len(df_pdf) == len(df_with_temp), "row counts diverged"

    results = dict(
        pdf_only=run_loro_local(df_pdf, models=models, label="PDF features only"),
        with_temp=run_loro_local(df_with_temp, models=models, label="PDF + temperature"),
    )

    path = os.path.join(out_dir, "local_loro_temp_ablation.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    print(f"\n  -> wrote {path}")

    _print_comparison(results, models)
    return results


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _print_comparison(results: dict, models: tuple[str, ...]) -> None:
    print("\n" + "=" * 78)
    print(" Local-only LORO  —  with temperature vs without")
    print("=" * 78)
    header_cls = "  ".join(f"F1[{c}]" for c in PHASE_CLASSES)
    print(f"{'config':<22s} {'model':<14s} {'acc':>6s} {'macroF1':>9s}  {header_cls}")
    for key, label in (("pdf_only", "PDF only (8 feat)"), ("with_temp", "+temp (11 feat)")):
        for m in models:
            s = results[key]["summary"][m]
            per_cls = "  ".join(f"{s['per_class_f1_mean'][c]:6.3f}" for c in PHASE_CLASSES)
            print(
                f"{label:<22s} {m:<14s} {s['mean_accuracy']:>6.3f} "
                f"{s['mean_macro_f1']:>9.3f}  {per_cls}"
            )

    print("\nDelta (with_temp − pdf_only):")
    print(f"{'model':<14s} {'Δ acc':>8s} {'Δ macroF1':>11s}  Δ per-class F1")
    for m in models:
        a = results["pdf_only"]["summary"][m]
        b = results["with_temp"]["summary"][m]
        d_acc = b["mean_accuracy"] - a["mean_accuracy"]
        d_f1 = b["mean_macro_f1"] - a["mean_macro_f1"]
        d_pc = "  ".join(
            f"{b['per_class_f1_mean'][c] - a['per_class_f1_mean'][c]:+6.3f}"
            for c in PHASE_CLASSES
        )
        print(f"{m:<14s} {d_acc:+8.3f} {d_f1:+11.3f}  {d_pc}")


def main() -> None:
    run_local_only_temp_ablation()


if __name__ == "__main__":
    main()
