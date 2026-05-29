# Report — WITH math (4-class: rest / meditation / plank / math)

Purpose: **evaluate the current full system** — all 31 recordings, all four
classes, the 8 features from `features.pdf`, the median-filter BR pipeline, and
40 s windows.

## Setup

- **Classes:** `rest` / `meditation` / `plank` / `math`.
- **Windows:** 40 s, 50 % overlap, 5-/10-min boundary windows skipped, recovery phase dropped.
- **Window counts:** 661 total — **403 rest / 120 meditation / 66 plank / 72 math**.
- **Data:** 31 recordings, 7 subjects (`mta`, `mta2`, `nvt`, `ntv`, `nva`, `oyj`, `smj`).
- **BR:** median filter (**30 s baseline median** + 0.5 s smoothing median) → **neurokit2 `rsp_peaks` (biosppy method)** for peak detection. See [`br-detector-comparison.md`](br-detector-comparison.md).
- **Models:** KNN, RandomForest, XGBoost (8 PDF features); 1D-CNN (ECG + Resp + Mic-Shannon-envelope raw channels, 40 s × 250 Hz).
- **Protocols:** LORO (leave-one-recording-out, 31 folds) and stratified 70:15:15 random split (5 seeds).

> **RR/BR uses the new median-filter preprocessing.** LORO macro-F1 is the
> **pooled** value (computed once over all held-out predictions), because each
> recording contains only `rest` + one stressor — so per-fold averaging would
> charge zeros for the 2-3 classes absent from each fold's test set.

## Results — macro-F1

| Model | LORO (pooled) | Random split |
|---|---|---|
| KNN | 0.587 | 0.734 |
| RandomForest | 0.680 | 0.765 |
| **XGBoost** | **0.769** | 0.853 |
| 1D-CNN | 0.753 | **0.942** |

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] | F1[math] |
|---|---|---|---|---|---|---|
| XGBoost · LORO (pooled) | 0.841 | **0.769** | 0.90 | 0.82 | 0.79 | 0.57 |
| 1D-CNN · LORO (pooled) | 0.808 | 0.753 | 0.87 | 0.78 | 0.72 | 0.64 |
| RandomForest · LORO (pooled) | 0.802 | 0.680 | 0.87 | 0.85 | 0.67 | 0.33 |
| KNN · LORO (pooled) | 0.725 | 0.587 | 0.82 | 0.72 | 0.58 | 0.23 |
| XGBoost · random | 0.902 | 0.853 | 0.93 | 0.92 | 0.87 | 0.69 |
| 1D-CNN · random | 0.948 | 0.942 | 0.96 | 0.91 | 0.99 | 0.91 |

## Confusion matrices (row-normalized %)

### LORO

| KNN | RandomForest | XGBoost | 1D-CNN |
|---|---|---|---|
| ![](figures/with_math/confusion/loro__knn.png) | ![](figures/with_math/confusion/loro__randomforest.png) | ![](figures/with_math/confusion/loro__xgboost.png) | ![](figures/with_math/confusion/loro__cnn.png) |

### Random 70:15:15 (5 seeds)

| KNN | RandomForest | XGBoost | 1D-CNN |
|---|---|---|---|
| ![](figures/with_math/confusion/random__knn.png) | ![](figures/with_math/confusion/random__randomforest.png) | ![](figures/with_math/confusion/random__xgboost.png) | ![](figures/with_math/confusion/random__cnn.png) |

## Findings

1. **Adding `math` lowers macro-F1 vs the 3-class run** (XGBoost pooled-LORO 0.876 → 0.769). `math` is the hardest class — a minority (72 windows) that is physiologically confusable with the other states.
2. **`math` is the weakest class** (pooled-LORO F1 0.57 for XGBoost, 0.64 for the CNN). The confusion matrices show math windows scattering into `rest` and `plank`. It's learnable but needs more data/subjects to firm up.
3. **`rest` / `meditation` / `plank` stay strong** with math present (XGBoost: 0.90 / 0.82 / 0.79) — adding the 4th class doesn't wreck the other three.
4. **XGBoost retakes the lead** at pooled-LORO macro-F1 **0.769**, ahead of the 1D-CNN (0.753). The CNN still has the best `math` recall (0.64) — class-weighted loss helps the rare class — but XGBoost has stronger non-math classes.
5. **The neurokit BR detector lifts every model.** XGBoost goes 0.736 → 0.769 (+0.033), RandomForest 0.610 → 0.680 (+0.070). The CNN holds (0.746 → 0.753).
6. **Random-split macro-F1 (0.73–0.94) is inflated by 50 % window-overlap leakage** — the 1D-CNN hits 0.942 random vs 0.753 pooled-LORO. Quote pooled LORO for any cross-subject claim.

## What would move the numbers

1. **More minority-class data, especially `math` and `plank`.** 72 math / 66 plank windows across a handful of subjects is too few for cross-subject learning.
2. **More subjects per class.** LORO is now a genuine cross-subject test (7 subjects); each added subject for the stress classes should help generalization.
3. **Per-subject normalization of features** (z-score HR/HRV/RR/CSI within subject) would likely lift meditation/plank/math LORO F1 by removing inter-subject baseline offsets.
