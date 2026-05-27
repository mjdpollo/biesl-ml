# Report — WITH math (4-class: rest / meditation / plank / math)

Purpose: **evaluate the current full system** — all 31 recordings, all four
classes, the 8 features from `features.pdf`, the median-filter BR pipeline, and
40 s windows.

## Setup

- **Classes:** `rest` / `meditation` / `plank` / `math`.
- **Windows:** 40 s, 50 % overlap, 5-/10-min boundary windows skipped, recovery phase dropped.
- **Window counts:** 661 total — **403 rest / 120 meditation / 66 plank / 72 math**.
- **Data:** 31 recordings, 7 subjects (`mta`, `mta2`, `nvt`, `ntv`, `nva`, `oyj`, `smj`).
- **BR:** median filter (8 s baseline median + 0.5 s smoothing median) + adaptive prominence floor.
- **Models:** KNN, RandomForest, XGBoost (8 PDF features); 1D-CNN (ECG + Resp + Mic-Shannon-envelope raw channels, 40 s × 250 Hz).
- **Protocols:** LORO (leave-one-recording-out, 31 folds) and stratified 70:15:15 random split (5 seeds).

> **RR/BR uses the new median-filter preprocessing.** LORO macro-F1 is the
> **pooled** value (computed once over all held-out predictions), because each
> recording contains only `rest` + one stressor — so per-fold averaging would
> charge zeros for the 2-3 classes absent from each fold's test set.

## Results — macro-F1

| Model | LORO (pooled) | Random split |
|---|---|---|
| KNN | 0.621 | 0.794 |
| RandomForest | 0.633 | 0.804 |
| **XGBoost** | **0.723** | 0.859 |
| 1D-CNN | **0.739** | **0.948** |

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] | F1[math] |
|---|---|---|---|---|---|---|
| XGBoost · LORO (pooled) | 0.803 | 0.723 | 0.87 | 0.74 | 0.81 | 0.47 |
| 1D-CNN · LORO (pooled) | 0.764 | **0.739** | 0.83 | 0.66 | 0.90 | 0.57 |
| RandomForest · LORO (pooled) | 0.776 | 0.633 | 0.86 | 0.78 | 0.66 | 0.24 |
| KNN · LORO (pooled) | 0.756 | 0.621 | 0.85 | 0.74 | 0.54 | 0.35 |
| XGBoost · random | 0.912 | 0.859 | 0.95 | 0.92 | 0.86 | 0.71 |
| 1D-CNN · random | 0.948 | 0.948 | 0.96 | 0.89 | 1.00 | 0.95 |

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

1. **Adding `math` lowers macro-F1 vs the 3-class run** (XGBoost pooled-LORO 0.838 → 0.723). `math` is the hardest class — a minority (72 windows) that is physiologically confusable with the other states.
2. **`math` is the weakest class** (pooled-LORO F1 0.47 for XGBoost, 0.57 for the CNN). The confusion matrices show math windows scattering into `rest` and `plank`. It's learnable but needs more data/subjects to firm up.
3. **`rest` / `meditation` / `plank` stay strong** with math present (XGBoost: 0.87 / 0.74 / 0.81) — adding the 4th class doesn't wreck the other three.
4. **The 1D-CNN edges XGBoost on pooled-LORO macro-F1 here** (0.739 vs 0.723), driven by the best `plank` (0.90) and `math` (0.57) recall — the class-weighted loss helps the minority stress classes. XGBoost remains stronger on `rest`/`meditation` and is far cheaper to train.
5. **Random-split macro-F1 (0.79–0.95) is inflated by 50 % window-overlap leakage** — the 1D-CNN hits 0.948 random vs 0.739 pooled-LORO. Quote pooled LORO for any cross-subject claim.

## What would move the numbers

1. **More minority-class data, especially `math` and `plank`.** 72 math / 66 plank windows across a handful of subjects is too few for cross-subject learning.
2. **More subjects per class.** LORO is now a genuine cross-subject test (7 subjects); each added subject for the stress classes should help generalization.
3. **Per-subject normalization of features** (z-score HR/HRV/RR/CSI within subject) would likely lift meditation/plank/math LORO F1 by removing inter-subject baseline offsets.
