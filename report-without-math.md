# Report — WITHOUT math (3-class: rest / meditation / plank)

Purpose: **check the effect of the median-filter BR analysis** on the 3-class
problem. This run uses the refreshed 31-recording dataset (math recordings
excluded → 25 recordings, 7 subjects), the 8 features from `features.pdf`, the
**median-filter BR pipeline**, and **40 s windows**.

> **RR/BR uses the new median-filter preprocessing** (8 s baseline median +
> 0.5 s smoothing median + adaptive prominence floor) — same as
> [`median-filter-adapted-br.md`](median-filter-adapted-br.md).
>
> **LORO macro-F1 is computed on the POOLED held-out predictions, not by
> averaging per-fold.** Each recording contains only `rest` + one stressor, so
> in a leave-one-recording-out fold 1–2 classes are entirely absent from the
> test set; per-fold macro-F1 averaging then charges zeros for the absent
> classes and badly understates performance. Pooling all held-out predictions
> first (the confusion matrices below) is the correct aggregation.
>
> **Comparison context.** The earlier 3-class result in [`report.md`](report.md)
> (XGBoost 0.817) was on only **2 subjects** with 60 s windows and the old
> moving-average + Chebyshev BR filter. This run has **7 subjects** and 40 s
> windows with the median-filter BR. XGBoost pooled-LORO here is **0.838** —
> comparable/slightly better despite the much harder 7-subject split, which is
> a good sign for the median-filter pipeline.

## Setup

- **Classes:** `rest` / `meditation` / `plank` (math recordings dropped).
- **Windows:** 40 s, 50 % overlap, 5-/10-min boundary windows skipped, recovery phase dropped.
- **Window counts:** 589 total — **403 rest / 120 meditation / 66 plank**.
- **BR:** median filter (8 s baseline median + 0.5 s smoothing median) + adaptive prominence floor.
- **Models:** KNN, RandomForest, XGBoost (8 PDF features); 1D-CNN (ECG + Resp + Mic-Shannon-envelope raw channels, 40 s × 250 Hz).
- **Protocols:** LORO (leave-one-recording-out, 25 folds) and stratified 70:15:15 random split (5 seeds).

## Results — macro-F1

| Model | LORO (pooled) | Random split |
|---|---|---|
| KNN | 0.770 | 0.884 |
| RandomForest | 0.807 | 0.869 |
| **XGBoost** | **0.838** | 0.925 |
| 1D-CNN | 0.767 | **0.945** |

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---|---|---|---|---|
| XGBoost · LORO (pooled) | 0.881 | **0.838** | 0.92 | 0.74 | 0.85 |
| RandomForest · LORO (pooled) | 0.874 | 0.807 | 0.92 | 0.79 | 0.71 |
| KNN · LORO (pooled) | 0.842 | 0.770 | 0.90 | 0.73 | 0.68 |
| 1D-CNN · LORO (pooled) | 0.810 | 0.767 | 0.87 | 0.63 | 0.80 |
| XGBoost · random | 0.948 | 0.925 | 0.96 | 0.91 | 0.90 |
| 1D-CNN · random | 0.946 | 0.945 | 0.96 | 0.88 | 0.99 |

## Confusion matrices (row-normalized %)

### LORO

| KNN | RandomForest | XGBoost | 1D-CNN |
|---|---|---|---|
| ![](figures/without_math/confusion/loro__knn.png) | ![](figures/without_math/confusion/loro__randomforest.png) | ![](figures/without_math/confusion/loro__xgboost.png) | ![](figures/without_math/confusion/loro__cnn.png) |

### Random 70:15:15 (5 seeds)

| KNN | RandomForest | XGBoost | 1D-CNN |
|---|---|---|---|
| ![](figures/without_math/confusion/random__knn.png) | ![](figures/without_math/confusion/random__randomforest.png) | ![](figures/without_math/confusion/random__xgboost.png) | ![](figures/without_math/confusion/random__cnn.png) |

## Findings

1. **All three classes generalize well across subjects** under pooled LORO. XGBoost: `rest` 0.92, `meditation` 0.74, `plank` 0.85. This is a genuine 7-subject cross-recording result — much more credible than the old 2-subject number.
2. **XGBoost is the best model** (pooled-LORO macro-F1 0.838), ahead of RandomForest (0.807), KNN (0.770) and the 1D-CNN (0.767).
3. **Random-split scores (0.87–0.95) are inflated by 50 % window-overlap leakage** and should not be quoted as cross-subject performance — the honest number is pooled LORO. The 1D-CNN shows the biggest gap (0.767 → 0.945).
4. **The median-filter BR pipeline holds up.** Despite a far harder evaluation (7 subjects vs 2, 40 s windows vs 60 s), pooled-LORO macro-F1 (0.838) is on par with / slightly above the old 2-subject pipeline — see the comparison caveat above about why this is not a clean ablation.
5. **`meditation` recall is the softest spot** (≈0.74): the confusion matrices show some meditation windows predicted as `rest`, consistent with quiet meditation breathing resembling rest in HR/HRV.
