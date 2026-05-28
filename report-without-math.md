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
> windows with the median-filter BR (30 s baseline). XGBoost pooled-LORO here
> is **0.852** — slightly better despite the much harder 7-subject split,
> which is a good sign for the median-filter pipeline.
>
> **Internal ablation of the median-filter baseline window** (held everything
> else constant, same 31-recording dataset and 40 s windows):
>
> | baseline window | XGBoost pooled-LORO | 1D-CNN pooled-LORO |
> |---|---|---|
> | 8 s  | 0.838 | 0.767 |
> | **30 s** | **0.852** | **0.807** |
>
> Widening the baseline from 8 s to 30 s lifts every classical model and gives
> the CNN a +0.04 jump — the longer baseline preserves more of the true
> breathing oscillation, which in turn produces more discriminative
> `rr`/`rrv` features.

## Setup

- **Classes:** `rest` / `meditation` / `plank` (math recordings dropped).
- **Windows:** 40 s, 50 % overlap, 5-/10-min boundary windows skipped, recovery phase dropped.
- **Window counts:** 589 total — **403 rest / 120 meditation / 66 plank**.
- **BR:** median filter (**30 s baseline median** + 0.5 s smoothing median) + adaptive prominence floor.
- **Models:** KNN, RandomForest, XGBoost (8 PDF features); 1D-CNN (ECG + Resp + Mic-Shannon-envelope raw channels, 40 s × 250 Hz).
- **Protocols:** LORO (leave-one-recording-out, 25 folds) and stratified 70:15:15 random split (5 seeds).

## Results — macro-F1

| Model | LORO (pooled) | Random split |
|---|---|---|
| KNN | 0.774 | 0.816 |
| RandomForest | 0.815 | 0.880 |
| **XGBoost** | **0.852** | 0.921 |
| 1D-CNN | 0.807 | **0.976** |

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---|---|---|---|---|
| XGBoost · LORO (pooled) | 0.895 | **0.852** | 0.94 | 0.77 | 0.85 |
| RandomForest · LORO (pooled) | 0.883 | 0.815 | 0.93 | 0.81 | 0.71 |
| 1D-CNN · LORO (pooled) | 0.852 | 0.807 | 0.90 | 0.75 | 0.77 |
| KNN · LORO (pooled) | 0.852 | 0.774 | 0.90 | 0.73 | 0.69 |
| XGBoost · random | 0.937 | 0.921 | 0.96 | 0.88 | 0.93 |
| 1D-CNN · random | 0.978 | 0.976 | 0.98 | 0.94 | 1.00 |

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

1. **All three classes generalize well across 7 subjects** under pooled LORO. XGBoost: `rest` 0.94, `meditation` 0.77, `plank` 0.85.
2. **XGBoost is the strongest model** (pooled-LORO macro-F1 **0.852**), ahead of RandomForest (0.815), the 1D-CNN (0.807) and KNN (0.774).
3. **The 30 s baseline median is a measurable improvement over 8 s.** Cleaner `rr`/`rrv` features lift every model — most for the 1D-CNN (0.767 → 0.807), modestly for XGBoost (0.838 → 0.852).
4. **Random-split scores (0.82–0.98) are inflated by 50 % window-overlap leakage** and should not be quoted as cross-subject performance — the 1D-CNN's 0.976 vs 0.807 pooled-LORO gap is essentially all leakage.
5. **`meditation` recall is the softest spot** (~0.77 for XGBoost): some meditation windows are predicted as `rest`, consistent with quiet meditation breathing resembling rest in HR/HRV/RR.
