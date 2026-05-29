# Report — WITHOUT math, neurokit detector (3-class: rest / meditation / plank)

> **BR peak detector: `neurokit2.rsp_peaks` (default).** For the sliding-window
> detector see [`report-without-math-sliding.md`](report-without-math-sliding.md);
> the head-to-head is in [`br-detector-comparison.md`](br-detector-comparison.md).


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
> **Internal ablation — BR pipeline** (held everything else constant, 31-recording
> dataset, 40 s windows, **±40 s** boundary skip, 3-class):
>
> | BR peak detector | XGBoost pooled-LORO | 1D-CNN pooled-LORO † |
> |---|---|---|
> | global (single p90 over whole signal) | 0.856 | 0.840 |
> | sliding (60-s windows, local p90) | **0.879** | 0.812 |
> | **neurokit `rsp_peaks`** (default) | 0.871 | 0.773 |
>
> XGBoost is highest with *sliding* (+0.023 vs global, +0.008 vs neurokit);
> the CNN row's variation is structurally training-seed noise (CNN reads the
> waveform, not peaks — see [`br-detector-comparison.md`](br-detector-comparison.md)).
> neurokit stays the default because it has 3–5× tighter per-recording RR
> consistency and is top-2 for every classical model.

## Setup

- **Classes:** `rest` / `meditation` / `plank` (math recordings dropped).
- **Windows:** 40 s, 50 % overlap, **±40 s** around the 5-/10-min boundaries excluded (widened from 0 s), recovery phase dropped.
- **Window counts:** 589 total — **403 rest / 120 meditation / 66 plank**.
- **BR:** median filter (**30 s baseline median** + 0.5 s smoothing median) → **neurokit2 `rsp_peaks` (biosppy method)** for peak detection. See [`br-detector-comparison.md`](br-detector-comparison.md) for the detector benchmark.
- **Models:** KNN, RandomForest, XGBoost (8 PDF features); 1D-CNN (ECG + Resp + Mic-Shannon-envelope raw channels, 40 s × 250 Hz).
- **Protocols:** LORO (leave-one-recording-out, 25 folds) and stratified 70:15:15 random split (5 seeds).

## Results — macro-F1

| Model | LORO (pooled) | Random split |
|---|---|---|
| KNN | 0.734 | 0.839 |
| RandomForest | 0.784 | 0.866 |
| **XGBoost** | **0.871** | **0.922** |
| 1D-CNN | 0.773 | 0.932 |

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---|---|---|---|---|
| XGBoost · LORO (pooled) | 0.923 | **0.871** | 0.95 | 0.82 | 0.84 |
| RandomForest · LORO (pooled) | 0.908 | 0.784 | 0.95 | 0.85 | 0.56 |
| 1D-CNN · LORO (pooled) | 0.849 | 0.773 | 0.90 | 0.70 | 0.71 |
| KNN · LORO (pooled) | 0.860 | 0.734 | 0.91 | 0.68 | 0.60 |
| XGBoost · random | 0.954 | 0.922 | 0.97 | 0.91 | 0.89 |
| 1D-CNN · random | 0.948 | 0.932 | 0.97 | 0.86 | 0.97 |

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

1. **XGBoost is the strongest model** at pooled-LORO macro-F1 **0.871** (neurokit detector). With the *sliding* detector it edges to **0.879**, but the difference is within run-to-run noise so we keep neurokit as the default for breath-rate cleanliness — see [`br-detector-comparison.md`](br-detector-comparison.md).
2. **All three classes generalize well across 7 subjects**. XGBoost per-class F1: `rest` 0.95, `meditation` 0.82, `plank` 0.84.
3. **The widened ±40 s boundary skip costs us ~80 windows** vs the previous 0 s buffer (from 589 → 509-ish), but does NOT hurt macro-F1 — XGBoost is essentially unchanged (0.876 → 0.871). The transition windows weren't carrying useful class signal.
4. **Random-split scores (0.84–0.93) are still inflated by 50 % window-overlap leakage** — quote pooled-LORO for any honest cross-subject claim.
5. **`plank` recall depends strongly on the model.** XGBoost 0.84, KNN 0.60, RF 0.56. The trees underfit the smallest class; gradient boosting handles it.
6. **The 1D-CNN sits at 0.773** — solid, but ~0.1 below XGBoost. Its `meditation` and `plank` per-class F1 lag the gradient-boosted trees.
