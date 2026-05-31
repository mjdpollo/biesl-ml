# Report — WITH math, neurokit detector (4-class: rest / meditation / plank / math)

> **BR peak detector: `neurokit2.rsp_peaks` (default).** For the sliding-window
> detector see [`report-with-math-sliding.md`](report-with-math-sliding.md);
> the head-to-head is in [`br-detector-comparison.md`](br-detector-comparison.md).


Purpose: **evaluate the current full system** — all 31 recordings, all four
classes, the 8 features from `features.pdf`, the median-filter BR pipeline, and
40 s windows.

## Setup

- **Classes:** `rest` / `meditation` / `plank` / `math`.
- **Windows:** 40 s, 50 % overlap, ±40 s around the 5-/10-min boundaries excluded, recovery phase dropped.
- **Data:** 30 recordings, 10 subjects.
- **Hardware/data-quality exclusions** (see [`src/exclusions.py`](src/exclusions.py)):
  - all 5-17 recordings, `mta_5_19_medi (1)`, `nvt_5_21_pla_2(1)`, `ntv_5_25_pla_2'10`, `mta_5_19_pla_1'40` excluded entirely;
  - `mta_5_21_medi` 0–150 s, `mta_5_26_math_8_12` 0–140 s excluded;
  - `oyj_5_22_pla_2'15`, `oyj_5_22_pla_1'50` plank phase only excluded;
  - `tnq_5_29_math_7_12` rest phase only excluded.
- **Window counts:** 477 total — 304 rest / 64 meditation / 45 plank / 64 math.
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
| KNN | 0.520 | 0.611 |
| RandomForest | 0.527 | 0.680 |
| XGBoost | 0.633 | 0.814 |
| **1D-CNN** | **0.725** | **0.939** |

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] | F1[math] |
|---|---|---|---|---|---|---|
| **1D-CNN** · LORO (pooled) | 0.795 | **0.725** | 0.88 | 0.77 | 0.78 | 0.47 |
| XGBoost · LORO (pooled) | 0.744 | 0.633 | 0.83 | 0.72 | 0.63 | 0.35 |
| RandomForest · LORO (pooled) | 0.717 | 0.527 | 0.82 | 0.71 | 0.51 | 0.07 |
| KNN · LORO (pooled) | 0.686 | 0.520 | 0.80 | 0.65 | 0.38 | 0.25 |
| XGBoost · random | 0.875 | 0.814 | 0.92 | 0.94 | 0.74 | 0.66 |
| 1D-CNN · random | 0.953 | 0.939 | 0.97 | 0.91 | 0.97 | 0.90 |

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

1. **The 1D-CNN tops this 4-class run** at pooled-LORO macro-F1 **0.725**, ahead of XGBoost (0.633). Adding new subjects (`nnn`, `tnq`) and a wider class set has flipped the model ranking on the 30-recording dataset — the CNN now wins on every per-class F1.
2. **`math` is still the weakest class** (CNN F1 0.47, XGBoost 0.35). The confusion matrices show math windows scattering into `rest` and `meditation`. Even with the new `nnn` and `tnq` recordings added, math is the hardest cross-subject class.
3. **The new dataset is harder than the previous run.** XGBoost dropped from 0.748 → 0.633 because the added subjects expose cross-subject variability the classical features don't fully capture.
4. **`rest` / `meditation` / `plank` per-class F1 stays robust** for the CNN (0.88 / 0.77 / 0.78). Adding the 4th class costs ~0.14 from the 3-class run (CNN: 0.865 → 0.725).
5. **Random-split macro-F1 (0.61–0.94) is inflated by 50 % window-overlap leakage** — the 1D-CNN hits 0.939 random vs 0.725 pooled-LORO. Quote pooled LORO for any cross-subject claim.

## What would move the numbers

1. **More minority-class data, especially `math` and `plank`.** 72 math / 66 plank windows across a handful of subjects is too few for cross-subject learning.
2. **More subjects per class.** LORO is now a genuine cross-subject test (7 subjects); each added subject for the stress classes should help generalization.
3. **Per-subject normalization of features** (z-score HR/HRV/RR/CSI within subject) would likely lift meditation/plank/math LORO F1 by removing inter-subject baseline offsets.
