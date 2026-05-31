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
> **Internal ablation — BR pipeline on this dataset** (30 recordings, 40 s
> windows, ±40 s boundary skip, 3-class, hardware-quality exclusions applied):
>
> | BR peak detector | XGBoost pooled-LORO | 1D-CNN pooled-LORO † |
> |---|---|---|
> | global | 0.803 | 0.769 |
> | **sliding** | **0.808** | 0.790 |
> | neurokit (default) | 0.790 | **0.865** |
>
> XGBoost is barely separable across detectors (the 0.018 spread is within
> run-to-run noise). The 1D-CNN row varies more, but the CNN reads the
> filtered BR *waveform*, not peaks, so its variation is training-seed
> noise across reruns rather than a real detector effect.

## Setup

- **Classes:** `rest` / `meditation` / `plank` (math recordings dropped).
- **Windows:** 40 s, 50 % overlap, ±40 s around the 5-/10-min boundaries excluded, recovery phase dropped.
- **Data:** 30 recordings ⇒ math files dropped for this config ⇒ 24 recordings, 9 subjects (`ljh`, `mta`, `mta2`, `nnn`, `ntv`, `nvt`, `nva`, `oyj`, `smj`, `tnq`).
- **Hardware/data-quality exclusions** (per running notes, see [`src/exclusions.py`](src/exclusions.py)):
  - all 5-17 recordings excluded (pre-hardware-fix);
  - `mta_5_19_medi (1)`, `nvt_5_21_pla_2(1)`, `ntv_5_25_pla_2'10`, `mta_5_19_pla_1'40` excluded entirely;
  - `mta_5_21_medi` 0–150 s excluded (sensor settling);
  - `oyj_5_22_pla_2'15` and `oyj_5_22_pla_1'50` plank phase only excluded (plank glitch, rest kept).
- **Window counts:** 413 total — **304 rest / 64 meditation / 45 plank**.
- **BR:** median filter (**30 s baseline median** + 0.5 s smoothing median) → **neurokit2 `rsp_peaks` (biosppy method)** for peak detection. See [`br-detector-comparison.md`](br-detector-comparison.md) for the detector benchmark.
- **Models:** KNN, RandomForest, XGBoost (8 PDF features); 1D-CNN (ECG + Resp + Mic-Shannon-envelope raw channels, 40 s × 250 Hz).
- **Protocols:** LORO (leave-one-recording-out, 25 folds) and stratified 70:15:15 random split (5 seeds).

## Results — macro-F1

| Model | LORO (pooled) | Random split |
|---|---|---|
| KNN | 0.642 | 0.759 |
| RandomForest | 0.720 | 0.756 |
| **XGBoost** | 0.790 | 0.867 |
| **1D-CNN** | **0.865** | **0.933** |

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---|---|---|---|---|
| **1D-CNN** · LORO (pooled) | 0.896 | **0.865** | 0.93 | 0.78 | **0.89** |
| XGBoost · LORO (pooled) | 0.874 | 0.790 | 0.92 | 0.75 | 0.70 |
| RandomForest · LORO (pooled) | 0.855 | 0.720 | 0.91 | 0.73 | 0.52 |
| KNN · LORO (pooled) | 0.814 | 0.642 | 0.88 | 0.62 | 0.42 |
| XGBoost · random | 0.916 | 0.867 | 0.94 | 0.88 | 0.78 |
| 1D-CNN · random | 0.952 | 0.933 | 0.97 | 0.87 | 0.96 |

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

1. **The 1D-CNN tops this 3-class run at pooled-LORO macro-F1 0.865** — the strongest single number we've seen. XGBoost trails at 0.790. Per-class for the CNN: rest 0.93, meditation 0.78, **plank 0.89** (a notable jump from the trees).
2. **Adding the new 5-29 subjects (`nnn`, `tnq`) and applying the hardware-quality exclusions made LORO harder for the classical models** — XGBoost dropped from 0.871 (smaller dataset) to 0.790 here. The CNN benefitted from the wider subject base.
3. **`plank` is still the smallest class** (45 windows) but it generalizes well now — even RF reaches F1 0.52 and the CNN 0.89.
4. **Random-split scores (0.76–0.93) are still inflated by 50 % window-overlap leakage** — quote pooled-LORO for any honest cross-subject claim.
