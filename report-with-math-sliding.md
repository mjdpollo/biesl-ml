# Report — WITH math, sliding-window detector (4-class: rest / meditation / plank / math)

> **BR peak detector: sliding window** (`src.preprocess.detect_br_peaks_sliding`):
> 60 s windows stepped by 30 s, each window's prominence floor = `0.25 × p90(|signal|)`
> computed **locally**. For the default neurokit detector see
> [`report-with-math-neurokit.md`](report-with-math-neurokit.md); for the head-to-head see
> [`br-detector-comparison.md`](br-detector-comparison.md).

## Setup

- **Classes:** `rest` / `meditation` / `plank` / `math`.
- **Windows:** 40 s, 50 % overlap, ±40 s around the 5-/10-min boundaries excluded, recovery phase dropped.
- **Data:** 30 recordings, 10 subjects (`ljh`, `mta`, `mta2`, `nnn`, `ntv`, `nvt`, `nva`, `oyj`, `smj`, `tnq`).
- **Hardware/data-quality exclusions** (per running notes, see [`src/exclusions.py`](src/exclusions.py)):
  - all 5-17 recordings excluded (pre-hardware-fix);
  - `mta_5_19_medi (1)` excluded (duplicate);
  - `nvt_5_21_pla_2(1)`, `ntv_5_25_pla_2'10`, `mta_5_19_pla_1'40` excluded (hardware glitches);
  - `mta_5_21_medi` 0–150 s and `mta_5_26_math_8_12` 0–140 s excluded (sensor settling);
  - `oyj_5_22_pla_2'15` and `oyj_5_22_pla_1'50` plank phase only excluded (plank glitch);
  - `tnq_5_29_math_7_12` rest phase only excluded (rest noisy, math is fine).
- **Window counts:** 477 total — **304 rest / 64 meditation / 45 plank / 64 math**.
- **BR:** median filter (30 s baseline + 0.5 s smoothing) → **sliding-window peak detector** (local p90 prominence floor per 60 s window).
- **Models:** KNN, RandomForest, XGBoost; 1D-CNN.
- **Protocols:** LORO (per-recording folds, pooled macro-F1) and 5-seed 70:15:15 random split.

## Results — macro-F1

| Model | LORO (pooled) | Random split |
|---|---|---|
| KNN | 0.539 | 0.689 |
| RandomForest | 0.578 | 0.661 |
| **XGBoost** | **0.658** | **0.794** |
| 1D-CNN † | 0.636 | 0.795 |

> † The 1D-CNN reads the filtered BR **waveform**, not detected peaks, so its
> performance is structurally invariant to the detector. The CNN row is the
> same architecture re-trained under the sliding-detector pipeline; its score
> differences vs neurokit/global are mostly training-seed noise.

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] | F1[math] |
|---|---|---|---|---|---|---|
| **XGBoost** · LORO (pooled) | 0.765 | **0.658** | 0.85 | 0.79 | 0.68 | 0.31 |
| 1D-CNN · LORO (pooled) | 0.725 | 0.636 | 0.84 | 0.57 | 0.91 | 0.22 |
| RandomForest · LORO (pooled) | 0.753 | 0.578 | 0.85 | 0.79 | 0.42 | 0.26 |
| KNN · LORO (pooled) | 0.688 | 0.539 | 0.81 | 0.76 | 0.34 | 0.25 |
| XGBoost · random | 0.864 | 0.794 | 0.92 | 0.92 | 0.72 | 0.62 |
| 1D-CNN · random | 0.836 | 0.795 | 0.89 | 0.78 | 0.93 | 0.58 |

### Comparison with the neurokit detector

(neurokit numbers pending the in-flight re-run on the 30-recording dataset;
the prior neurokit numbers were on the smaller dataset before the new
`tnq_*`, `nnn_*`, `nvt_5_29_medi`, and `mta_5_29_pla_*` recordings were
added — see [report-with-math-neurokit.md](report-with-math-neurokit.md) once it lands.)

## Confusion matrices — sliding detector

Rows are true labels, columns are predictions. Row-normalized to 100 %.

### LORO — Classical

| KNN | RandomForest | XGBoost |
|---|---|---|
| ![](figures/with_math/confusion_sliding/loro__knn.png) | ![](figures/with_math/confusion_sliding/loro__randomforest.png) | ![](figures/with_math/confusion_sliding/loro__xgboost.png) |

### LORO — 1D-CNN

![](figures/with_math/confusion_sliding/loro__cnn.png)

### Random 70:15:15 (5 seeds) — Classical

| KNN | RandomForest | XGBoost |
|---|---|---|
| ![](figures/with_math/confusion_sliding/random__knn.png) | ![](figures/with_math/confusion_sliding/random__randomforest.png) | ![](figures/with_math/confusion_sliding/random__xgboost.png) |

### Random 70:15:15 — 1D-CNN

![](figures/with_math/confusion_sliding/random__cnn.png)

## Findings

1. **XGBoost reaches pooled-LORO macro-F1 0.658**, with **1D-CNN at 0.636**. Per-class XGBoost: rest 0.85, meditation 0.79, plank 0.68, math 0.31.
2. **Adding the new subjects (`nnn`, `tnq`) makes LORO harder.** Previous sliding-detector 4-class XGBoost was 0.764 on the smaller dataset (no `nnn`/`tnq`, no partial exclusions). Same pipeline, more cross-subject diversity → 0.658. This is the real-world cost of expanding the subject pool.
3. **`math` is still the weakest class** at F1 0.31 (XGBoost). 64 windows across 3 subjects is on the edge of learnability cross-subject.
4. **Per-class for the CNN is interesting**: best `plank` recall of any model (F1 0.91 LORO), worst `meditation` (0.57) — the class-weighted CE loss does help the rare class.
5. **The 4-class setup costs ~0.15 macro-F1** vs the without-math 3-class run with the same exclusions (XGBoost 0.808 → 0.658). math is genuinely hard to add.

## Reproduce

```bash
BR_PEAK_METHOD=sliding uv run python scripts/run_split_reports.py
```
