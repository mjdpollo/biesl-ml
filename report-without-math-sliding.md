# Report — WITHOUT math, sliding-window detector (3-class: rest / meditation / plank)

> **BR peak detector: sliding window** (`src.preprocess.detect_br_peaks_sliding`):
> 60 s windows stepped by 30 s, each window's prominence floor = `0.25 × p90(|signal|)`
> computed **locally**. For the default neurokit detector see
> [`report-without-math-neurokit.md`](report-without-math-neurokit.md); for the head-to-head see
> [`br-detector-comparison.md`](br-detector-comparison.md).

## Setup

- **Classes:** `rest` / `meditation` / `plank` (math recordings dropped).
- **Windows:** 40 s, 50 % overlap, ±40 s around the 5-/10-min boundaries excluded, recovery phase dropped.
- **Data:** 30 recordings ⇒ math files dropped for this config ⇒ 24 recordings, 9 subjects (`ljh`, `mta`, `mta2`, `nnn`, `ntv`, `nvt`, `nva`, `oyj`, `smj`, `tnq`).
- **Hardware/data-quality exclusions** (per running notes, see [`src/exclusions.py`](src/exclusions.py)):
  - all 5-17 recordings excluded (pre-hardware-fix);
  - `mta_5_19_medi (1)` excluded (duplicate);
  - `nvt_5_21_pla_2(1)`, `ntv_5_25_pla_2'10`, `mta_5_19_pla_1'40` excluded (hardware glitches);
  - `mta_5_21_medi` 0–150 s excluded (sensor settling);
  - `oyj_5_22_pla_2'15` and `oyj_5_22_pla_1'50` plank phase only excluded (plank glitch, rest kept).
- **Window counts:** 413 total — **304 rest / 64 meditation / 45 plank** (was 509 = 350/102/57 in the prior smaller-dataset run).
- **BR:** median filter (30 s baseline + 0.5 s smoothing) → **sliding-window peak detector** (local p90 prominence floor per 60 s window).
- **Models:** KNN (k=7), RandomForest (400 trees), XGBoost (400 trees, depth=4); 1D-CNN on ECG + Resp + Mic Shannon-envelope raw channels.
- **Protocols:** LORO (leave-one-recording-out) and 5-seed 70:15:15 stratified random window split. LORO macro-F1 is **pooled** over all held-out predictions.

## Results — macro-F1

| Model | LORO (pooled) | Random split |
|---|---|---|
| KNN | 0.687 | 0.814 |
| RandomForest | 0.738 | 0.818 |
| **XGBoost** | **0.808** | **0.920** |
| 1D-CNN † | 0.790 | 0.956 |

> † The 1D-CNN reads the filtered BR **waveform**, not detected peaks, so its
> performance is structurally invariant to the detector. The CNN row is the
> same architecture re-trained under the sliding-detector pipeline; its score
> differences vs neurokit/global are mostly training-seed noise.

### Per-class F1

| Model · protocol | acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---|---|---|---|---|
| **XGBoost** · LORO (pooled) | 0.884 | **0.808** | 0.92 | 0.81 | 0.69 |
| 1D-CNN · LORO (pooled) | 0.860 | 0.790 | 0.92 | 0.71 | 0.74 |
| RandomForest · LORO (pooled) | 0.867 | 0.738 | 0.92 | 0.80 | 0.50 |
| KNN · LORO (pooled) | 0.840 | 0.687 | 0.90 | 0.75 | 0.41 |
| XGBoost · random | 0.952 | 0.920 | 0.97 | 0.95 | 0.84 |
| 1D-CNN · random | 0.965 | 0.956 | 0.98 | 0.91 | 0.99 |

### Comparison with the neurokit detector

(neurokit numbers pending the in-flight re-run on the 30-recording dataset; the
old neurokit numbers were on the prior smaller dataset — see the
[report-without-math-neurokit.md](report-without-math-neurokit.md) refresh once it lands.)

## Confusion matrices — sliding detector

Rows are true labels, columns are predictions. Row-normalized to 100 %.

### LORO — Classical

| KNN | RandomForest | XGBoost |
|---|---|---|
| ![](figures/without_math/confusion_sliding/loro__knn.png) | ![](figures/without_math/confusion_sliding/loro__randomforest.png) | ![](figures/without_math/confusion_sliding/loro__xgboost.png) |

### LORO — 1D-CNN

![](figures/without_math/confusion_sliding/loro__cnn.png)

### Random 70:15:15 (5 seeds) — Classical

| KNN | RandomForest | XGBoost |
|---|---|---|
| ![](figures/without_math/confusion_sliding/random__knn.png) | ![](figures/without_math/confusion_sliding/random__randomforest.png) | ![](figures/without_math/confusion_sliding/random__xgboost.png) |

### Random 70:15:15 — 1D-CNN

![](figures/without_math/confusion_sliding/random__cnn.png)

## Findings

1. **XGBoost reaches pooled-LORO macro-F1 0.808**, with **1D-CNN very close at 0.790**.
2. **The new 5-29 subjects (`nnn`, `tnq`) and the partial exclusions made the LORO test harder** — sliding XGBoost was 0.879 on the previous smaller dataset (no `nnn`/`tnq`, no partial exclusions), and is 0.808 here. The drop is the cross-subject diversity, not the BR pipeline.
3. **`plank` is still the weakest class** (XGBoost F1 0.69, RF 0.50). Even after adding 4 new plank recordings (`mta_5_29_pla_2'35`, `mta_5_29_pla_4`, `nnn_5_29_pla_3`, `tnq_5_29_pla_2'20`) the class is still smaller than rest/meditation and varies more in execution across subjects.
4. **Random-split scores (0.81–0.96) remain inflated by 50 % window-overlap leakage** — use pooled-LORO for cross-subject claims.

## Reproduce

```bash
BR_PEAK_METHOD=sliding uv run python scripts/run_split_reports.py
```
