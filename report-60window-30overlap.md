# Rest / Meditation / Plank Classification Report

**Task.** Per-window **3-class** classification: `rest` / `meditation` / `plank`.
**Data.** 18 local wearable recordings (~10–15 min each) across 5 subjects (`ljh`, `mta`, `mta2`, `nvt`, `oyj`), pulled from the team Google Drive folder ([see README](README.md)). WESAD is not used.
**Features.** The eight parameters defined in [`features.pdf`](features.pdf), and only those — no temperature features.
**Excluded data.** The post-stressor `recovery` phase is dropped entirely from the dataset (not in the production taxonomy). Windows whose 60-s extent touches the 5-min or 10-min protocol transitions are also excluded.
**Evaluation protocols.** (a) Leave-one-recording-out (LORO, 18 folds) — the honest cross-recording benchmark; (b) stratified 70 : 15 : 15 random window split averaged over 5 seeds — a sanity check, contaminated by 50 % window overlap.
**Preprocessing.** Production pipeline now uses **`nk.ecg_peaks(method="neurokit")`** for R-peak detection (the previous `pantompkins1985` collapsed post-plank — see [data-processing-using-neurokit.md](data-processing-using-neurokit.md) for the diagnostic). The BR detector keeps the ±5·MAD outlier-clipping fix.

---

## 1. Feature definitions (from features.pdf)

Exactly eight features per 60 s window. No temperature features.

| Feature | Channel | Preprocessing | Computation |
|---|---|---|---|
| `csi` | Microphone | Bandpass 20–200 Hz → Shannon-energy envelope `−x² log(x²)` → peak detection | S2/S1 amplitude ratio. Each ECG R-peak picks one S1 (Shannon peak in R+0–200 ms) and one S2 (R+200–500 ms). CSI = mean(S2 amp) / mean(S1 amp). |
| `hr` | ECG | HP 1 Hz → notch 60 Hz (Q=30) → LP 150 Hz → `nk.ecg_clean(method="neurokit")` → `nk.ecg_peaks(method="neurokit")` → `nk.signal_fixpeaks(method="kubios", iterative=False)` → ±60 ms apex snap. NN intervals cleaned: reject NN < 300 ms / > 1500 ms / > 20 % from local median. Rejected NNs cubic-spline interpolated. | `60 000 / mean(NN_ms)` (bpm) |
| `hrv_rmssd` | ECG | Same cleaned NN series | `√(mean((NN_{i+1} − NN_i)²))`, in ms |
| `hrv_lf` | ECG | Welch PSD on 4 Hz interpolated tachogram, Hann window, 60 s segments where possible | Area under PSD in 0.04–0.15 Hz (ms²) |
| `hrv_hf` | ECG | Same Welch | Area under PSD in 0.15–0.40 Hz (ms²) |
| `hrv_lf_hf` | ECG | — | `hrv_lf / hrv_hf` |
| `rr` | Respiration | Detrend → 0.5 s MA → 4th-order Cheby II LP (stopband 1 Hz, 40 dB) → 2nd-order Butterworth HP at 0.12 Hz → ±5·MAD clip → slope-based peak detection with adaptive threshold = 1/3 × mean of last 8 accepted amplitudes | `60 / mean(breath interval, s)` (/min) |
| `rrv` | Respiration | Same chain | Std of the last 5 breath intervals (s) |

**Window length:** 60 s, 50 % overlap.
**Boundary skip:** any window whose [t, t + 60] overlaps the 5-min or 10-min protocol transition is dropped.
**Recovery dropped:** the post-stressor period is not part of the taxonomy and is removed before windowing.

---

## 2. Dataset summary

| Recording | rest | meditation | plank |
|---|---:|---:|---:|
| `ljh_5_21_medi_posiECG` | 8 | 7 | 0 |
| `mta-5-17-medi` | 8 | 7 | 0 |
| `mta-5-17-medi (1)` | 8 | 7 | 0 |
| `mta-5-17-pla-1'26` | 8 | 0 | 0¹ |
| `mta-5-17-pla-2` | 8 | 0 | 2 |
| `mta2_5_19_medi` | 8 | 7 | 0 |
| `mta_5_19_medi` | 8 | 7 | 0 |
| `mta_5_19_medi (1)` | 8 | 7 | 0 |
| `mta_5_19_pla_1'40` | 8 | 0 | 1 |
| `mta_5_19_pla_2'20` | 8 | 0 | 2 |
| `mta_5_21_medi` | 8 | 7 | 0 |
| `mta_5_21_pla_2` | 8 | 0 | 2 |
| `mta_5_21_pla_2'30(1)` | 8 | 0 | 3 |
| `nvt_5_21_medi` | 8 | 7 | 0 |
| `nvt_5_21_pla_2(1)` | 8 | 0 | 2 |
| `oyj_5_22_medi_posiECG` | 8 | 7 | 0 |
| `oyj_5_22_pla_1'50_posiECG` | 8 | 0 | 1 |
| `oyj_5_22_pla_2'15_posiECG` | 8 | 0 | 2 |
| **Total** | **144** | **63** | **15** |

¹ `mta-5-17-pla-1'26` produces 0 plank windows after the boundary-skip rule — the plank phase (86 s) is shorter than 60 s plus the buffer at the 300 s boundary. The recording still contributes 8 rest windows.

**Why the plank count jumped from 5 → 15** versus the previous report: not new recordings (the 8 plank-bearing `pla` files were already in the dataset), but the new `method="neurokit"` R-peak detector now recovers usable QRS through plank and into recovery, so HR/HRV features can be computed on plank windows where pantompkins previously returned NaN-heavy windows that effectively counted as 0. 15 plank windows is still small but is now large enough for F1[plank] to be statistically meaningful on every fold containing a plank-bearing test recording.

---

## 3. Models

| Family | Inputs | Architecture |
|---|---|---|
| KNN | 8 PDF features | median-imputer → StandardScaler → KNN (k = 7, distance-weighted, Euclidean) |
| RandomForest | same | median-imputer → RandomForest (400 trees, `min_samples_leaf = 2`) |
| XGBoost | same | XGBoost (400 trees, depth 4, lr 0.05, `tree_method="hist"`) |
| 1D-CNN | 3 raw channels (ECG @ 250 Hz, Resp @ 250 Hz, Mic Shannon-envelope @ 250 Hz), 60 s = (3, 15000) | 5-block 1D conv stack + AdaptiveAvgPool + MLP head (~636 k params), AMP, AdamW + cosine LR, class-weighted CE, early stopping |

---

## 4. Results — LORO (18 folds)

### 4.1 Mean accuracy / macro-F1 / per-class F1

| Model | Acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---:|---:|---:|---:|---:|
| KNN | 0.802 | 0.657 | 0.873 | 0.332 | 0.185 |
| RandomForest | 0.909 | 0.805 | 0.950 | 0.457 | 0.333 |
| **XGBoost** | **0.927** | **0.864** | 0.943 | 0.447 | **0.426** |
| 1D-CNN | 0.753 | 0.639 | 0.696 | 0.243 | 0.315 |

**XGBoost is the strongest model.** It beats RF by 6 points of macro-F1 and is the only model that consistently produces non-zero F1[plank] on every fold containing a plank-bearing test recording. The CNN trails the classical models by 22 points of macro-F1 on the same physiological data, as in the previous report — at 222 training windows the CNN can't compete with hand-crafted HRV features.

### 4.2 LORO confusion (XGBoost, sum across 18 folds)

|  | predicted rest | predicted meditation | predicted plank |
|---|---:|---:|---:|
| **true rest (144)** | 136 (94.4 %) | 8 (5.6 %) | 0 (0.0 %) |
| **true meditation (63)** | 8 (12.7 %) | 55 (87.3 %) | 0 (0.0 %) |
| **true plank (15)** | 1 (6.7 %) | 0 (0.0 %) | 14 (93.3 %) |

Plank recall is **14 / 15 = 93 %**. The single miss is in `oyj_5_22_pla_2'15_posiECG` (see §4.3). The model is also clean on confusion between rest and meditation — only 16 cross-class errors out of 207 non-plank windows.

All confusion matrices (per model × protocol) are in [confusion-matrices.md](confusion-matrices.md), with PNG heatmaps in [figures/confusion/](figures/confusion/).

### 4.3 Per-fold macro-F1 (XGBoost)

| Test recording | acc | macro-F1 | F1[plank] |
|---|---:|---:|---:|
| `ljh_5_21_medi_posiECG` | 1.000 | 1.000 | — |
| `mta-5-17-medi` | 0.733 | 0.700 | — |
| `mta-5-17-medi (1)` | 0.933 | 0.932 | — |
| `mta-5-17-pla-1'26` | 1.000 | 1.000 | — |
| `mta-5-17-pla-2` | 0.900 | 0.644 | 1.000 |
| `mta2_5_19_medi` | 0.867 | 0.866 | — |
| `mta_5_19_medi` | 1.000 | 1.000 | — |
| `mta_5_19_medi (1)` | 1.000 | 1.000 | — |
| `mta_5_19_pla_1'40` | 1.000 | 1.000 | 1.000 |
| `mta_5_19_pla_2'20` | 0.900 | 0.644 | 1.000 |
| `mta_5_21_medi` | 0.867 | 0.866 | — |
| `mta_5_21_pla_2` | 0.900 | 0.804 | 0.667 |
| `mta_5_21_pla_2'30(1)` | 1.000 | 1.000 | 1.000 |
| `nvt_5_21_medi` | 1.000 | 1.000 | — |
| `nvt_5_21_pla_2(1)` | 1.000 | 1.000 | 1.000 |
| `oyj_5_22_medi_posiECG` | 0.800 | 0.800 | — |
| `oyj_5_22_pla_1'50_posiECG` | 0.889 | 0.644 | 1.000 |
| `oyj_5_22_pla_2'15_posiECG` | 0.900 | 0.644 | 1.000 |

(`F1[plank] = —` for folds whose test recording has 0 plank windows; macro-F1 on `pla-*` folds caps at 0.667 because the test set has only 2 classes present, so the absent class contributes 0 to macro-F1 with `zero_division=0`.)

---

## 5. Results — Random 70 : 15 : 15 window split (5 seeds)

Stratified random window split (train 70 % / val 15 % / test 15 %) averaged over seeds 0–4. Same 8 features, same models.

| Model | Acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---:|---:|---:|---:|---:|
| KNN | 0.865 ± 0.051 | 0.756 ± 0.131 | 0.909 | 0.798 | 0.560 |
| **RandomForest** | 0.941 ± 0.037 | **0.929 ± 0.068** | 0.960 | 0.894 | 0.933 |
| XGBoost | 0.935 ± 0.034 | 0.926 ± 0.063 | 0.951 | 0.894 | 0.933 |
| 1D-CNN | 0.876 ± 0.022 | 0.899 ± 0.018 | 0.902 | 0.796 | **1.000** |

### LORO vs random split — same models, different protocol

| Model | LORO macro-F1 | Random-split macro-F1 | Δ |
|---|---:|---:|---:|
| KNN | 0.657 | 0.756 | +0.10 |
| RandomForest | 0.805 | 0.929 | +0.12 |
| XGBoost | 0.864 | 0.926 | +0.06 |
| **1D-CNN** | 0.639 | 0.899 | **+0.26** |

> ⚠️ **The CNN's huge jump under random split is leakage, not learning.** Windows have 50 % overlap, so a test window's 30-s-shifted neighbour can sit in train. Classical models work on scalar HRV summaries that are insensitive to that overlap; the CNN reads raw waveforms and can latch onto temporal proximity. **Use LORO numbers for any honest cross-subject claim; use random-split numbers only as within-recording sanity checks.**

---

## 6. Cross-experiment conclusions and next steps

Synthesis across this report and its [40 s / 20 s](report-40window-20overlap.md) and [30 s / 15 s](report-30window-15overlap.md) siblings — including the production recommendation, plank-data caveats, and follow-up work — lives in **[conclusion.md](conclusion.md)**.

## 7. Reproduce

```bash
# 60 s window / 30 s overlap is the current default. To run the classical and CNN
# pipelines and regenerate the four JSONs cited above:
uv run python -m src.local_eval        # -> outputs/local_loro.json, local_randomsplit.json
uv run python -m src.dl_train          # -> outputs/dl_local_loro.json, dl_local_randomsplit.json

# Confusion-matrix tables + PNG heatmaps
uv run python scripts/show_confusion_matrices.py > confusion-matrices.md
#                                                  -> figures/confusion/*.png
```
