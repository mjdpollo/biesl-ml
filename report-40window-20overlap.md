# Rest / Meditation / Plank Classification Report — 40 s window / 20 s overlap

**Task.** Per-window **3-class** classification: `rest` / `meditation` / `plank`.
**Window:** 40 s, 50 % overlap (step = 20 s). Boundary skip and recovery drop rules unchanged.
**Data.** Same 18 local recordings used in [report-60window-30overlap.md](report-60window-30overlap.md).
**Preprocessing.** Same chain: HP 1 Hz → notch 60 Hz → LP 150 Hz; R-peaks via `nk.ecg_peaks(method="neurokit")` + `nk.signal_fixpeaks(method="kubios", iterative=False)`. See [data-processing-using-neurokit.md](data-processing-using-neurokit.md) for the signal-side details.
**Evaluation protocols.** (a) LORO across 18 recordings; (b) 70 : 15 : 15 random window split, 5 seeds.

**Window-shrink caveat (HRV LF/HF).** `features.pdf` specifies Welch on 60 s NN segments for `hrv_lf` / `hrv_hf`. With a 40 s window the cleaned-NN series spans ~30–40 s, so `nperseg = min(len(tach), 60·fs_interp)` collapses to whatever the window provides. Bands are still computed, but the LF (0.04–0.15 Hz) estimate sits right at the spectral resolution limit — expect more noise in these two features than at 60 s. The model results should be interpreted with this in mind: macro-F1 improvements come mostly from the **larger sample size**, not from cleaner LF/HF.

---

## 1. Dataset summary

| Recording | rest | meditation | plank |
|---|---:|---:|---:|
| `ljh_5_21_medi_posiECG` | 13 | 12 | 0 |
| `mta-5-17-medi` | 13 | 12 | 0 |
| `mta-5-17-medi (1)` | 13 | 12 | 0 |
| `mta-5-17-pla-1'26` | 13 | 0 | 2 |
| `mta-5-17-pla-2` | 13 | 0 | 4 |
| `mta2_5_19_medi` | 13 | 12 | 0 |
| `mta_5_19_medi` | 13 | 12 | 0 |
| `mta_5_19_medi (1)` | 13 | 12 | 0 |
| `mta_5_19_pla_1'40` | 13 | 0 | 3 |
| `mta_5_19_pla_2'20` | 13 | 0 | 5 |
| `mta_5_21_medi` | 13 | 12 | 0 |
| `mta_5_21_pla_2` | 13 | 0 | 4 |
| `mta_5_21_pla_2'30(1)` | 13 | 0 | 5 |
| `nvt_5_21_medi` | 13 | 12 | 0 |
| `nvt_5_21_pla_2(1)` | 13 | 0 | 4 |
| `oyj_5_22_medi_posiECG` | 13 | 12 | 0 |
| `oyj_5_22_pla_1'50_posiECG` | 13 | 0 | 3 |
| `oyj_5_22_pla_2'15_posiECG` | 13 | 0 | 4 |
| **Total** | **234** | **108** | **34** |

**Window count vs the 60 s baseline:** 376 (vs 222) — roughly +70 % more training examples. Plank: 34 (vs 15) — **2.3× more plank windows**. Notably `mta-5-17-pla-1'26` (which produced 0 plank windows at 60 s because of the boundary skip) now contributes 2 plank windows at 40 s.

---

## 2. Results — LORO (18 folds)

### 2.1 Mean accuracy / macro-F1 / per-class F1

| Model | Acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---:|---:|---:|---:|---:|
| KNN | 0.835 | 0.742 | 0.886 | 0.406 | 0.365 |
| RandomForest | 0.932 | 0.849 | 0.948 | 0.459 | 0.466 |
| **XGBoost** | **0.945** | **0.879** | 0.958 | 0.458 | **0.484** |
| 1D-CNN | 0.820 | 0.740 | 0.754 | 0.340 | 0.481 |

### 2.2 LORO confusion (XGBoost, sum across 18 folds)

|  | predicted rest | predicted meditation | predicted plank |
|---|---:|---:|---:|
| **true rest (234)** | 228 (97.4 %) | 6 (2.6 %) | 0 (0.0 %) |
| **true meditation (108)** | 14 (13.0 %) | 94 (87.0 %) | 0 (0.0 %) |
| **true plank (34)** | 1 (2.9 %) | 1 (2.9 %) | 32 (94.1 %) |

Plank recall = **32 / 34 = 94.1 %**. Plank precision remains very high — only 0 false-positive plank predictions on 342 non-plank windows. The 1 plank → meditation confusion is the lone leak.

Confusion PNGs (per model × protocol) in [figures/confusion/win40_ov20/](figures/confusion/win40_ov20/); full markdown in [confusion-matrices-40window-20overlap.md](confusion-matrices-40window-20overlap.md).

### 2.3 Per-fold macro-F1 (XGBoost)

| Test recording | n_test | acc | macro-F1 | F1[plank] |
|---|---:|---:|---:|---:|
| `ljh_5_21_medi_posiECG` | 25 | 1.000 | 1.000 | — |
| `mta-5-17-medi` | 25 | 0.840 | 0.838 | — |
| `mta-5-17-medi (1)` | 25 | 0.880 | 0.877 | — |
| `mta-5-17-pla-1'26` | 15 | 1.000 | 1.000 | 1.000 |
| `mta-5-17-pla-2` | 17 | 1.000 | 1.000 | 1.000 |
| `mta2_5_19_medi` | 25 | 0.800 | 0.788 | — |
| `mta_5_19_medi` | 25 | 1.000 | 1.000 | — |
| `mta_5_19_medi (1)` | 25 | 1.000 | 1.000 | — |
| `mta_5_19_pla_1'40` | 16 | 1.000 | 1.000 | 1.000 |
| `mta_5_19_pla_2'20` | 18 | 0.889 | 0.639 | 1.000 |
| `mta_5_21_medi` | 25 | 0.960 | 0.960 | — |
| `mta_5_21_pla_2` | 17 | 0.941 | 0.619 | 0.857 |
| `mta_5_21_pla_2'30(1)` | 18 | 1.000 | 1.000 | 1.000 |
| `nvt_5_21_medi` | 25 | 1.000 | 1.000 | — |
| `nvt_5_21_pla_2(1)` | 17 | 0.941 | 0.910 | 0.857 |
| `oyj_5_22_medi_posiECG` | 25 | 0.880 | 0.879 | — |
| `oyj_5_22_pla_1'50_posiECG` | 16 | 0.938 | 0.653 | 1.000 |
| `oyj_5_22_pla_2'15_posiECG` | 17 | 0.941 | 0.653 | 1.000 |

(`F1[plank] = —` for folds whose test recording has 0 plank windows; macro-F1 on `pla-*` folds caps at 0.667 because the meditation class is absent in those test sets, with `zero_division=0`.)

---

## 3. Results — Random 70 : 15 : 15 window split (5 seeds)

| Model | Acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---:|---:|---:|---:|---:|
| KNN | 0.923 ± 0.031 | 0.890 ± 0.056 | 0.945 | 0.899 | 0.825 |
| RandomForest | 0.951 ± 0.034 | 0.941 ± 0.051 | 0.961 | 0.935 | 0.928 |
| **XGBoost** | **0.965 ± 0.031** | **0.958 ± 0.041** | 0.972 | 0.954 | 0.950 |
| 1D-CNN | 0.912 ± 0.011 | 0.928 ± 0.009 | 0.928 | 0.855 | **1.000** |

### LORO vs random split — same models, different protocol

| Model | LORO macro-F1 | Random-split macro-F1 | Δ |
|---|---:|---:|---:|
| KNN | 0.742 | 0.890 | +0.148 |
| RandomForest | 0.849 | 0.941 | +0.092 |
| XGBoost | 0.879 | 0.958 | +0.079 |
| **1D-CNN** | 0.740 | 0.928 | **+0.188** |

> ⚠️ The CNN's gap is still a leakage signature — windows overlap 50 %, so a test window's 20-s-shifted neighbour can sit in train. The classical-vs-CNN delta on this metric (+0.19 vs +0.08 for XGBoost) is the diagnostic; use LORO for any honest cross-subject claim.

---

## 4. Cross-experiment conclusions and next steps

See **[conclusion.md](conclusion.md)** for the synthesis across this report and its [60 s / 30 s](report-60window-30overlap.md) and [30 s / 15 s](report-30window-15overlap.md) siblings.

## 5. Reproduce

```bash
# The 40 s / 20 s overlap variant is produced by the variant driver. It edits
# src/features.py's WINDOW_S to 40.0, runs the full classical + CNN pipeline,
# and restores WINDOW_S to 60.0 on exit:
bash scripts/run_window_variants.sh
# -> outputs/win40_ov20/{local_loro,local_randomsplit,dl_local_loro,dl_local_randomsplit}.json

# Confusion-matrix tables + PNGs for this variant only:
uv run python scripts/show_confusion_matrices.py \
    --inputs-dir outputs/win40_ov20 \
    --figures-dir figures/confusion/win40_ov20 \
  > confusion-matrices-40window-20overlap.md
```
