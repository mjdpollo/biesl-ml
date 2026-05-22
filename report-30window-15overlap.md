# Rest / Meditation / Plank Classification Report — 30 s window / 15 s overlap

**Task.** Per-window **3-class** classification: `rest` / `meditation` / `plank`.
**Window:** 30 s, 50 % overlap (step = 15 s). Boundary skip and recovery drop rules unchanged.
**Data.** Same 18 local recordings used in [report-60window-30overlap.md](report-60window-30overlap.md).
**Preprocessing.** Same chain: HP 1 Hz → notch 60 Hz → LP 150 Hz; R-peaks via `nk.ecg_peaks(method="neurokit")` + `nk.signal_fixpeaks(method="kubios", iterative=False)`. See [data-processing-using-neurokit.md](data-processing-using-neurokit.md) for the signal-side details.
**Evaluation protocols.** (a) LORO across 18 recordings; (b) 70 : 15 : 15 random window split, 5 seeds.

**⚠️ Window-shrink caveat (HRV LF/HF).** `features.pdf` specifies Welch on 60 s NN segments for `hrv_lf` / `hrv_hf`. At a 30 s window, the cleaned-NN series often spans **less than 30 s of cumulative NN time**, which trips the early-return in [`features.py:_lf_hf`](src/features.py) (`if t[-1] - t[0] < 30.0: return nan, nan`). A substantial fraction of windows therefore have NaN LF / HF / LF_HF features, which the median imputer fills with the global median. This is the dominant reason that XGBoost LORO macro-F1 **drops** from 0.864 (60 s) and 0.879 (40 s) to 0.809 at 30 s — more training examples, but per-window feature quality is materially worse. F1[plank] is still tied with 40 s, because plank windows during high-effort phases tend to have enough valid NN time even at 30 s windows.

---

## 1. Dataset summary

| Recording | rest | meditation | plank |
|---|---:|---:|---:|
| `ljh_5_21_medi_posiECG` | 18 | 17 | 0 |
| `mta-5-17-medi` | 18 | 17 | 0 |
| `mta-5-17-medi (1)` | 18 | 17 | 0 |
| `mta-5-17-pla-1'26` | 18 | 0 | 3 |
| `mta-5-17-pla-2` | 18 | 0 | 6 |
| `mta2_5_19_medi` | 18 | 17 | 0 |
| `mta_5_19_medi` | 18 | 17 | 0 |
| `mta_5_19_medi (1)` | 18 | 17 | 0 |
| `mta_5_19_pla_1'40` | 18 | 0 | 4 |
| `mta_5_19_pla_2'20` | 18 | 0 | 7 |
| `mta_5_21_medi` | 18 | 17 | 0 |
| `mta_5_21_pla_2` | 18 | 0 | 6 |
| `mta_5_21_pla_2'30(1)` | 18 | 0 | 8 |
| `nvt_5_21_medi` | 18 | 17 | 0 |
| `nvt_5_21_pla_2(1)` | 18 | 0 | 6 |
| `oyj_5_22_medi_posiECG` | 18 | 17 | 0 |
| `oyj_5_22_pla_1'50_posiECG` | 18 | 0 | 5 |
| `oyj_5_22_pla_2'15_posiECG` | 18 | 0 | 7 |
| **Total** | **324** | **153** | **52** |

**Window count vs the 60 s baseline:** 529 (vs 222) — **2.4× more training examples**. Plank: 52 (vs 15) — **3.5× more plank windows**. Every plank-bearing recording now contributes ≥ 3 plank windows; `mta-5-17-pla-1'26` (which had 0 plank windows at 60 s) contributes 3.

---

## 2. Results — LORO (18 folds)

### 2.1 Mean accuracy / macro-F1 / per-class F1

| Model | Acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---:|---:|---:|---:|---:|
| KNN | 0.859 | 0.757 | 0.891 | 0.409 | 0.445 |
| RandomForest | 0.887 | 0.796 | 0.909 | 0.419 | 0.465 |
| **XGBoost** | **0.904** | **0.809** | 0.919 | 0.419 | **0.486** |
| 1D-CNN | 0.842 | 0.748 | 0.790 | 0.362 | 0.453 |

### 2.2 LORO confusion (XGBoost, sum across 18 folds)

|  | predicted rest | predicted meditation | predicted plank |
|---|---:|---:|---:|
| **true rest (324)** | 303 (93.5 %) | 20 (6.2 %) | 1 (0.3 %) |
| **true meditation (153)** | 32 (20.9 %) | 120 (78.4 %) | 1 (0.7 %) |
| **true plank (52)** | 1 (1.9 %) | 1 (1.9 %) | 50 (96.2 %) |

Plank recall = **50 / 52 = 96.2 %** (highest of the three window sizes). But the false-positive cost shows up too: 2 non-plank windows are misclassified as plank (1 rest, 1 meditation), where 60 s and 40 s both had 0 plank false positives. Meditation precision also drops (78.4 % vs 87.3 % at 60 s, 87.0 % at 40 s).

Confusion PNGs in [figures/confusion/win30_ov15/](figures/confusion/win30_ov15/); full markdown in [confusion-matrices-30window-15overlap.md](confusion-matrices-30window-15overlap.md).

### 2.3 Per-fold macro-F1 (XGBoost)

| Test recording | n_test | acc | macro-F1 | F1[plank] |
|---|---:|---:|---:|---:|
| `ljh_5_21_medi_posiECG` | 35 | 0.886 | 0.883 | — |
| `mta-5-17-medi` | 35 | 0.829 | 0.827 | — |
| `mta-5-17-medi (1)` | 35 | 0.771 | 0.770 | — |
| `mta-5-17-pla-1'26` | 21 | 1.000 | 1.000 | 1.000 |
| `mta-5-17-pla-2` | 24 | 0.958 | 0.636 | 0.909 |
| `mta2_5_19_medi` | 35 | 0.829 | 0.555 | — |
| `mta_5_19_medi` | 35 | 1.000 | 1.000 | — |
| `mta_5_19_medi (1)` | 35 | 1.000 | 1.000 | — |
| `mta_5_19_pla_1'40` | 22 | 1.000 | 1.000 | 1.000 |
| `mta_5_19_pla_2'20` | 25 | 0.920 | 0.625 | 0.933 |
| `mta_5_21_medi` | 35 | 0.943 | 0.942 | — |
| `mta_5_21_pla_2` | 24 | 0.917 | 0.618 | 0.909 |
| `mta_5_21_pla_2'30(1)` | 26 | 1.000 | 1.000 | 1.000 |
| `nvt_5_21_medi` | 35 | 0.886 | 0.883 | — |
| `nvt_5_21_pla_2(1)` | 24 | 0.958 | 0.657 | 1.000 |
| `oyj_5_22_medi_posiECG` | 35 | 0.543 | 0.542 | — |
| `oyj_5_22_pla_1'50_posiECG` | 23 | 0.826 | 0.625 | 1.000 |
| `oyj_5_22_pla_2'15_posiECG` | 25 | 1.000 | 1.000 | 1.000 |

(`F1[plank] = —` for folds whose test recording has 0 plank windows.) The `oyj_5_22_medi_posiECG` fold at acc=0.543 is the notable degradation vs the 60 s baseline (acc=0.800, macroF1=0.800) and 40 s (acc=0.880, macroF1=0.879) — the shorter window's noisier features land this `medi` recording in the cross-class confusion region.

---

## 3. Results — Random 70 : 15 : 15 window split (5 seeds)

| Model | Acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---:|---:|---:|---:|---:|
| KNN | 0.907 ± 0.033 | 0.908 ± 0.037 | 0.926 | 0.854 | 0.945 |
| RandomForest | 0.922 ± 0.015 | 0.931 ± 0.012 | 0.937 | 0.868 | 0.987 |
| **XGBoost** | **0.945 ± 0.017** | **0.950 ± 0.015** | 0.957 | 0.904 | 0.988 |
| 1D-CNN | 0.950 ± 0.029 | **0.959 ± 0.023** | 0.958 | 0.919 | **1.000** |

### LORO vs random split — same models, different protocol

| Model | LORO macro-F1 | Random-split macro-F1 | Δ |
|---|---:|---:|---:|
| KNN | 0.757 | 0.908 | +0.151 |
| RandomForest | 0.796 | 0.931 | +0.135 |
| XGBoost | 0.809 | 0.950 | +0.141 |
| **1D-CNN** | 0.748 | 0.959 | **+0.211** |

> ⚠️ Overlap-leakage is largest at the 30 s window because the 15 s step means a test window's nearest train neighbour is shifted by only 15 s. Even the classical Δ is now ~+0.14 (vs +0.08 at 40 s, +0.06 at 60 s), which means **random-split numbers at 30 s are mostly noise** — the CNN reaches macro-F1 0.959 by reading raw waveforms 15 s apart from each other, not by generalising.

---

## 4. Cross-experiment conclusions and next steps

See **[conclusion.md](conclusion.md)** for the synthesis across this report and its [60 s / 30 s](report-60window-30overlap.md) and [40 s / 20 s](report-40window-20overlap.md) siblings.

## 5. Reproduce

```bash
# The 30 s / 15 s overlap variant is produced by the variant driver. It edits
# src/features.py's WINDOW_S to 30.0, runs the full classical + CNN pipeline,
# and restores WINDOW_S to 60.0 on exit:
bash scripts/run_window_variants.sh
# -> outputs/win30_ov15/{local_loro,local_randomsplit,dl_local_loro,dl_local_randomsplit}.json

# Confusion-matrix tables + PNGs for this variant only:
uv run python scripts/show_confusion_matrices.py \
    --inputs-dir outputs/win30_ov15 \
    --figures-dir figures/confusion/win30_ov15 \
  > confusion-matrices-30window-15overlap.md
```
