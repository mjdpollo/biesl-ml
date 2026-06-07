# ML report — curated 20-file set, wavelet ECG filter, sliding BR detector

> 4 classes: **rest / meditation / plank / math**. Companion to the
> Poincaré diagnostics in [`poincare-report.md`](poincare-report.md).
> Built for the prof meeting (2026-06-08).

## What changed since the previous report

1. **+4 recordings** vs the previous 16-file curated set: `ljh_6_5_pla_2`,
   `ljh_6_5_pla_2(1)`, `oyj_6_6_math_11`, `smj_6_6_math_17`.
   Plank now covers **2 subjects** again (`mta` + `ljh`); math covers
   **4 subjects** (`mta`, `nvt`, `oyj`, `smj`).
2. **ECG filter** replaced: Butterworth 1–150 Hz + mains-notch → **wavelet
   denoise + 5–45 Hz band-keep** (`sym4` DWT, Donoho soft-threshold on the
   in-band detail levels, out-of-band detail and the deepest
   approximation zeroed). See [`src/preprocess.py:filter_ecg`](../src/preprocess.py).
3. **Feature set unchanged.** 8 features: `csi`, `hr`, `hrv_rmssd`,
   `sd2_sd1`, `sd1_x_sd2`, `ss`, `rr`, `rrv`.

> **Note on the file list.** The user-provided "good file" list said
> `oyj_6_6_math_6`, but the GDrive math folder only contains
> `oyj_6_6_math_11.txt` — assumed to be a typo and used `math_11`. If
> `math_6` is the intended file it should be re-uploaded and the run
> repeated.

## Setup

- **Data — 20 recordings, 5 subjects** (`mta` ×12, `mta2`, `ljh` ×2,
  `nvt` ×2, `oyj`, `smj` ×2). Stressor coverage:

  | Medi (6) | Plank (6) | Math (8) |
  |---|---|---|
  | `mta_5_19_medi` | `mta_5_19_pla_2'20(1)` | `mta_5_26_math_11_13` |
  | `mta2_5_19_medi` | `mta_5_26_pla_3'30` | `mta_6_3_math_8` |
  | `mta_6_4_medi` | `mta_6_4_pla_2` | `mta_6_3_math_10` |
  | `mta_6_4_medi(1)` | `mta_6_4_pla_2'10` | `mta_6_3_math_10(1)` |
  | `nvt_5_21_medi` | `ljh_6_5_pla_2` | `mta_6_3_math_14` |
  | `smj_5_22_medi` | `ljh_6_5_pla_2(1)` | `nvt_5_26_math_7_10` |
  | | | `oyj_6_6_math_11` |
  | | | `smj_6_6_math_17` |

- **Features (8).** `csi, hr, hrv_rmssd, sd2_sd1, sd1_x_sd2, ss, rr, rrv`.
  SD1 / SD2 are computed internally only.

- **ECG filter (new).** Wavelet (sym4 DWT) — 5–45 Hz band-keep with
  Donoho soft-threshold denoising on the in-band detail coefficients.
  No mains notch needed (60 Hz lives entirely in D2 which is zeroed).
  Replaces the previous Butterworth 1–150 Hz + mains-notch chain.
- **BR peak detector: sliding** (60-s window stepped by 30 s, local p90
  prominence floor).

- **Windowing.** Anchor-based at 2-s slide. Per-feature centered windows:

  | Feature | Window (s) |
  |---|---:|
  | HR              | 10 |
  | RMSSD / SD2_SD1 / SD1×SD2 / SS | 60 |
  | RR              | 40 |
  | RRV             | 60 |
  | CSI             | 40 |

  Asymmetric −10 / +30 s buffer around the 5-min cue. Recovery dropped.

- **Window counts.** 4 340 anchors — **2 420 rest / 720 meditation / 240
  plank / 960 math**.

- **Models.** KNN (k=7, distance), RandomForest (400 trees), XGBoost
  (400 trees, depth 4); 1D-CNN on ECG + filtered Resp + Mic
  Shannon-envelope (3 ch × 10 000 samples, 40 s @ 250 Hz).

- **Protocol — LORO only** (20 folds, pooled macro-F1).

## Headline — pooled-LORO macro-F1

| Model | acc | macro-F1 | F1[rest] | F1[medi] | F1[plank] | F1[math] |
|---|---:|---:|---:|---:|---:|---:|
| KNN              | 0.717 | 0.596 | 0.81 | 0.69 | 0.30 | 0.57 |
| RandomForest     | 0.744 | 0.653 | 0.84 | 0.76 | 0.45 | 0.57 |
| XGBoost          | 0.776 | 0.707 | 0.85 | 0.81 | 0.56 | **0.60** |
| **1D-CNN**       | **0.793** | **0.753** | **0.87** | **0.93** | **0.79** | 0.48 |

The 1D-CNN wins macro-F1 (0.75) and accuracy (0.79), driven by the best
`meditation` (F1 0.93) and `plank` (0.79) per-class numbers. XGBoost
remains the best classical model and the only model whose `math` F1
beats the CNN's.

## Confusion matrices (LORO, row-normalized)

| KNN | RandomForest | XGBoost | 1D-CNN |
|---|---|---|---|
| ![](figures/confusion/loro__knn.png) | ![](figures/confusion/loro__randomforest.png) | ![](figures/confusion/loro__xgboost.png) | ![](figures/confusion/loro__cnn.png) |

Numeric form (rows = true class):

**XGBoost**

|             | rest | medi | plank | math |
|---|---:|---:|---:|---:|
| rest        | 0.85 | 0.05 | 0.00 | 0.10 |
| meditation  | 0.11 | **0.84** | 0.00 | 0.05 |
| plank       | 0.06 | 0.00 | 0.51 | 0.42 |
| math        | 0.28 | 0.05 | 0.07 | 0.60 |

**1D-CNN**

|             | rest | medi | plank | math |
|---|---:|---:|---:|---:|
| rest        | 0.87 | 0.05 | 0.01 | 0.06 |
| meditation  | 0.01 | **0.93** | 0.05 | 0.02 |
| plank       | 0.05 | 0.07 | **0.88** | 0.00 |
| math        | 0.31 | 0.19 | 0.02 | 0.48 |

## Comparison vs the previous run (16-file curated set, Butterworth ECG)

| Model | prev (16 files, Butterworth) | **current (20 files, wavelet)** | Δ |
|---|---:|---:|---:|
| KNN          | 0.534 | 0.596 | **+0.062** |
| RandomForest | 0.615 | 0.653 | **+0.038** |
| XGBoost      | 0.652 | 0.707 | **+0.055** |
| 1D-CNN       | 0.686 | 0.753 | **+0.067** |

Per-class deltas — XGBoost:

| Class | prev | curr | Δ |
|---|---:|---:|---:|
| rest        | 0.87 | 0.85 | −0.02 |
| meditation  | 0.91 | 0.81 | −0.10 |
| **plank**   | 0.27 | **0.56** | **+0.29** |
| math        | 0.59 | 0.60 | +0.01 |

Per-class deltas — 1D-CNN:

| Class | prev | curr | Δ |
|---|---:|---:|---:|
| rest        | 0.85 | 0.87 | +0.02 |
| meditation  | 0.68 | **0.93** | **+0.25** |
| **plank**   | 0.51 | **0.79** | **+0.28** |
| math        | 0.71 | 0.48 | −0.23 |

**Reading.**

1. **Plank is rescued by the `ljh` subject.** Adding the two `ljh_6_5_pla_*`
   recordings means LORO finally has *cross-subject* plank evidence —
   when an `mta` plank fold is held out, the training pool now contains
   `ljh` plank data, and vice versa. Plank F1 jumps 0.27 → 0.56
   (XGBoost) and 0.51 → 0.79 (CNN).
2. **CNN meditation also jumps** (0.68 → 0.93) — likely because the
   additional `oyj`/`smj` math subjects let the CNN form a sharper
   "math fingerprint", which in turn reduces the medi→math
   misclassifications that hurt the previous run.
3. **XGBoost meditation regresses** slightly (0.91 → 0.81) — the
   classical feature set is less able to separate the broader medi pool
   from the (now also broader) math pool when the only autonomic axes
   are HR + Poincaré ratios.
4. **CNN math regresses** (0.71 → 0.48) — the previous run had a tight
   `mta_6_3_math_*` cluster the CNN over-fit on; with more math subjects
   (smj, oyj) that fingerprint dilutes. This is a more honest
   cross-subject number.
5. **Wavelet ECG filter** is hard to isolate from the data-size change,
   but the across-the-board ~+0.05 LORO macro-F1 lift suggests it isn't
   hurting. Mains rejection is cleaner (60 Hz lives entirely in D2 and
   is zeroed by band-keep), and the Donoho soft threshold likely
   suppresses high-frequency muscle noise during plank.

## TL;DR (for the prof meeting)

- **20 recordings, 5 subjects, 4 classes.** All "good files" per the
  curator review.
- **1D-CNN reaches pooled-LORO macro-F1 0.753**, accuracy 0.79.
  Per-class: rest 0.87, **meditation 0.93**, **plank 0.88 recall / 0.79
  F1**, math 0.48.
- XGBoost (best classical): macro-F1 0.707, accuracy 0.78, **math F1 0.60**.
- Compared to the previous 16-file Butterworth run, every model gained
  0.04–0.07 macro-F1; plank improved most (+0.28 CNN, +0.29 XGBoost)
  because plank finally has 2 subjects.
- The wavelet 5–45 Hz ECG filter replaces the prior Butterworth chain
  and includes built-in mains rejection.
- **Math remains the hardest class for the CNN**, but XGBoost handles it
  better. Worth discussing whether to ensemble.

## Reproduce

```bash
uv run python scripts/run_split_reports.py            # main training
uv run python scripts/dump_preprocessed_nn.py         # NN/BR dump
uv run python scripts/plot_poincare.py                # Poincaré plots
```

Outputs:

| File | Contents |
|---|---|
| `outputs/split_reports.json` | per-class F1, pooled confusion matrices |
| `figures/with_math/confusion/loro__*.png` | confusion-matrix heatmaps |
| `outputs/preprocessed_nn.json` | per-recording NN + BR intervals |
| `figures/poincare/*.png` | Poincaré scatter plots |
