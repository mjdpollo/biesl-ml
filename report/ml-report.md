# ML report — 4-class multimodal stress classification

> 4 classes: **rest / meditation / plank / math**. Companion to the Poincaré
> diagnostics in [`poincare-report.md`](poincare-report.md).

## Setup

- **Data.** 31 recordings, 10 subjects (`ljh`, `mta`, `mta2`, `nnn`,
  `ntv`, `nvt`, `nva`, `oyj`, `smj`, `tnq`). 8 .txt files moved to
  `data/_excluded/` for hardware / artifact reasons (all 5-17 files, the
  duplicate `mta_5_19_medi (1)`, `nvt_5_21_pla_2(1)`, `ntv_5_25_pla_2'10`,
  `mta_5_19_pla_1'40`, `oyj_5_22_pla_2'15_posiECG`, `oyj_5_22_pla_1'50_posiECG`,
  `mta_5_26_math_8_12`). Partial exclusions (rest-phase dropped from a
  single recording, stress kept): `mta_5_21_medi`,
  `oyj_5_22_medi_posiECG`, `nnn_5_29_pla_3`, `tnq_5_29_pla_2'20`,
  `tnq_5_29_math_7_12`. Full set in [`src/exclusions.py`](../src/exclusions.py).
- **Windowing.** Anchor-based at a **2-s slide**. Each feature uses its
  **own window length** centered on the anchor:

  | Feature | Window (s) |
  |---|---:|
  | HR              | 10 |
  | RMSSD / SD1 / SD2 / SD1/SD2 / SS | 60 |
  | RR              | 40 |
  | RRV             | 60 |
  | CSI             | 40 |

  Anchors near the rest → stress transition are dropped with an
  **asymmetric** −10 / +30 s buffer (`[290 s, 330 s]`), per patient
  guidance. The recovery phase is dropped entirely.
- **Features (9 total).** `csi, hr, hrv_rmssd, sd1, sd2, sd1_sd2, ss, rr,
  rrv`. LF / HF / LF-HF were removed: Welch on a sub-60-s tachogram is
  noise. Poincaré non-linear features replace them.
- **Window counts.** 6 020 anchors total — **3 146 rest / 960 meditation
  / 594 plank / 1 320 math**.
- **Models.** KNN (k=7, distance), RandomForest (400 trees), XGBoost
  (400 trees, depth 4); 1D-CNN on ECG + filtered Resp + Mic Shannon-
  envelope raw channels (3 × 10 000 samples per window, 40 s @ 250 Hz,
  AMP + AdamW + cosine LR + per-recording robust z-score).
- **Protocol — LORO only.** Leave one recording out (31 folds). Macro-F1
  is **pooled** over all held-out predictions (every recording covers
  only `rest` + one stressor, so per-fold averaging would charge zeros
  for 2-3 classes absent from each fold's test set). The 5-seed random
  70:15:15 split has been dropped — at a 2-s anchor step, neighbouring
  windows are near-duplicates so random splitting leaks heavily and the
  score collapses to ~1.0 for every model. It carries no information.

## Headline — pooled-LORO macro-F1

| Model | acc | macro-F1 | F1[rest] | F1[medi] | F1[plank] | F1[math] |
|---|---:|---:|---:|---:|---:|---:|
| KNN              | 0.648 | 0.593 | 0.73 | 0.71 | 0.43 | 0.50 |
| RandomForest     | 0.684 | 0.629 | 0.77 | 0.75 | 0.51 | 0.48 |
| XGBoost          | 0.732 | 0.690 | 0.80 | **0.85** | 0.58 | 0.53 |
| **1D-CNN**       | **0.773** | **0.767** | **0.85** | 0.72 | **0.96** | **0.55** |

## Confusion matrices (LORO, row-normalized)

| KNN | RandomForest | XGBoost | 1D-CNN |
|---|---|---|---|
| ![](figures/confusion/loro__knn.png) | ![](figures/confusion/loro__randomforest.png) | ![](figures/confusion/loro__xgboost.png) | ![](figures/confusion/loro__cnn.png) |

LORO row-normalized matrices in numeric form (rows = true class):

**XGBoost**

|             | rest | medi | plank | math |
|---|---:|---:|---:|---:|
| rest        | 0.85 | 0.02 | 0.02 | 0.11 |
| meditation  | 0.18 | 0.80 | 0.01 | 0.01 |
| plank       | 0.16 | 0.02 | 0.52 | 0.30 |
| math        | 0.42 | 0.00 | 0.07 | 0.51 |

**1D-CNN**

|             | rest | medi | plank | math |
|---|---:|---:|---:|---:|
| rest        | 0.87 | 0.01 | 0.00 | 0.12 |
| meditation  | 0.11 | 0.66 | 0.00 | 0.23 |
| plank       | 0.06 | 0.01 | 0.93 | 0.01 |
| math        | 0.33 | 0.12 | 0.01 | 0.55 |

## Findings

1. **The 1D-CNN wins LORO macro-F1 (0.767)** ahead of XGBoost (0.690).
   The CNN reads the raw filtered waveform so it has access to per-window
   morphology that the 9 hand-crafted features collapse away.
2. **The CNN nails `plank`** (F1 = 0.96, recall 0.93) — by far the best
   plank score across any model we've trained. The mic Shannon-energy
   envelope + ECG-amplitude pattern during effort appears to be highly
   discriminative.
3. **The CNN trades `meditation` for `plank`.** Where XGBoost reaches
   F1 = 0.85 on meditation, the CNN drops to 0.72 — meditation windows
   get pulled into `math` (23 % of the time per the confusion matrix).
   These are the two "low-HR/elevated-HRV" classes; the 1D conv stack
   doesn't separate them as cleanly as Poincaré + RR features do.
4. **`math` is still the hardest cross-subject class** (best F1 = 0.55
   from the CNN). 1 320 math windows across 6 subjects (`mta`, `nnn`,
   `nva`, `nvt`, `tnq`, plus `mta_5_26`) is on the edge of learnable —
   42 % of math windows are misclassified as `rest` by XGBoost (and
   33 % by the CNN), exposing how close math's autonomic signature is
   to a busy resting state for some subjects.
5. **`rest` is reliably called rest** by every model (F1 ≥ 0.73). The
   remaining 12–16 % of rest mass spills into `math`, never into the
   physical-effort classes — physiology is consistent here.

## Comparison to the previous pipeline (40-s 50 % overlap, 8 features)

For context — same dataset, previous windowing:

| | old (40-s / 50 %, 8 feats) | **new (anchor-based, 9 feats incl. Poincaré)** |
|---|---|---|
| windows | 477 | **6 020** |
| XGBoost LORO macro-F1 | 0.658 | **0.690** |
| 1D-CNN LORO macro-F1  | 0.636 | **0.767** |
| 1D-CNN F1[plank] LORO | 0.91  | **0.96** |
| 1D-CNN F1[math]  LORO | 0.22  | **0.55** |

Replacing LF/HF/LF-HF with the Poincaré set + tightening the boundary
buffer + reusing the same anchor grid across classical & CNN models
**moved the 4-class CNN from 0.636 → 0.767 LORO macro-F1**. The math
class alone improves by 0.33 F1.

## Reproduce

```bash
# 1. Refresh data from Google Drive (if needed; see README.md)
# 2. Rebuild features + raw windows + run all 4 models under LORO:
BR_PEAK_METHOD=neurokit uv run python scripts/run_split_reports.py

# 3. (Optional) Cross-detector ablation (global / sliding / neurokit):
uv run python scripts/run_detector_compare.py
```

Outputs land in `outputs/split_reports.json` (numbers) and
`figures/with_math/confusion/` (PNGs); both have been mirrored into
this report directory.
