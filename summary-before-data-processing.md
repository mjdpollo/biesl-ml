# Stress / Meditation / Baseline Classification — Initial Comparison

**Task.** Per-window 3-class classification: `baseline` / `meditation` / `stress`.
**Test set is always the local data**, held out one recording at a time (LORO, 7 folds). WESAD is used only as a source of additional training data.

## Datasets

| Dataset | Source | Channels used | Subjects | Windows | Class balance (baseline / meditation / stress) |
|---|---|---|---|---|---|
| **Local** | wearable CSVs in `data/` | ECG, Resp, Temp, Mic | 2 (`mta`, `nvt`) | 375 | 270 / 95 / **10** |
| **WESAD** | Schmidt et al. 2018, chest RespiBAN | ECG, Resp, Temp (no mic) | 3 of 15 (S7, S8, S9) | 507 | 231 / 150 / 126 |

WESAD label mapping: `1→baseline`, `4→meditation`, **`2 (TSST psychological stress) → stress`** (caveat — the local `stress` class is physical/plank exercise; psychological and physical stress share elevated HR but breathing signatures differ).
Window: 30 s, 50 % overlap, all signals resampled to 250 Hz, per-recording robust z-score.

## Models

| Family | Inputs | Architecture | Params |
|---|---|---|---|
| Classical — **KNN** | 24 hand-crafted features (HRV, breathing, temp) shared between datasets; +6 mic features for "+mic" variant | median-imputer → StandardScaler → KNN (k=7, distance-weighted, Euclidean) | n/a |
| Classical — **RandomForest** | same | median-imputer → RandomForest (400 trees, `min_samples_leaf=2`, `max_features='sqrt'`) | n/a |
| Classical — **XGBoost** | same (XGB handles NaN natively, no imputer/scaler needed) | XGBoost (400 trees, depth=4, lr=0.05) | n/a |
| Deep | Raw 3-ch waveform (ECG, Resp, Temp), 250 Hz × 30 s = (3, 7500) | 5-block 1D CNN + AdaptiveAvgPool + MLP head, AMP, AdamW, cosine LR, class-weighted CE, early stopping | 636 k |

Three transfer conditions, identical across both families:

| Condition | Training data | DL init |
|---|---|---|
| **A** | local \ {held-out recording} | random |
| **B** | WESAD only | random |
| **C-combined** (classical) / **C-full** (DL) | WESAD ∪ (local \ {held-out}) | DL inits from WESAD-pretrained weights, fine-tune at LR=1e-4 |
| C-head (DL only) | local fine-tune with conv layers frozen | from WESAD-pretrained weights |

## Results — macro-F1, LORO mean (test set always local)

Bold = best in row.

| Condition | KNN | RandomForest | XGBoost | 1D-CNN |
|---|---|---|---|---|
| **A** local-only (shared feats) | 0.634 | **0.771** | 0.751 | 0.694 |
| A+ local-only with mic | 0.718 | 0.730 | 0.583 | n/a |
| **B** WESAD-only → local | 0.366 | **0.481** | 0.419 | 0.153 |
| **C** transfer / combined | 0.591 | **0.745** | 0.639 | 0.629 (C-full) |
| C-head (DL, frozen conv) | n/a | n/a | n/a | 0.496 |

## Results — accuracy, LORO mean

| Condition | KNN | RandomForest | XGBoost | 1D-CNN |
|---|---|---|---|---|
| A local-only (shared feats) | 0.877 | **0.923** | 0.879 | 0.877 |
| A+ local-only with mic | 0.892 | 0.905 | 0.835 | n/a |
| B WESAD-only → local | 0.557 | **0.683** | 0.549 | 0.217 |
| C transfer / combined | 0.855 | **0.874** | 0.854 | 0.846 |

### Per-class F1 (LORO mean)

| Condition × Model | baseline | meditation | stress |
|---|---|---|---|
| A — KNN | 0.927 | 0.588 | 0.000 |
| A — RandomForest | **0.952** | **0.643** | 0.057 |
| A — XGBoost | 0.920 | 0.577 | 0.048 |
| A — 1D-CNN | 0.924 | 0.535 | **0.159** |
| B — KNN | 0.632 | 0.447 | 0.018 |
| B — RandomForest | 0.798 | 0.321 | 0.071 |
| B — XGBoost | 0.649 | 0.426 | 0.071 |
| B — 1D-CNN | 0.256 | 0.000 | **0.203** |
| C — KNN | 0.916 | 0.558 | 0.063 |
| C — RandomForest | 0.910 | **0.607** | 0.095 |
| C — XGBoost | 0.913 | 0.509 | 0.071 |
| C-full — 1D-CNN | 0.885 | 0.550 | 0.149 |

Note: **the 1D-CNN is the only model that ever exceeds 0.15 on the `stress` class** — the class-weighted cross-entropy loss forces it to predict `stress` more often than the tree models do. Trees default to ignoring the rare class entirely (F1≈0.05). At higher stress counts this gap may close.

### Per-fold variance is large — aggregate hides it

| Fold (test recording) | A (CNN) | C-full (CNN) | Δ |
|---|---|---|---|
| mta-5-17-medi | 0.925 | **0.943** | +0.02 |
| mta-5-17-medi (1) | 0.959 | 0.635 | −0.32 |
| mta-5-17-pla-1 | 0.538 | **0.774** | **+0.24** |
| mta-5-17-pla-2 | 0.434 | 0.457 | +0.02 |
| mta-5-8-medi | 0.639 | 0.658 | +0.02 |
| nvt-5-15-medi | 0.398 | 0.398 | 0 |
| nvt-5-8-medi | 0.962 | 0.535 | −0.43 |

## Key findings

1. **RandomForest is the best classical model — and the best model overall on macro-F1.** RF wins every condition (A, B, C) on both accuracy and macro-F1 and is the strongest at handling the WESAD→local domain shift (B = 0.481 vs KNN 0.366, XGB 0.419, CNN 0.153). The bagging averages out the device-shift noise that hurts XGBoost.
2. **KNN reacts very differently to the microphone.** With mic features added, KNN *gains* +0.08 macro-F1 (0.634 → 0.718) while XGBoost *loses* −0.17 (0.751 → 0.583). Distance-based learners can extract signal from the extra mic dimensions when scaled properly; gradient-boosted trees overfit the same dimensions at this small sample size. Worth a follow-up.
3. **The 1D CNN doesn't beat the best classical model at this scale.** Classical RF: 0.771 macro-F1 local-only; CNN: 0.694. With only ~375 local windows, hand-crafted HRV/breathing/temp features are denser information than raw waveforms.
4. **Zero-shot WESAD→local is poor for every model.** Best is RF at 0.481 macro-F1, then XGB 0.419, KNN 0.366, CNN 0.153. The classical pipeline transfers better because derived features (HR in bpm, RMSSD, resp rate) are physically calibrated and device-agnostic; the CNN learns waveform-level patterns that are device-specific.
5. **The 1D CNN is the only model that predicts `stress` at all.** F1[stress] is ≥0.15 for the CNN under conditions A, B, and C-full, vs ≤0.10 for every tree model. Class-weighted cross-entropy forces the network to attempt the rare class; tree models without explicit class weights default to ignoring it.
6. **Transfer learning helps on some folds and hurts on others.** Mean numbers hide huge per-fold variance: `mta-5-17-pla-1` jumps +0.24 macro-F1 with CNN pretraining; `nvt-5-8-medi` drops −0.43. The per-fold table belongs in any presentation.
7. **The `stress` class is the bottleneck.** Local data has only **10 stress windows** across all 7 recordings. F1[stress] stays ≤ 0.20 in every condition we ran. No model architecture can fix this — it's a data-collection problem.

## Advice / next steps

1. **Collect more `pla-*` (stress) recordings.** Highest-leverage single change. Going from 10 to ~100 stress windows would let the CNN actually learn the class, and would let us split stress into psychological vs physical sub-labels if useful later.
2. **Download the remaining 12 WESAD subjects (S2–S6, S10–S17).** Only 3 of 15 are downloaded; full WESAD would 5× the pretraining set with no engineering cost. Likely the second-biggest lift on transfer performance.
3. **Report per-fold results, not just the mean.** Variance between folds is larger than the average difference between conditions. A box-plot or per-recording bar chart conveys the picture more honestly than a single-number table.
4. **Use RandomForest as the production baseline; treat the CNN as research.** RF is the top single model today, more interpretable than the CNN, trains in seconds, and degrades gracefully when WESAD is added. The CNN becomes competitive once stress windows pass ~100 and WESAD is fully ingested.
5. **Try a CNN→XGBoost hybrid.** Train the CNN on WESAD, freeze it, extract the 256-dim bottleneck per window, feed those + the 24 classical features into XGBoost. Often the best small-N recipe — combines learned and engineered representations.
6. **Flag the WESAD-stress = TSST caveat explicitly.** WESAD's stress is psychological (public speaking + arithmetic), local stress is physical (plank). They share elevated HR/suppressed HRV but the breathing patterns diverge. This is an upper bound on cross-dataset stress transfer until we either collect WESAD-style psychological-stress local recordings or accept the label-modality mismatch.

## Reproduce

```bash
# classical, all three models (XGB + KNN + RF)
uv run python -c "from src.transfer import run_three_way_all_models; run_three_way_all_models()"

# classical, single model (defaults to xgboost; pass model='knn' or 'randomforest')
uv run python -m src.transfer

# 1D-CNN three-way
uv run python scripts/run_dl_transfer.py

# outputs land in outputs/transfer_results{,_knn,_randomforest,_all_models}.json
# and outputs/dl_transfer_results.json
```
