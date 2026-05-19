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
| Classical | 24 hand-crafted features (HRV, breathing, temp) shared between datasets; +6 mic features for "+mic" variant | XGBoost (n_est=400, depth=4) | n/a |
| Deep | Raw 3-ch waveform (ECG, Resp, Temp), 250 Hz × 30 s = (3, 7500) | 5-block 1D CNN + AdaptiveAvgPool + MLP head, AMP, AdamW, cosine LR, class-weighted CE, early stopping | 636 k |

Three transfer conditions, identical across both families:

| Condition | Training data | DL init |
|---|---|---|
| **A** | local \ {held-out recording} | random |
| **B** | WESAD only | random |
| **C-combined** (classical) / **C-full** (DL) | WESAD ∪ (local \ {held-out}) | DL inits from WESAD-pretrained weights, fine-tune at LR=1e-4 |
| C-head (DL only) | local fine-tune with conv layers frozen | from WESAD-pretrained weights |

## Results — LORO mean (test set always local)

| Condition | XGBoost acc | XGBoost macro-F1 | 1D-CNN acc | 1D-CNN macro-F1 |
|---|---|---|---|---|
| **A** local-only (shared feats) | 0.879 | **0.751** | 0.877 | 0.694 |
| A+ local-only with mic | 0.835 | 0.583 | n/a | n/a |
| **B** WESAD-only → local | 0.549 | 0.419 | 0.217 | 0.153 |
| **C** transfer / combined | 0.854 | 0.639 | 0.846 | 0.629 (C-full) |
| C-head (DL, frozen conv) | n/a | n/a | 0.750 | 0.496 |

### Per-class F1 (LORO mean)

| Condition | baseline | meditation | stress |
|---|---|---|---|
| A — XGB local-only | 0.920 | 0.577 | **0.048** |
| A — CNN local-only | 0.924 | 0.535 | **0.159** |
| B — XGB WESAD-only | 0.649 | 0.426 | 0.071 |
| B — CNN WESAD-only | 0.256 | 0.000 | 0.203 |
| C — XGB combined | 0.913 | 0.509 | 0.071 |
| C-full — CNN pretrain + FT | 0.885 | 0.550 | 0.149 |

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

1. **Hand-crafted features are still the strongest single approach.** XGBoost on HRV+resp+temp features beats the 1D CNN at this dataset size (0.751 vs 0.694 macro-F1 local-only). The CNN needs more data to compete; classical scales gracefully to small N.
2. **The microphone (CPS phonocardiogram branch) hurts at this scale.** Local-only XGBoost loses 0.17 macro-F1 when the mic features are added — overfitting on heterogeneous mic noise across only 7 recordings. Mic strategies need revisiting once more recordings are available.
3. **Zero-shot WESAD→local is poor for both model families** (0.42 XGB, 0.15 CNN). The classical pipeline transfers better because derived features (HR in bpm, RMSSD, resp rate) are physically calibrated and device-agnostic; the CNN learns waveform-level patterns that are device-specific.
4. **Transfer learning helps on some folds and hurts on others.** Mean C-full ≈ mean A, but `mta-5-17-pla-1` jumps by +0.24 macro-F1 with pretraining while `nvt-5-8-medi` drops by 0.43. Reporting only the mean is misleading; the per-fold table belongs in any presentation.
5. **The `stress` class is the bottleneck.** Local data has only **10 stress windows** across all 7 recordings. F1[stress] stays ≤ 0.20 in every condition we ran. No model architecture can fix this — it's a data-collection problem.

## Advice / next steps

1. **Collect more `pla-*` (stress) recordings.** Highest-leverage single change. Going from 10 to ~100 stress windows would let the CNN actually learn the class, and would let us split stress into psychological vs physical sub-labels if useful later.
2. **Download the remaining 12 WESAD subjects (S2–S6, S10–S17).** Only 3 of 15 are downloaded; full WESAD would 5× the pretraining set with no engineering cost. Likely the second-biggest lift on transfer performance.
3. **Report per-fold results, not just the mean.** Variance between folds is larger than the average difference between conditions. A box-plot or per-recording bar chart conveys the picture more honestly than a single-number table.
4. **Use XGBoost as the production baseline; treat the CNN as research.** At today's data scale, the classical pipeline is more accurate, more interpretable, and trains in seconds without a GPU. The CNN becomes competitive once stress windows pass ~100 and WESAD is fully ingested.
5. **Try a CNN→XGBoost hybrid.** Train the CNN on WESAD, freeze it, extract the 256-dim bottleneck per window, feed those + the 24 classical features into XGBoost. Often the best small-N recipe — combines learned and engineered representations.
6. **Flag the WESAD-stress = TSST caveat explicitly.** WESAD's stress is psychological (public speaking + arithmetic), local stress is physical (plank). They share elevated HR/suppressed HRV but the breathing patterns diverge. This is an upper bound on cross-dataset stress transfer until we either collect WESAD-style psychological-stress local recordings or accept the label-modality mismatch.

## Reproduce

```bash
# classical (XGBoost) three-way
uv run python -m src.transfer

# 1D-CNN three-way
uv run python scripts/run_dl_transfer.py

# outputs land in outputs/{transfer_results.json, dl_transfer_results.json}
```
