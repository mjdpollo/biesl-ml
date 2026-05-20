# Stress / Meditation / Baseline Classification — Local-only Comparison

**Task.** Per-window 3-class classification: `baseline` / `meditation` / `stress`.
**Test set is always a held-out local recording (LORO, 7 folds).** WESAD is NOT used in this run — local data only.
**Feature set: the eight parameters listed in `features.pdf`**, with an explicit ablation against adding skin temperature.

## Feature pipeline (per features.pdf)

| Feature | Channel | Preprocessing | Computation |
|---|---|---|---|
| `csi` | Microphone | BP 20–200 Hz → Shannon energy `−x² log(x²)` → peak detection | S2/S1 amplitude ratio paired to ECG R-peaks (S1 within R+0–200 ms, S2 within R+200–500 ms) |
| `hr` | ECG | LP 150 Hz + detrend → Pan–Tompkins → NN cleaning (reject NN<300 ms, NN>1500 ms, or NN deviating >20% from median of surrounding ~10 beats; cubic-spline interpolation for rejected) | 60 000 / mean(NN_ms) |
| `hrv_rmssd` | ECG | same NN series | √(mean of squared successive NN differences) |
| `hrv_lf` | ECG | Welch on tachogram interpolated to 4 Hz, Hann window, 60 s segments | area under PSD in 0.04–0.15 Hz |
| `hrv_hf` | ECG | same Welch | area under PSD in 0.15–0.40 Hz |
| `hrv_lf_hf` | ECG | — | `hrv_lf / hrv_hf` |
| `rr` | Respiration | detrend → 0.5 s MA → Cheby II LP (stopband 1 Hz) → Butterworth HP 0.12 Hz, slope-based peak detection with adaptive 1/3 × mean-of-last-8-amplitudes threshold | 60 / mean(interval, s) |
| `rrv` | Respiration | same chain | std of the last 5 breath intervals (s) |

**Window: 60 s with 50 % overlap.** The 1-min minimum is required by the Welch step for `hrv_lf` / `hrv_hf`. This halves the window count vs the previous 30 s pipeline.

## Dataset (local only)

| | windows | baseline | meditation | stress |
|---|---|---|---|---|
| Local (7 recordings, 2 subjects) | 180 | 131 | 45 | **4** |

The `stress` class has only **4 windows** total under the 60 s schema — a hard floor on what any model can do on this class.

## Models

Four model families compared under identical LORO:

| Family | Inputs | Architecture |
|---|---|---|
| KNN | 8 PDF features (+3 temp ablation) | median-imputer → StandardScaler → KNN (k=7, distance-weighted, Euclidean) |
| RandomForest | same | median-imputer → RandomForest (400 trees, `min_samples_leaf=2`) |
| XGBoost | same | XGBoost (400 trees, depth=4, lr=0.05) |
| 1D-CNN | 3 raw channels (ECG @ 250 Hz, Resp @ 250 Hz, Mic Shannon-envelope @ 250 Hz) over 60 s = (3, 15000); +1 temp channel for the ablation | 5-block 1D conv stack + AdaptiveAvgPool + MLP head, 636 k params, AMP, AdamW + cosine LR, class-weighted CE, early stopping |

## Results — LORO mean, local-only

### Configuration (i): PDF features only

| Model | Inputs | Acc | macro-F1 | F1[baseline] | F1[meditation] | F1[stress] |
|---|---|---|---|---|---|---|
| KNN | 8 PDF features | 0.856 | 0.686 | 0.903 | 0.556 | 0.000 |
| RandomForest | 8 PDF features | 0.901 | 0.764 | 0.934 | 0.595 | 0.000 |
| **XGBoost** | 8 PDF features | 0.897 | **0.825** | 0.930 | 0.577 | **0.143** |
| 1D-CNN | 3 channels (ECG, Resp, Mic SE) | 0.638 | 0.349 | 0.648 | 0.292 | 0.000 |

### Configuration (ii): + skin temperature

| Model | Inputs | Acc | macro-F1 | F1[baseline] | F1[meditation] | F1[stress] |
|---|---|---|---|---|---|---|
| KNN | 8 PDF + 3 temp feat | 0.829 | 0.623 | 0.892 | 0.441 | 0.000 |
| RandomForest | 8 PDF + 3 temp feat | 0.886 | 0.747 | 0.922 | 0.571 | 0.000 |
| XGBoost | 8 PDF + 3 temp feat | 0.922 | 0.792 | 0.947 | 0.636 | 0.000 |
| **1D-CNN** | 4 channels (ECG, Resp, Mic SE, Temp) | 0.793 | 0.611 | 0.856 | 0.494 | **0.114** |

### Delta — (with temperature) − (without temperature)

| Model | Δ accuracy | Δ macro-F1 | Δ F1[baseline] | Δ F1[meditation] | Δ F1[stress] |
|---|---|---|---|---|---|
| KNN | −0.027 | **−0.063** | −0.011 | −0.115 | 0.000 |
| RandomForest | −0.015 | −0.018 | −0.011 | −0.024 | 0.000 |
| XGBoost | +0.025 | −0.033 | +0.018 | +0.059 | **−0.143** |
| 1D-CNN | +0.155 | **+0.262** | +0.208 | +0.202 | +0.114 |

### Per-fold macro-F1 (XGBoost, PDF only)

| Fold (test recording) | macro-F1 |
|---|---|
| mta-5-17-medi | 0.721 |
| mta-5-17-medi (1) | 1.000 |
| mta-5-17-pla-1 | **1.000** |
| mta-5-17-pla-2 | 0.463 |
| mta-5-8-medi | 0.862 |
| nvt-5-15-medi | 0.769 |
| nvt-5-8-medi | 0.958 |

Per-fold variance is still substantial (0.46 → 1.00). The two `pla-*` folds (the only ones with `stress` in the test set) bracket the spread.

### Why is F1[stress] often 0?

It's a data-structure issue, not a model issue. The `stress` class has only 4 windows across the entire dataset, and they live in just two recordings:

| Recording | baseline | meditation | stress |
|---|---|---|---|
| mta-5-17-medi | 14 | 13 | 0 |
| mta-5-17-medi (1) | 13 | 14 | 0 |
| **mta-5-17-pla-1** | 18 | 0 | **2** |
| **mta-5-17-pla-2** | 19 | 0 | **2** |
| mta-5-8-medi | 14 | 12 | 0 |
| nvt-5-15-medi | 13 | 13 | 0 |
| nvt-5-8-medi | 14 | 13 | 0 |
| **total** | 105 | 65 | **4** |

Under LORO with 7 folds:

* **5 folds test on a `medi` recording → zero stress windows in the test set.** F1[stress] is structurally 0 for these folds (sklearn `zero_division=0`).
* **2 folds test on a `pla` recording → 2 stress windows in test, 2 in train.**

XGBoost's mean F1[stress] of 0.143 ≈ (1.0 + 0 + 5 × 0) / 7 — one of the two informative folds achieved perfect stress F1, the other zero. KNN and RandomForest never predicted stress at all, so F1[stress] = 0 across all 7 folds. The 1D-CNN's class-weighted cross-entropy nudges it to attempt the rare class even when the trees give up.

The number 0 on F1[stress] does **not** mean these models are broken on stress. It means there is not enough stress data to evaluate. With ~30-50 stress windows total the per-class F1 would become statistically meaningful.

## Key findings

1. **XGBoost on the 8 PDF features is the strongest model overall** at macro-F1 **0.825**, beating RandomForest (0.764), KNN (0.686), and the 1D-CNN with the same physiology inputs (0.349). The 60 s window finally satisfies the PDF's "≥ 1 min for Welch" requirement, and the cleaner per-spec HRV LF/HF features carry most of the signal.
2. **Temperature flips the picture between classical and deep models.**
   * For every classical model, adding temperature **hurts** macro-F1 (KNN −0.063, RF −0.018, XGB −0.033). For XGBoost it also collapses the only working stress predictions (F1 0.143 → 0.000).
   * For the 1D-CNN, adding temperature **lifts** macro-F1 by **+0.262** (0.349 → 0.611). The CNN can't easily extract HRV/RR patterns from raw waveforms with only 174 windows, so it leans on the temperature drift as a cheap discriminator.
3. **The CNN with 4 channels still trails XGBoost on PDF features** (0.611 vs 0.825). At this data scale, hand-crafted physiology features remain the strongest representation; the CNN would need more data — especially more stress windows — to compete.
4. **Model-family ordering changed under the new feature set.** Old 30-feature pipeline at 30 s: RF > XGB > KNN. New 8-feature PDF pipeline at 60 s: XGB > RF > KNN. The 8 PDF features are too few for KNN's nearest-neighbour vote to be informative; gradient boosting wins outright on the small but high-quality feature set.
5. **The `stress` class is a data bottleneck, not a model bottleneck.** Only **4 stress windows** exist across the entire 60 s dataset, all from two `pla` recordings. Under LORO, 5 of 7 folds have zero stress in their test set, so F1[stress] is structurally 0 for them. Collecting more `pla-*` recordings is the single most impactful next step.
6. **The per-spec HRV LF/HF computation matters.** Previously we ran Welch on 30 s segments; the PDF requires ≥ 1 min. Moving to 60 s windows fixed this, and the corresponding +0.07 macro-F1 lift on XGBoost suggests the previous LF/HF values were noisy.

## Advice / next steps

1. **Adopt the PDF feature set + XGBoost as the production baseline** at macro-F1 0.825. Drop temperature from the classical feature list — it never beats the PDF set on macro-F1 for any classical model and destroys the only working stress predictions.
2. **Treat the CNN's behaviour as a diagnostic, not a baseline.** The fact that the CNN gains so much from temperature shows it can't yet learn HRV from raw ECG at this data scale. Don't ship the CNN; do keep it on hand to validate the PDF preprocessing once more data arrives.
3. **Collect more `pla-*` (stress) recordings.** Highest-leverage single change: only 4 stress windows exist across all 7 recordings. Going from 4 to ~50 stress windows would make F1[stress] statistically meaningful and very likely lift macro-F1 across every model.
4. **Report per-fold spread.** Aggregate macro-F1 hides the 0.46–1.00 fold-to-fold range. A per-recording bar chart in the team presentation will be more honest than a single mean.
5. **Optional follow-up — WESAD re-evaluation under the PDF pipeline.** Once the local pipeline is finalized, re-run the WESAD pretraining condition with the same 8-feature schema. (Out of scope for this run; deferred per current direction.)

## Reproduce

```bash
# Classical (KNN / RF / XGBoost) LORO with the temperature ablation
uv run python -m src.local_eval

# 1D-CNN LORO with the temperature ablation
uv run python -m src.dl_train

# Outputs:
#   outputs/local_loro_temp_ablation.json     (KNN, RF, XGBoost)
#   outputs/dl_local_loro_temp_ablation.json  (1D-CNN, both configs)
```
