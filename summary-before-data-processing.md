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

KNN (k=7, distance-weighted, Euclidean), RandomForest (400 trees, `min_samples_leaf=2`), XGBoost (400 trees, depth=4, lr=0.05).

## Results — LORO mean, local-only

### Configuration (i): PDF features only — 8 features

| Model | Acc | macro-F1 | F1[baseline] | F1[meditation] | F1[stress] |
|---|---|---|---|---|---|
| KNN | 0.856 | 0.686 | 0.903 | 0.556 | 0.000 |
| RandomForest | 0.901 | 0.764 | 0.934 | 0.595 | 0.000 |
| **XGBoost** | 0.897 | **0.825** | 0.930 | 0.577 | **0.143** |

### Configuration (ii): PDF + skin temperature — 11 features

| Model | Acc | macro-F1 | F1[baseline] | F1[meditation] | F1[stress] |
|---|---|---|---|---|---|
| KNN | 0.829 | 0.623 | 0.892 | 0.441 | 0.000 |
| RandomForest | 0.886 | 0.747 | 0.922 | 0.571 | 0.000 |
| XGBoost | 0.922 | 0.792 | 0.947 | 0.636 | 0.000 |

### Delta — (with temperature) − (PDF only)

| Model | Δ accuracy | Δ macro-F1 | Δ F1[baseline] | Δ F1[meditation] | Δ F1[stress] |
|---|---|---|---|---|---|
| KNN | −0.027 | **−0.063** | −0.011 | −0.115 | 0.000 |
| RandomForest | −0.015 | −0.018 | −0.011 | −0.024 | 0.000 |
| XGBoost | +0.025 | −0.033 | +0.018 | +0.059 | **−0.143** |

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

## Key findings

1. **The PDF feature set works.** XGBoost on just 8 PDF features lands macro-F1 **0.825** under LORO — better than the previous 30-feature pipeline at 30 s windows (0.751 with the same model). Cleaner physiology features beat more features at this data scale.
2. **Adding temperature hurts macro-F1 for every model** (−0.018 to −0.063). The biggest collapse is on XGBoost's `stress` F1: 0.143 → 0.000. Temperature carries information about the majority class (baseline) but pushes the model away from the rare `stress` class. Accuracy slightly improves for XGBoost (+0.025) because the model trades stress recall for more correct baseline predictions.
3. **XGBoost beats RandomForest beats KNN** under the new feature set (macro-F1 0.825 / 0.764 / 0.686). This reverses the ordering from the old 30-feature run, where RF was strongest. The 8 PDF features are too few for KNN's nearest-neighbour vote to be informative; trees thrive on them; gradient boosting wins outright.
4. **The `stress` class is the bottleneck — even harder than before.** Only **4 stress windows** exist at the 60 s schema (was 10 at 30 s). Most LORO folds have zero stress in train and zero in test; F1[stress] is structurally near zero. Only XGBoost reaches 0.143 on it. **Collecting more `pla-*` recordings is the single highest-leverage next step.**
5. **The per-recording HRV LF/HF computation is now spec-compliant.** Previously we ran Welch on 30 s segments; the PDF requires ≥ 1 min. Moving to 60 s windows fixes this, and the corresponding macro-F1 lift suggests the previous LF/HF values were noisy.

## Advice / next steps

1. **Adopt the PDF feature set as the canonical pipeline.** Use XGBoost-on-PDF-features as the production baseline (macro-F1 0.825). Drop temperature from the feature list — it never beats the PDF set on macro-F1 and erases the only working `stress` predictions.
2. **Collect more `pla-*` (stress) recordings.** Highest-leverage single change: with only 4 stress windows total, no model can learn the class. Going from 4 to ~50 stress windows would dramatically improve every metric. If feasible, also collect math/cognitive stress recordings so the class has within-subject variety.
3. **Report per-fold spread.** Aggregate macro-F1 hides the 0.46–1.00 fold-to-fold range. A per-recording bar chart in the team presentation will be more honest than a single mean.
4. **Optional follow-up — WESAD re-evaluation under the PDF pipeline.** Once the local pipeline is finalized, re-run the WESAD pretraining condition with the same 8-feature schema to see whether transfer behaves differently when both sides use the same physiology features. (Out of scope for this run; user said local-only first.)

## Reproduce

```bash
# Build features + run local-only LORO with the temperature ablation
uv run python -m src.local_eval

# Output:
#   outputs/local_loro_temp_ablation.json  (per-fold + summary, both configs)
```
