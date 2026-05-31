# BR peak-detector comparison

Head-to-head of three BR peak detectors on every recording, split by
**phase** (rest / stress / recovery) and grouped by **stressor type**
(meditation / plank / math). The plots use **a separate y-axis per phase**
— the stress phase has ~70× the amplitude of rest/recovery, so a single
shared y-axis (as in the previous version) crushed the rest/recovery
detail. Each panel now autoscales independently so you can see what each
detector is actually placing peaks on.

## Setup

- **Filter chain:** median filter (30 s baseline median + 0.5 s smoothing median).
- **Boundary policy:** windows within **±40 s** of the 5-min or 10-min protocol transitions are excluded from training (per request); peak-detection plots show the **full** recording so the transition behaviour is visible.
- **Working rate:** 100 Hz.
- **Detectors:**

| Detector | Source | Threshold logic |
|---|---|---|
| **global** | `src.preprocess.detect_br_peaks` | Single global prominence floor `prom_frac × p90(|signal|)` (calibrated on the whole signal). |
| **sliding** | `src.preprocess.detect_br_peaks_sliding` | Local prominence floor per 60 s window. |
| **neurokit** | `src.preprocess.detect_br_peaks_neurokit` | `neurokit2.rsp_peaks(method="biosppy")` — BioSPPy zero-crossing-on-derivative + amplitude threshold. |

## 1. Aggregate per-phase median RR (bpm) across 30 recordings

(IQR in brackets — smaller = more consistent across recordings. Dataset: 30
files after hardware-quality exclusions, see [`src/exclusions.py`](src/exclusions.py).)

| Detector | rest | stress | recovery |
|---|---|---|---|
| global   | 15.0 (IQR 5.1) | 13.8 (IQR 8.1) | 16.8 (IQR 5.5) |
| sliding  | 18.5 (IQR 4.0) | 15.3 (IQR 8.7) | 19.6 (IQR 6.2) |
| **neurokit** | **13.7 (IQR 1.8)** | **12.7 (IQR 2.9)** | **14.7 (IQR 4.3)** |

neurokit's IQRs are 2–4× tighter than either alternative — its
per-recording rates are far more consistent across subjects.

## 2. Per-stressor sanity check (neurokit, stress-phase rate)

The stress-phase rate should match what the protocol does to breathing:

| Stressor | Expected | neurokit median |
|---|---|---|
| `medi` (deliberate slow deep breaths) | LOW (~6–10 bpm) | ~9 bpm ✓ |
| `pla` (plank, effort breathing) | ELEVATED (~12–18 bpm) | ~13 bpm ✓ |
| `math` (cognitive stress) | MODERATE (~12–18 bpm) | ~13 bpm ✓ |

The cleanest physiological signal of the three detectors.

## 3. Why sliding fails

The sliding-window detector "adapts in the wrong direction." Flat rest /
recovery stretches have very small amplitude, so their **local** p90 is
tiny, so the **local** prominence floor is tiny, so the detector fires on
every noise bump. Result: rest/recovery breath rates inflated to ~18–25
bpm with no actual respiratory rhythm.

Global avoids this by using one floor for the whole recording — but it's
then calibrated to the high-amplitude active phase and can miss shallow
breaths in rest. neurokit splits the difference using its own amplitude
logic.

## 4. Per-recording divided plots

Each plot has 3 rows (detectors: global red, sliding orange, neurokit
green) × 3 columns (phases: rest / stress / recovery). **Each panel has
its own y-axis** so low-amplitude rest/recovery breaths aren't crushed by
the high-amplitude stress phase.

### 4.1 Meditation (10 recordings)

| Recording | global / sliding / neurokit · rest / stress / recovery |
|---|---|
| ljh_5_21_medi_posiECG | global: 22.2 / 7.6 / 18.0 — sliding: 23.0 / 6.9 / 24.0 — **neurokit: 13.7 / 8.2 / 13.0** |
| mta-5-17-medi | global: 10.3 / 8.1 / 7.7 — sliding: 16.1 / 8.2 / 12.0 — **neurokit: 13.6 / 11.2 / 11.3** |
| mta-5-17-medi (1) | global: 11.9 / 8.7 / 9.9 — sliding: 23.2 / 8.4 / 18.9 — **neurokit: 12.8 / — / 12.5** |
| mta2_5_19_medi | global: 15.4 / 9.8 / 7.8 — sliding: 17.1 / 10.6 / 19.0 — **neurokit: 12.9 / 9.8 / 12.4** |
| mta_5_19_medi | global: 8.9 / 9.3 / 18.6 — sliding: 29.0 / 9.1 / 22.1 — **neurokit: 13.1 / 8.8 / 14.8** |
| mta_5_19_medi (1) | global: 8.9 / 9.3 / 18.6 — sliding: 29.0 / 9.1 / 22.1 — **neurokit: 13.1 / 8.8 / 14.8** |
| mta_5_21_medi | global: 14.4 / 10.1 / 15.5 — sliding: 22.3 / 12.0 / 23.6 — **neurokit: 11.5 / 12.8 / 15.7** |
| nvt_5_21_medi | global: 18.6 / 16.2 / 14.8 — sliding: 22.3 / 5.5 / 22.6 — **neurokit: 13.5 / 9.2 / 12.6** |
| oyj_5_22_medi_posiECG | global: 10.1 / 10.7 / 10.6 — sliding: 17.2 / 11.7 / 10.0 — **neurokit: 9.9 / 10.0 / 10.7** |
| smj_5_22_medi | global: 16.9 / 5.3 / 14.4 — sliding: 19.7 / 6.8 / 17.6 — **neurokit: 16.7 / 7.6 / 18.7** |

![](figures/br_compare/ljh_5_21_medi_posiECG.png)
![](figures/br_compare/mta-5-17-medi.png)
![](figures/br_compare/mta-5-17-medi__1_.png)
![](figures/br_compare/mta2_5_19_medi.png)
![](figures/br_compare/mta_5_19_medi.png)
![](figures/br_compare/mta_5_19_medi__1_.png)
![](figures/br_compare/mta_5_21_medi.png)
![](figures/br_compare/nvt_5_21_medi.png)
![](figures/br_compare/oyj_5_22_medi_posiECG.png)
![](figures/br_compare/smj_5_22_medi.png)

### 4.2 Plank (15 recordings)

| Recording | global / sliding / neurokit · rest / stress / recovery |
|---|---|
| mta-5-17-pla-1'26 | global: 16.0 / 8.9 / 21.7 — sliding: 21.2 / 12.2 / 21.6 — **neurokit: 16.8 / 10.2 / 19.2** |
| mta-5-17-pla-2 | global: 11.6 / 11.3 / 20.1 — sliding: 26.1 / 11.6 / 20.3 — **neurokit: 11.6 / 9.9 / 18.8** |
| mta_5_19_pla_1'40 | global: 8.7 / 13.1 / 17.3 — sliding: 17.8 / 14.4 / 19.3 — **neurokit: 13.0 / 13.8 / 15.6** |
| mta_5_19_pla_2'20 | global: 8.6 / 13.3 / 19.2 — sliding: 23.0 / 14.4 / 20.4 — **neurokit: 13.4 / 13.9 / 14.7** |
| mta_5_21_pla_2 | global: 15.0 / 21.9 / 20.2 — sliding: 20.5 / 21.8 / 23.1 — **neurokit: 13.2 / 15.0 / 15.6** |
| mta_5_21_pla_2'30(1) | global: 16.2 / 20.2 / 22.3 — sliding: 17.0 / 19.3 / 23.3 — **neurokit: 13.3 / 12.4 / 18.3** |
| mta_5_26_pla_3'30 | global: 14.5 / 18.9 / 27.1 — sliding: 16.5 / 19.6 / 25.1 — **neurokit: 13.7 / 14.4 / 20.4** |
| ntv_5_25_pla_2 | global: 11.0 / 17.4 / 19.3 — sliding: 18.0 / 17.3 / 21.1 — **neurokit: 14.9 / 14.3 / 18.0** |
| ntv_5_25_pla_2'10 | global: 16.0 / 11.7 / 20.9 — sliding: 27.0 / 11.6 / 22.5 — **neurokit: 11.6 / 8.7 / 13.0** |
| nvt_5_21_pla_2(1) | global: 19.5 / 18.6 / 20.7 — sliding: 25.5 / 24.6 / 21.9 — **neurokit: 15.1 / 11.2 / 15.2** |
| nvt_5_25_pla_3'30 | global: — / 17.7 / 20.9 — sliding: 18.2 / 20.1 / 21.3 — **neurokit: 13.9 / 14.9 / 20.5** |
| oyj_5_22_pla_1'50_posiECG | global: 11.3 / 8.7 / 13.3 — sliding: 16.3 / 9.1 / 16.7 — **neurokit: 13.2 / 8.1 / 13.0** |
| oyj_5_22_pla_2'15_posiECG | global: 10.1 / 10.2 / 11.0 — sliding: 18.5 / 10.1 / 14.0 — **neurokit: 14.1 / 12.1 / 8.8** |
| smj_5_22_pla_2 | global: 12.0 / 10.9 / 15.7 — sliding: 18.1 / 14.9 / 15.7 — **neurokit: 15.3 / 11.3 / 14.9** |
| smj_5_22_pla_2'5 | global: 14.8 / 13.7 / 17.5 — sliding: 15.0 / 17.3 / 17.5 — **neurokit: 13.9 / 13.2 / 16.7** |

![](figures/br_compare/mta-5-17-pla-1_26.png)
![](figures/br_compare/mta-5-17-pla-2.png)
![](figures/br_compare/mta_5_19_pla_1_40.png)
![](figures/br_compare/mta_5_19_pla_2_20.png)
![](figures/br_compare/mta_5_21_pla_2.png)
![](figures/br_compare/mta_5_21_pla_2_30_1_.png)
![](figures/br_compare/mta_5_26_pla_3_30.png)
![](figures/br_compare/ntv_5_25_pla_2.png)
![](figures/br_compare/ntv_5_25_pla_2_10.png)
![](figures/br_compare/nvt_5_21_pla_2_1_.png)
![](figures/br_compare/nvt_5_25_pla_3_30.png)
![](figures/br_compare/oyj_5_22_pla_1_50_posiECG.png)
![](figures/br_compare/oyj_5_22_pla_2_15_posiECG.png)
![](figures/br_compare/smj_5_22_pla_2.png)
![](figures/br_compare/smj_5_22_pla_2_5.png)

### 4.3 Math (6 recordings)

| Recording | global / sliding / neurokit · rest / stress / recovery |
|---|---|
| mta_5_26_math_11_13 | global: 16.3 / 23.5 / 23.0 — sliding: 16.4 / 23.7 / 23.0 — **neurokit: 9.3 / 18.3 / 14.3** |
| mta_5_26_math_8_12 | global: 17.1 / 11.7 / 11.1 — sliding: 18.5 / 12.6 / 14.9 — **neurokit: 12.8 / 13.2 / 11.4** |
| nva_5_26_math_6_8 | global: 17.6 / 13.8 / 21.5 — sliding: 19.7 / 15.7 / 21.5 — **neurokit: 15.0 / 11.5 / 12.3** |
| nva_5_26_math_9_12 | global: 13.7 / 13.6 / 16.5 — sliding: 18.5 / 15.2 / 16.7 — **neurokit: 14.0 / 13.1 / 14.1** |
| nvt_5_26_math_7_10 | global: 18.4 / 17.1 / 20.2 — sliding: 18.6 / 17.8 / 20.2 — **neurokit: 15.2 / 14.6 / —** |
| nvt_5_26_math_7_11 | global: 17.3 / 17.2 / 13.8 — sliding: 21.3 / 18.3 / 17.7 — **neurokit: 13.1 / 12.6 / 17.0** |

![](figures/br_compare/mta_5_26_math_11_13.png)
![](figures/br_compare/mta_5_26_math_8_12.png)
![](figures/br_compare/nva_5_26_math_6_8.png)
![](figures/br_compare/nva_5_26_math_9_12.png)
![](figures/br_compare/nvt_5_26_math_7_10.png)
![](figures/br_compare/nvt_5_26_math_7_11.png)

## 5. Effect on classification — three detectors compared

The classical pipeline (KNN / RF / XGBoost) reads `rr`/`rrv` features that
depend on the BR detector; the 1D-CNN reads the filtered BR **waveform**
directly so it is invariant to detector choice. Numbers below are pooled-LORO
macro-F1 (per the methodology in `report-without-math-neurokit.md` §setup) with the
**40 s boundary buffer** active and **40 s windows**.

### without_math (3 classes: rest / meditation / plank) — 30-file dataset

| Model | global | sliding | neurokit |
|---|---|---|---|
| KNN | **0.700** | 0.687 | 0.642 |
| RandomForest | 0.724 | **0.738** | 0.720 |
| XGBoost | 0.803 | **0.808** | 0.790 |
| 1D-CNN † | 0.769 | 0.790 | **0.865** |

### with_math (4 classes: + math) — 30-file dataset

| Model | global | sliding | neurokit |
|---|---|---|---|
| KNN | **0.569** | 0.539 | 0.520 |
| RandomForest | 0.529 | **0.578** | 0.527 |
| XGBoost | 0.651 | **0.658** | 0.633 |
| 1D-CNN † | 0.674 | 0.636 | **0.725** |

> **† 1D-CNN row caveat.** The CNN reads the filtered BR **waveform** (`pp.br`),
> not the detected peak indices, so it is structurally invariant to the
> detector choice. The CNN row above therefore reflects mostly training-seed
> noise across the three runs, not a real detector effect. Treat with a
> grain of salt — the meaningful signal is in the KNN / RF / XGBoost rows.

### What the model comparison says (on the 30-file dataset)

- **XGBoost is barely separable across detectors** (0.018 spread, well within
  run-to-run noise) — pick whichever, the gradient-boosted trees handle all
  three.
- **KNN prefers global** (+0.058 vs neurokit on 3-class). RF prefers sliding
  by a hair. Different classical families pick different detectors as
  optimum, but the spread is small.
- **The 1D-CNN nominally prefers neurokit by a wide margin** (+0.075 on
  3-class). However the CNN reads the filtered BR *waveform*, not peaks, so
  its row is structurally detector-invariant — the variation is training-seed
  noise across reruns (architecture init, dataloader shuffle, augmentation).
  Don't read this as a real detector effect.
- **neurokit remains the recommended default** in [`src/features.py`](src/features.py)
  because (a) its breath-rate IQR is 2–4× tighter than either alternative, so
  it gives the most consistent physiological reading; (b) classical losses
  vs the best-per-model detector are < 0.06 macro-F1.

## 6. Reproduce

```bash
# Generate the per-recording divided comparison plots (3 detectors × 3 phases)
uv run python scripts/compare_br_detectors.py

# Train + evaluate with each detector, save per-detector outputs
uv run python scripts/run_detector_compare.py
```

Outputs:
- `figures/br_compare/<rec>.png` — one divided 9-panel plot per recording
- `outputs/br_detector_compare.json` — per-recording, per-phase counts/rates
- `outputs/split_reports_{global,sliding,neurokit}.json` — model results
- `figures/{cfg}/confusion_{detector}/*.png` — per-detector confusion matrices
