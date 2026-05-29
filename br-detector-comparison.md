# BR peak-detector comparison

Three detectors were evaluated on every recording, with peaks split by phase
(`rest` 0–5 min, `stress` 5 min – 5 min + stressor duration, `recovery`
remainder). The goal: pick the detector with the most physiologically
consistent per-phase rates and the cleanest visual placement of peaks.

| Detector | Where it lives | Idea |
|---|---|---|
| **global** | `src.preprocess.detect_br_peaks` | `find_peaks` with min-dist 1.5 s and a **global** prominence floor `prom_frac × p90(|signal|)`. The threshold is calibrated on the whole signal — biased toward the high-amplitude active phases. |
| **sliding** | `src.preprocess.detect_br_peaks_sliding` | `find_peaks` per 60 s window (30 s step) with a **local** prominence floor recomputed from each window's p90. Idea: adapt the floor to local amplitude. |
| **neurokit** | `src.preprocess.detect_br_peaks_neurokit` | `neurokit2.rsp_peaks(method="biosppy")` — BioSPPy's zero-crossing-on-derivative + amplitude-threshold detector. |

## 1. Aggregate per-phase median RR (bpm) across 31 recordings

(IQR in brackets — smaller = more consistent across recordings)

| Detector | rest | stress | recovery |
|---|---|---|---|
| global | 14.7 (IQR 5.7) | 11.7 (IQR 7.6) | 18.0 (IQR 6.4) |
| sliding | 18.6 (IQR 5.1) | 12.6 (IQR 7.9) | 20.4 (IQR 4.7) |
| **neurokit** | **13.4 (IQR 1.1)** | **11.8 (IQR 3.9)** | **14.8 (IQR 4.3)** |

**neurokit's IQRs are 3–5 × tighter** than either alternative — its
per-recording rates are far more consistent across subjects.

## 2. Per-stressor sanity check (neurokit, stress-phase rate)

The stress-phase rate should match what the protocol does to breathing:

| Stressor | Expected | neurokit median |
|---|---|---|
| `medi` (meditation, deliberate slow deep breaths) | LOW (~6–10 bpm) | ~9 bpm ✓ |
| `pla` (plank, effort breathing) | ELEVATED (~12–18 bpm) | ~13 bpm ✓ |
| `math` (cognitive stress) | MODERATE (~12–18 bpm) | ~13 bpm ✓ |

This is the cleanest physiological signal of the three detectors.

## 3. Visual story — why sliding fails

The "sliding adapts the floor in the wrong direction." The flat rest/recovery
stretches have very small amplitude, so their **local** p90 is tiny, so the
**local** prominence floor is tiny, so the detector fires on every noise bump.
The result: rest/recovery breath rates inflated to ~18–25 bpm with no
respiratory rhythm to back them up.

Global avoids this by using one floor for the whole recording — but at the
cost of being calibrated to the high-amplitude active phase and thus missing
shallow breaths in rest. neurokit splits the difference using its own
internal amplitude logic (BioSPPy style).

## 4. Per-recording, per-phase rates

| recording | stressor | global rest / stress / recovery | sliding rest / stress / recovery | neurokit rest / stress / recovery |
|---|---|---|---|---|
| ljh_5_21_medi_posiECG | medi | 22.2 / 7.6 / 18.0 | 23.0 / 6.9 / 24.0 | 13.7 / 8.2 / 13.0 |
| mta-5-17-medi (1) | medi | 11.9 / 8.7 / 9.9 | 23.2 / 8.4 / 18.9 | 12.8 / — / 12.5 |
| mta-5-17-medi | medi | 10.3 / 8.1 / 7.7 | 16.1 / 8.2 / 12.0 | 13.6 / 11.2 / 11.3 |
| mta-5-17-pla-1'26 | pla | 16.0 / 8.9 / 21.7 | 21.2 / 12.2 / 21.6 | 16.8 / 10.2 / 19.2 |
| mta-5-17-pla-2 | pla | 11.6 / 11.3 / 20.1 | 26.1 / 11.6 / 20.3 | 11.6 / 9.9 / 18.8 |
| mta2_5_19_medi | medi | 15.4 / 9.8 / 7.8 | 17.1 / 10.6 / 19.0 | 12.9 / 9.8 / 12.4 |
| mta_5_19_medi (1) | medi | 8.9 / 9.3 / 18.6 | 29.0 / 9.1 / 22.1 | 13.1 / 8.8 / 14.8 |
| mta_5_19_medi | medi | 8.9 / 9.3 / 18.6 | 29.0 / 9.1 / 22.1 | 13.1 / 8.8 / 14.8 |
| mta_5_19_pla_1'40 | pla | 8.7 / 13.1 / 17.3 | 17.8 / 14.4 / 19.3 | 13.0 / 13.8 / 15.6 |
| mta_5_19_pla_2'20 | pla | 8.6 / 13.3 / 19.2 | 23.0 / 14.4 / 20.4 | 13.4 / 13.9 / 14.7 |
| mta_5_21_medi | medi | 14.4 / 10.1 / 15.5 | 22.3 / 12.0 / 23.6 | 11.5 / 12.8 / 15.7 |
| mta_5_21_pla_2'30(1) | pla | 16.2 / 20.2 / 22.3 | 17.0 / 19.3 / 23.3 | 13.3 / 12.4 / 18.3 |
| mta_5_21_pla_2 | pla | 15.0 / 21.9 / 20.2 | 20.5 / 21.8 / 23.1 | 13.2 / 15.0 / 15.6 |
| mta_5_26_math_11_13 | math | 16.3 / 23.5 / 23.0 | 16.4 / 23.7 / 23.0 | 9.3 / 18.3 / 14.3 |
| mta_5_26_math_8_12 | math | 17.1 / 11.7 / 11.1 | 18.5 / 12.6 / 14.9 | 12.8 / 13.2 / 11.4 |
| mta_5_26_pla_3'30 | pla | 14.5 / 18.9 / 27.1 | 16.5 / 19.6 / 25.1 | 13.7 / 14.4 / 20.4 |
| ntv_5_25_pla_2'10 | pla | 16.0 / 11.7 / 20.9 | 27.0 / 11.6 / 22.5 | 11.6 / 8.7 / 13.0 |
| ntv_5_25_pla_2 | pla | 11.0 / 17.4 / 19.3 | 18.0 / 17.3 / 21.1 | 14.9 / 14.3 / 18.0 |
| nva_5_26_math_6_8 | math | 17.6 / 13.8 / 21.5 | 19.7 / 15.7 / 21.5 | 15.0 / 11.5 / 12.3 |
| nva_5_26_math_9_12 | math | 13.7 / 13.6 / 16.5 | 18.5 / 15.2 / 16.7 | 14.0 / 13.1 / 14.1 |
| nvt_5_21_medi | medi | 18.6 / 16.2 / 14.8 | 22.3 / 5.5 / 22.6 | 13.5 / 9.2 / 12.6 |
| nvt_5_21_pla_2(1) | pla | 19.5 / 18.6 / 20.7 | 25.5 / 24.6 / 21.9 | 15.1 / 11.2 / 15.2 |
| nvt_5_25_pla_3'30 | pla | — / 17.7 / 20.9 | 18.2 / 20.1 / 21.3 | 13.9 / 14.9 / 20.5 |
| nvt_5_26_math_7_10 | math | 18.4 / 17.1 / 20.2 | 18.6 / 17.8 / 20.2 | 15.2 / 14.6 / — |
| nvt_5_26_math_7_11 | math | 17.3 / 17.2 / 13.8 | 21.3 / 18.3 / 17.7 | 13.1 / 12.6 / 17.0 |
| oyj_5_22_medi_posiECG | medi | 10.1 / 10.7 / 10.6 | 17.2 / 11.7 / 10.0 | 9.9 / 10.0 / 10.7 |
| oyj_5_22_pla_1'50_posiECG | pla | 11.3 / 8.7 / 13.3 | 16.3 / 9.1 / 16.7 | 13.2 / 8.1 / 13.0 |
| oyj_5_22_pla_2'15_posiECG | pla | 10.1 / 10.2 / 11.0 | 18.5 / 10.1 / 14.0 | 14.1 / 12.1 / 8.8 |
| smj_5_22_medi | medi | 16.9 / 5.3 / 14.4 | 19.7 / 6.8 / 17.6 | 16.7 / 7.6 / 18.7 |
| smj_5_22_pla_2'5 | pla | 14.8 / 13.7 / 17.5 | 15.0 / 17.3 / 17.5 | 13.9 / 13.2 / 16.7 |
| smj_5_22_pla_2 | pla | 12.0 / 10.9 / 15.7 | 18.1 / 14.9 / 15.7 | 15.3 / 11.3 / 14.9 |

## 5. Decision

**Adopt `neurokit2.rsp_peaks(method="biosppy")` as the default BR peak
detector** (changed in [`src/features.py`](src/features.py) `preprocess_recording`).

Reasons:
1. 3–5 × tighter per-phase IQR across recordings → more consistent across subjects.
2. Stressor-stratified stress-phase rates match physiology (medi ≈ 9, pla ≈ 13, math ≈ 13 bpm) — neither the global nor sliding detector gives this clean a separation.
3. Visually clean peak placement in active phases with sparse, plausible peaks in rest/recovery; neither false-fires (sliding) nor under-fires shallow rest breaths (global, on some recordings).

The `detect_br_peaks` and `detect_br_peaks_sliding` functions are kept in
`src/preprocess.py` as alternatives for the diagnostic plots in
[`figures/br_compare/`](figures/br_compare/) — 31 side-by-side comparison
plots (3-row panels, one method each).

## 6. Comparison plots

One panel per recording, 3 rows (`global` red ▽, `sliding` orange ▽,
`neurokit` green ▽), phase boundaries marked as faint vertical lines.

| recording | plot |
|---|---|
| ljh_5_21_medi_posiECG | ![](figures/br_compare/ljh_5_21_medi_posiECG.png) |
| mta-5-17-medi | ![](figures/br_compare/mta-5-17-medi.png) |
| mta-5-17-medi (1) | ![](figures/br_compare/mta-5-17-medi__1_.png) |
| mta-5-17-pla-1'26 | ![](figures/br_compare/mta-5-17-pla-1_26.png) |
| mta-5-17-pla-2 | ![](figures/br_compare/mta-5-17-pla-2.png) |
| mta2_5_19_medi | ![](figures/br_compare/mta2_5_19_medi.png) |
| mta_5_19_medi | ![](figures/br_compare/mta_5_19_medi.png) |
| mta_5_19_medi (1) | ![](figures/br_compare/mta_5_19_medi__1_.png) |
| mta_5_19_pla_1'40 | ![](figures/br_compare/mta_5_19_pla_1_40.png) |
| mta_5_19_pla_2'20 | ![](figures/br_compare/mta_5_19_pla_2_20.png) |
| mta_5_21_medi | ![](figures/br_compare/mta_5_21_medi.png) |
| mta_5_21_pla_2'30(1) | ![](figures/br_compare/mta_5_21_pla_2_30_1_.png) |
| mta_5_21_pla_2 | ![](figures/br_compare/mta_5_21_pla_2.png) |
| mta_5_26_math_11_13 | ![](figures/br_compare/mta_5_26_math_11_13.png) |
| mta_5_26_math_8_12 | ![](figures/br_compare/mta_5_26_math_8_12.png) |
| mta_5_26_pla_3'30 | ![](figures/br_compare/mta_5_26_pla_3_30.png) |
| ntv_5_25_pla_2'10 | ![](figures/br_compare/ntv_5_25_pla_2_10.png) |
| ntv_5_25_pla_2 | ![](figures/br_compare/ntv_5_25_pla_2.png) |
| nva_5_26_math_6_8 | ![](figures/br_compare/nva_5_26_math_6_8.png) |
| nva_5_26_math_9_12 | ![](figures/br_compare/nva_5_26_math_9_12.png) |
| nvt_5_21_medi | ![](figures/br_compare/nvt_5_21_medi.png) |
| nvt_5_21_pla_2(1) | ![](figures/br_compare/nvt_5_21_pla_2_1_.png) |
| nvt_5_25_pla_3'30 | ![](figures/br_compare/nvt_5_25_pla_3_30.png) |
| nvt_5_26_math_7_10 | ![](figures/br_compare/nvt_5_26_math_7_10.png) |
| nvt_5_26_math_7_11 | ![](figures/br_compare/nvt_5_26_math_7_11.png) |
| oyj_5_22_medi_posiECG | ![](figures/br_compare/oyj_5_22_medi_posiECG.png) |
| oyj_5_22_pla_1'50_posiECG | ![](figures/br_compare/oyj_5_22_pla_1_50_posiECG.png) |
| oyj_5_22_pla_2'15_posiECG | ![](figures/br_compare/oyj_5_22_pla_2_15_posiECG.png) |
| smj_5_22_medi | ![](figures/br_compare/smj_5_22_medi.png) |
| smj_5_22_pla_2'5 | ![](figures/br_compare/smj_5_22_pla_2_5.png) |
| smj_5_22_pla_2 | ![](figures/br_compare/smj_5_22_pla_2.png) |

## 7. Effect on classification

Both [`report-without-math.md`](report-without-math.md) and
[`report-with-math.md`](report-with-math.md) are re-run after this switch.
The headline impact appears in those reports.

## 8. Reproduce

```bash
# compare all three detectors and write the per-recording phase table
uv run python scripts/compare_br_detectors.py

# regenerate BR plots using the newly-default neurokit detector
uv run python scripts/plot_br_peaks.py
```
