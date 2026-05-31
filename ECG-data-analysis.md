# ECG-based HR / HRV analysis

Per-recording, per-phase medians of the ECG-derived features computed by
[`src.features.preprocess_recording`](src/features.py) on the 40 s windows
**after** all hardware-quality exclusions are applied
(see [`src/exclusions.py`](src/exclusions.py)). This page is the ECG-side
companion to [`br-detector-comparison.md`](br-detector-comparison.md).

`n` is the number of windows that fell in that phase after exclusions;
medians collapse the per-window features. These values are independent of
the BR peak detector (they come from the ECG path), so the same numbers
hold for the global / sliding / neurokit runs in `outputs/split_reports_*.json`.


These values are independent of the BR peak detector — they come from the
ECG channel and the clean-NN-interval pipeline (`src.preprocess.clean_nn_intervals`),
applied to the 40-s windows after the partial / boundary exclusions. `n` is
the number of windows that fell in that phase after exclusions; the medians
collapse the per-window features.

### 7.1 Aggregate medians per stressor type

| Stressor | Phase | HR (bpm) | RMSSD (ms) | LF (ms²) | HF (ms²) | LF/HF |
|---|---|---|---|---|---|---|
| medi | rest | 71.6 | 39.7 | 244 | 256 | 0.94 |
| medi | **stress** | 73.7 | 42.7 | 2 073 | 396 | **6.71** |
| pla  | rest | 75.2 | 34.2 | 286 | 250 | 1.06 |
| pla  | **stress** | **87.3** | 41.7 | 196 | 168 | 1.54 |
| math | rest | 79.0 | 30.0 | 300 | 171 | 1.05 |
| math | **stress** | **86.3** | 24.5 | 144 | 165 | 0.90 |

Physiology sanity check:
- **HR rises in pla and math stress** (rest ~75–79 → stress ~86–87 bpm) but is essentially flat in **medi** (~72 → ~74). Meditation's deliberate slow deep breathing keeps HR low even though it's labelled "stress" phase.
- **LF/HF jumps to ~6.7 in medi stress** — the deep slow breaths concentrate power in the very-low end of the LF band. pla and math stay around 1.0–1.5, the normal sympatho-vagal regime.
- **RMSSD drops in math stress** (30 → 24 ms) — cognitive stress reduces beat-to-beat variability slightly. pla stress *raises* RMSSD because effort breathing increases respiratory sinus arrhythmia.

### 7.2 Per-recording, per-phase HR/HRV

#### Meditation (8 recordings)

| Recording | Phase | n | HR (bpm) | RMSSD (ms) | LF (ms²) | HF (ms²) | LF/HF |
|---|---|---|---|---|---|---|---|
| ljh_5_21_medi_posiECG | rest | 11 | 69.2 | 43.7 | 417 | 267 | 1.67 |
| ljh_5_21_medi_posiECG | stress | 8 | 71.8 | 42.9 | 4 511 | 175 | 24.66 |
| mta2_5_19_medi | rest | 11 | 69.9 | 47.6 | 448 | 825 | 0.91 |
| mta2_5_19_medi | stress | 8 | 76.0 | 50.8 | 1 960 | 600 | 2.61 |
| mta_5_19_medi | rest | 11 | 76.0 | 27.9 | 189 | 245 | 0.94 |
| mta_5_19_medi | stress | 8 | 76.1 | 48.6 | 1 540 | 499 | 3.26 |
| mta_5_21_medi | rest | 3 | 60.7 | 55.6 | 940 | 622 | 1.51 |
| mta_5_21_medi | stress | 8 | 68.0 | 36.1 | 1 549 | 312 | 4.98 |
| nvt_5_21_medi | rest | 11 | 66.5 | 31.8 | 98 | 171 | 0.40 |
| nvt_5_21_medi | stress | 8 | 64.3 | 52.5 | 7 940 | 279 | 26.51 |
| nvt_5_29_medi | rest | 11 | 72.6 | 47.4 | 165 | 588 | 0.30 |
| nvt_5_29_medi | stress | 8 | 74.9 | 39.7 | 459 | 432 | 1.23 |
| oyj_5_22_medi_posiECG | rest | 11 | 73.7 | 28.0 | 282 | 209 | 1.42 |
| oyj_5_22_medi_posiECG | stress | 8 | 70.2 | 27.8 | 2 185 | 117 | 18.46 |
| smj_5_22_medi | rest | 11 | 76.5 | 49.9 | 207 | 253 | 0.65 |
| smj_5_22_medi | stress | 8 | 80.8 | 42.5 | 4 069 | 499 | 8.45 |

#### Plank (14 recordings)

| Recording | Phase | n | HR (bpm) | RMSSD (ms) | LF (ms²) | HF (ms²) | LF/HF |
|---|---|---|---|---|---|---|---|
| mta_5_19_pla_2'20 | rest | 11 | 74.1 | 34.7 | 693 | 504 | 1.23 |
| mta_5_19_pla_2'20 | stress | 3 | 85.0 | 29.7 | 223 | 110 | 1.54 |
| mta_5_21_pla_2 | rest | 11 | 66.0 | 42.3 | 300 | 447 | 0.90 |
| mta_5_21_pla_2 | stress | 2 | 79.7 | 113.3 | – | – | – |
| mta_5_21_pla_2'30(1) | rest | 11 | 63.2 | 35.9 | 84 | 352 | 0.21 |
| mta_5_21_pla_2'30(1) | stress | 3 | 85.0 | 29.7 | 223 | 110 | 1.54 |
| mta_5_26_pla_3'30 | rest | 11 | 89.4 | 34.7 | 136 | 517 | 0.26 |
| mta_5_26_pla_3'30 | stress | 6 | 107.8 | 21.8 | 67 | 53 | 1.92 |
| mta_5_29_pla_2'35 | rest | 11 | 67.2 | 69.8 | 497 | 1 478 | 0.31 |
| mta_5_29_pla_2'35 | stress | 3 | 89.9 | 49.2 | 35 | 97 | 0.28 |
| mta_5_29_pla_4 | rest | 11 | 67.6 | 66.8 | 341 | 1 400 | 0.26 |
| mta_5_29_pla_4 | stress | 8 | 87.3 | 41.5 | 168 | 201 | 0.58 |
| nnn_5_29_pla_3 | rest | 11 | 94.4 | 16.8 | 368 | 45 | 6.95 |
| nnn_5_29_pla_3 | stress | 5 | 83.2 | 70.2 | – | – | – |
| ntv_5_25_pla_2 | rest | 11 | 83.1 | 18.6 | 95 | 84 | 2.55 |
| ntv_5_25_pla_2 | stress | 2 | 46.5 | 152.0 | – | – | – |
| nvt_5_25_pla_3'30 | rest | 11 | 76.3 | 22.0 | 94 | 130 | 0.81 |
| nvt_5_25_pla_3'30 | stress | 6 | 72.5 | 111.0 | 3 110 | 539 | 5.77 |
| oyj_5_22_pla_1'50_posiECG | rest | 11 | 72.9 | 21.8 | 430 | 155 | 2.36 |
| oyj_5_22_pla_2'15_posiECG | rest | 11 | 72.1 | 21.9 | 370 | 64 | 5.19 |
| smj_5_22_pla_2 | rest | 11 | 76.4 | 46.8 | 462 | 358 | 1.19 |
| smj_5_22_pla_2 | stress | 2 | 132.2 | 23.4 | – | – | – |
| smj_5_22_pla_2'5 | rest | 11 | 86.3 | 33.6 | 202 | 353 | 0.88 |
| smj_5_22_pla_2'5 | stress | 2 | 95.0 | 41.9 | – | – | – |
| tnq_5_29_pla_2'20 | rest | 11 | 94.1 | 14.3 | 92 | 43 | 1.96 |
| tnq_5_29_pla_2'20 | stress | 3 | 137.8 | 40.6 | 673 | 251 | 3.26 |

#### Math (8 recordings)

| Recording | Phase | n | HR (bpm) | RMSSD (ms) | LF (ms²) | HF (ms²) | LF/HF |
|---|---|---|---|---|---|---|---|
| mta_5_26_math_11_13 | rest | 11 | 87.5 | 33.9 | 114 | 565 | 0.17 |
| mta_5_26_math_11_13 | stress | 8 | 91.1 | 27.7 | 86 | 170 | 0.58 |
| mta_5_26_math_8_12 | rest | 4 | 82.2 | 77.2 | 491 | 751 | 0.57 |
| mta_5_26_math_8_12 | stress | 8 | 89.2 | 35.1 | 109 | 214 | 0.47 |
| nnn_5_29_math_6_8 | rest | 11 | 89.3 | 32.5 | 428 | 68 | 5.68 |
| nnn_5_29_math_6_8 | stress | 8 | 94.1 | 23.4 | 546 | 138 | 3.85 |
| nva_5_26_math_6_8 | rest | 11 | 78.8 | 21.6 | 300 | 103 | 3.66 |
| nva_5_26_math_6_8 | stress | 8 | 83.4 | 24.5 | 289 | 202 | 1.38 |
| nva_5_26_math_9_12 | rest | 11 | 79.2 | 20.8 | 372 | 104 | 3.51 |
| nva_5_26_math_9_12 | stress | 8 | 82.8 | 29.9 | 172 | 126 | 1.14 |
| nvt_5_26_math_7_10 | rest | 11 | 72.1 | 29.8 | 158 | 260 | 0.64 |
| nvt_5_26_math_7_10 | stress | 8 | 79.0 | 22.7 | 149 | 149 | 0.84 |
| nvt_5_26_math_7_11 | rest | 11 | 70.6 | 30.0 | 75 | 238 | 0.32 |
| nvt_5_26_math_7_11 | stress | 8 | 79.3 | 24.2 | 144 | 172 | 0.89 |
| tnq_5_29_math_7_12 | stress | 8 | 106.3 | 14.1 | 85 | 41 | 2.08 |

