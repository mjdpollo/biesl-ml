# Median-filter-adapted breathing (BR) analysis

This report documents the switch from the previous BR filter chain
(moving-average → Chebyshev II → Butterworth) to a **median-filter** approach
with an **adaptable window**, and shows the detected breath peaks for every
recording in the refreshed dataset (31 recordings, 7 subjects, stressors:
medi / plank / math).

## 1. Method

Implemented in [`src/preprocess.py`](src/preprocess.py) `filter_br()` and
`detect_br_peaks()`.

### Filtering — two median passes (adaptable windows)

| Step | Operation | Default window | Why median, not linear |
|---|---|---|---|
| 1. Baseline removal | `baseline = median_filter(x, baseline_window_s)`, then `x − baseline` | **8 s** | A long median tracks slow respiratory-baseline wander and the large DC drift in the raw belt signal. Subtracting it removes drift **without ringing** — a linear high-pass rings around the step transients that body motion produces at the phase boundaries; a median filter does not. |
| 2. Despike / smooth | `median_filter(detrended, smooth_window_s)` | **0.5 s** | A short median removes residual motion spikes while preserving the breath waveform shape (median filtering is edge-preserving, unlike a moving average that rounds the peaks). |

Both window lengths are **parameters** of `filter_br()` — widen
`baseline_window_s` for very slow breathing, widen `smooth_window_s` for
noisier mic-coupled belts. The working sample rate is 100 Hz.

### Peak detection

`scipy.signal.find_peaks` with:

* **minimum inter-peak distance 1.5 s** (caps the rate at 40 breaths/min and acts as the slope surrogate from features.pdf);
* an **adaptive global prominence floor** = `prom_frac × p90(|signal|)` where the 90th-percentile amplitude reflects the *active* breathing depth (robust to the flat rest/recovery stretches that dominate the median). Default `prom_frac = 0.25`. This is the single most important knob for suppressing noise peaks in low-amplitude regions.
* the features.pdf **adaptive amplitude threshold** (1/3 × mean of the last 8 accepted breath amplitudes) on top, for breath-to-breath depth variation.

## 2. Key finding — BR signal quality is phase-dependent

The breathing channel captures a **strong, clean signal during deep-breathing phases (meditation, plank, math stress)** but is **near noise-level during quiet rest/recovery**. On a representative meditation recording the filtered-signal robust amplitude (1.4826·MAD) is:

| Phase | Robust amplitude | Interpretation |
|---|---|---|
| rest (0–5 min) | ~18 | essentially noise — no clear breathing rhythm |
| meditation (5–10 min) | ~1265 | clean deep breaths, ~70× larger |
| recovery (>10 min) | ~20 | back to noise level |

**Consequence:** breath peaks are reliable during the stressor phase, but the rest/recovery peaks are weak detections on a low-SNR signal. The per-recording "mean RR" below is therefore dominated by, and most trustworthy for, the active phase. Quiet-breathing rate should be treated as approximate until the belt coupling during rest is improved.

This is a sensor/coupling finding, not a detector bug: the zoomed rest signal shows choppy ±50-unit fluctuation with no respiratory periodicity, while the meditation signal shows textbook ±5000-unit breaths.

## 3. Per-recording breath counts

`breaths` = accepted peaks over the whole recording; `RR` = 60 / mean inter-breath interval (1–12 s intervals only). `ecg` notes the R-peak polarity flagged by the filename (`posiECG` → positive).

### Meditation (10)

| Recording | breaths | mean RR (bpm) | ecg | duration (s) |
|---|---|---|---|---|
| ljh_5_21_medi_posiECG | 199 | 16.9 | positive | 909 |
| mta-5-17-medi | 175 | 13.2 | negative | 923 |
| mta-5-17-medi (1) | 235 | 15.7 | negative | 906 |
| mta2_5_19_medi | 178 | 13.2 | negative | 909 |
| mta_5_19_medi | 233 | 17.1 | negative | 898 |
| mta_5_19_medi (1) | 233 | 17.1 | negative | 898 |
| mta_5_21_medi | 213 | 17.4 | negative | 761 |
| nvt_5_21_medi | 233 | 17.2 | negative | 856 |
| oyj_5_22_medi_posiECG | 145 | 12.3 | positive | 744 |
| smj_5_22_medi | 185 | 14.8 | negative | 771 |

### Plank (15)

| Recording | breaths | mean RR (bpm) | ecg | duration (s) |
|---|---|---|---|---|
| mta-5-17-pla-1'26 | 141 | 20.2 | negative | 715 |
| mta-5-17-pla-2 | 153 | 18.7 | negative | 747 |
| mta_5_19_pla_1'40 | 161 | 17.5 | negative | 720 |
| mta_5_19_pla_2'20 | 183 | 16.1 | negative | 761 |
| mta_5_21_pla_2 | 186 | 19.0 | negative | 590 |
| mta_5_21_pla_2'30(1) | 206 | 19.2 | negative | 644 |
| mta_5_26_pla_3'30 | 220 | 19.1 | negative | 770 |
| ntv_5_25_pla_2 | 129 | 19.3 | negative | 622 |
| ntv_5_25_pla_2'10 | 154 | 17.9 | negative | 611 |
| nvt_5_21_pla_2(1) | 229 | 22.4 | negative | 685 |
| nvt_5_25_pla_3'30 | 143 | 19.8 | negative | 705 |
| oyj_5_22_pla_1'50_posiECG | 112 | 14.1 | positive | 585 |
| oyj_5_22_pla_2'15_posiECG | 126 | 13.9 | positive | 606 |
| smj_5_22_pla_2 | 123 | 14.1 | negative | 633 |
| smj_5_22_pla_2'5 | 136 | 15.3 | negative | 569 |

### Math (6)

| Recording | breaths | mean RR (bpm) | ecg | duration (s) |
|---|---|---|---|---|
| mta_5_26_math_11_13 | 220 | 20.4 | negative | 665 |
| mta_5_26_math_8_12 | 159 | 16.6 | negative | 767 |
| nva_5_26_math_6_8 | 189 | 18.2 | negative | 626 |
| nva_5_26_math_9_12 | 154 | 16.1 | negative | 622 |
| nvt_5_26_math_7_10 | 190 | 18.1 | negative | 630 |
| nvt_5_26_math_7_11 | 218 | 19.0 | negative | 694 |

Plank/math stress phases show higher breath rates (16–22 bpm) than meditation (12–17 bpm) — consistent with elevated respiration under physical and cognitive stress and slowed breathing during meditation.

## 4. Peak plots (all recordings)

Each plot shows the median-filtered BR (blue), detected breath peaks (red ▽), the raw resampled BR for shape reference (grey, z-scored on a hidden twin axis), and the 5-/10-min protocol boundaries (dashed). The y-axis is dominated by the high-amplitude stressor phase, so rest/recovery appear compressed near zero — that compression *is* the low-SNR finding from §2.

### Meditation

![ljh_5_21_medi_posiECG](figures/br/ljh_5_21_medi_posiECG.png)
![mta-5-17-medi](figures/br/mta-5-17-medi.png)
![mta-5-17-medi (1)](figures/br/mta-5-17-medi__1_.png)
![mta2_5_19_medi](figures/br/mta2_5_19_medi.png)
![mta_5_19_medi](figures/br/mta_5_19_medi.png)
![mta_5_19_medi (1)](figures/br/mta_5_19_medi__1_.png)
![mta_5_21_medi](figures/br/mta_5_21_medi.png)
![nvt_5_21_medi](figures/br/nvt_5_21_medi.png)
![oyj_5_22_medi_posiECG](figures/br/oyj_5_22_medi_posiECG.png)
![smj_5_22_medi](figures/br/smj_5_22_medi.png)

### Plank

![mta-5-17-pla-1'26](figures/br/mta-5-17-pla-1_26.png)
![mta-5-17-pla-2](figures/br/mta-5-17-pla-2.png)
![mta_5_19_pla_1'40](figures/br/mta_5_19_pla_1_40.png)
![mta_5_19_pla_2'20](figures/br/mta_5_19_pla_2_20.png)
![mta_5_21_pla_2](figures/br/mta_5_21_pla_2.png)
![mta_5_21_pla_2'30(1)](figures/br/mta_5_21_pla_2_30_1_.png)
![mta_5_26_pla_3'30](figures/br/mta_5_26_pla_3_30.png)
![ntv_5_25_pla_2](figures/br/ntv_5_25_pla_2.png)
![ntv_5_25_pla_2'10](figures/br/ntv_5_25_pla_2_10.png)
![nvt_5_21_pla_2(1)](figures/br/nvt_5_21_pla_2_1_.png)
![nvt_5_25_pla_3'30](figures/br/nvt_5_25_pla_3_30.png)
![oyj_5_22_pla_1'50_posiECG](figures/br/oyj_5_22_pla_1_50_posiECG.png)
![oyj_5_22_pla_2'15_posiECG](figures/br/oyj_5_22_pla_2_15_posiECG.png)
![smj_5_22_pla_2](figures/br/smj_5_22_pla_2.png)
![smj_5_22_pla_2'5](figures/br/smj_5_22_pla_2_5.png)

### Math

![mta_5_26_math_11_13](figures/br/mta_5_26_math_11_13.png)
![mta_5_26_math_8_12](figures/br/mta_5_26_math_8_12.png)
![nva_5_26_math_6_8](figures/br/nva_5_26_math_6_8.png)
![nva_5_26_math_9_12](figures/br/nva_5_26_math_9_12.png)
![nvt_5_26_math_7_10](figures/br/nvt_5_26_math_7_10.png)
![nvt_5_26_math_7_11](figures/br/nvt_5_26_math_7_11.png)

## 5. How to reproduce / retune

```bash
# Regenerate every BR peak plot + the per-recording index
uv run python scripts/plot_br_peaks.py
```

To retune the adaptable windows or the prominence floor, edit the defaults in
`src/preprocess.py` (`BR_BASELINE_WINDOW_S`, `BR_SMOOTH_WINDOW_S`, and
`detect_br_peaks(prom_frac=...)`) or pass them through — e.g. a longer
`smooth_window_s` for noisier belts, a higher `prom_frac` to be stricter about
what counts as a breath in low-SNR rest segments.

## 6. Open items

1. **Rest/recovery BR is low-SNR** for most recordings — the belt/mic couples well only during deep or labored breathing. If quiet-breathing rate matters for the downstream `rest` class, the sensor coupling during rest needs attention.
2. **`prom_frac` is a single global value.** A per-phase or locally-adaptive prominence would detect shallow rest breaths without admitting noise, at the cost of complexity. Deferred until rest-phase BR quality is confirmed.
3. The classification pipeline ([`report.md`](report.md)) still needs to be re-run on this refreshed 31-recording dataset with the `math` class added; that is tracked separately.
