# Data Processing Report (historical baseline — `pantompkins1985`)

> **Production has moved to `method="neurokit"` for R-peak detection.** This document describes the legacy `pantompkins1985` chain (whole-recording detection) which is now the **baseline-only** comparison reference. The production-aligned doc is [data-processing-using-neurokit.md](data-processing-using-neurokit.md); the per-phase isolation variant explored along the way is [data-processing-using-phase.md](data-processing-using-phase.md). All `report.md` numbers come from the **neurokit** variant, not this one.

End-to-end signal preprocessing applied to every recording before windowing /
feature extraction / classification.

All code is in [`src/preprocess.py`](src/preprocess.py),
[`src/features.py`](src/features.py), and [`src/plots.py`](src/plots.py).

---

## 1. Input data

9 recordings, ~12–16 min each, 4 channels per recording, tab-separated CSV/TXT.
Column header names vary across files (some have trailing spaces, a `rime_br`
typo, no `_s` suffix) so the loader reads by **column index**:

| index | column | typical sampling rate |
|---:|---|---|
| 0, 1 | mic time, mic value | ~2000 Hz |
| 2, 3 | BR time, BR value | ~500 Hz |
| 4, 5 | ECG time, ECG value | ~500 Hz |
| 6, 7 | temp time, temp value (°C) | ~1 Hz, irregular |

Phase boundaries within each recording are derived from the filename:
`rest = [0, 300)`, `stress = [300, 300 + stress_dur)`,
`recovery = [300 + stress_dur, end]`, where `stress_dur` is `300 s` for
`medi`/`math` and the parsed `mm'ss` plank duration for `pla`.

---

## 2. Per-channel preprocessing pipelines

### 2.1 ECG (target 500 Hz, R-peak detection)

> Filter chain updated 2026-05-23 from `LP 150 Hz + median detrend` to `HP 1 Hz → notch 60 Hz (Q=30) → LP 150 Hz`. The displayed signal and time-domain features now see explicit mains rejection and baseline-wander removal, in addition to Pan–Tompkins's own 5–15 Hz detection-side prefilter.

| Step | Code | Effect |
|---|---|---|
| 1. Resample | [`features.py:preprocess_recording`](src/features.py) | linear interpolate ECG onto a uniform 500 Hz grid |
| 2. Filter | [`preprocess.py:filter_ecg`](src/preprocess.py) | (a) 4th-order Butterworth **high-pass at 1 Hz** (baseline-wander removal) → (b) **IIR notch at 60 Hz**, Q=30 (mains rejection; default 60 Hz for KR mains, `mains=50.0` passed by [`wesad_io.py`](src/wesad_io.py) for the German recordings) → (c) 4th-order Butterworth **low-pass at 150 Hz**. All sections SOS form, `sosfiltfilt` (zero-phase). The 1 Hz HP supersedes the previous median detrend. |
| 3. Detect | [`preprocess.py:detect_ecg_rpeaks`](src/preprocess.py) | flip sign (R-peaks deflect *negative* in this device) → `neurokit2.ecg_clean(method="neurokit")` → `neurokit2.ecg_peaks(method="pantompkins1985")` → `neurokit2.signal_fixpeaks(method="kubios", iterative=False)` → ±60 ms apex snap. **kubios `iterative=False`** since 2026-05-23: the iterative mode recomputes outlier thresholds over the whole recording, which is theoretically unsound when the signal has phase-dependent variance (rest vs plank). The change made essentially zero difference to detected R-peak counts (see §3 below) — kept as a principled cleanup. |
| 4. Clean NN | [`preprocess.py:clean_nn_intervals`](src/preprocess.py) | reject NN < 300 ms, NN > 1500 ms, or NN deviating > 20 % from the median of its ~10 nearest neighbours; cubic-spline interpolate the rejected ones |

### 2.2 Respiration (target 100 Hz, breath-peak detection)

| Step | Code | Effect |
|---|---|---|
| 1. Resample | [`features.py:preprocess_recording`](src/features.py) | linear interpolate BR onto a uniform 100 Hz grid |
| 2. Filter | [`preprocess.py:filter_br`](src/preprocess.py) | detrend (subtract mean) → 0.5 s moving average → 4th-order Cheby II low-pass (stopband edge 1 Hz, 40 dB attenuation, SOS form) → 2nd-order Butterworth high-pass at 0.12 Hz |
| 3. **Outlier clip** | [`preprocess.py:detect_br_peaks`](src/preprocess.py) (`clip_mad=5.0`) | clamp signal to ±5·MAD (median absolute deviation × 1.4826). **Critical fix** — see §4 |
| 4. Detect | same | `scipy.signal.find_peaks` with ≥ 1.5 s spacing → per-candidate AC = `br[p] − (V1 + V2)/2` where V1/V2 are the local minima either side → adaptive threshold `mean(last 8 accepted) / 3` |

### 2.3 Microphone / PCG (target 2 kHz, S1/S2 detection)

| Step | Code | Effect |
|---|---|---|
| 1. Resample | [`features.py:preprocess_recording`](src/features.py) | uniform 2 kHz grid |
| 2. Filter | [`preprocess.py:filter_mic_pcg`](src/preprocess.py) | 4th-order Butterworth bandpass 20–200 Hz (SOS) — heart sound band |
| 3. Envelope | [`preprocess.py:shannon_energy`](src/preprocess.py) | Shannon energy `SE = −x² log(x²)`, smoothed with a 50 ms moving average |
| 4. Detect | [`preprocess.py:detect_pcg_peaks`](src/preprocess.py) | `find_peaks` with 80 ms min distance, prominence 0.2·std of the envelope |
| 5. Classify | feature extractor | each ECG R-peak picks one S1 (Shannon peak in R+0–200 ms) and one S2 (R+200–500 ms) for CSI |

---

## 3. Per-recording detection summary

Numbers below come from running the *exact* current pipeline (HP 1 → notch 60 → LP 150 on ECG, kubios `iterative=False`, full chain unchanged on BR/PCG) on the dataset. Reasonable HR range for resting adults: 60–100 bpm. Reasonable breath rate: 8–16 / min.

### 3.1 Whole-recording averages

| Recording | duration (s) | #R-peaks | HR (bpm) | #BR-peaks | BR (/min) |
|---|---:|---:|---:|---:|---:|
| `ljh_5_21_medi_posiECG` | 917 | 1085 | 71.0 | 209 | 13.7 |
| `mta-5-17-medi` | 931 | 1104 | 71.2 | 163 | 10.5 |
| `mta-5-17-medi (1)` | 914 | 1110 | 72.9 | 165 | 10.8 |
| `mta-5-17-pla-1'26` | 721 | 402 | 33.5 | 125 | 10.4 |
| `mta-5-17-pla-2` | 754 | 401 | 31.9 | 139 | 11.1 |
| `mta2_5_19_medi` | 918 | 1075 | 70.3 | 198 | 12.9 |
| `mta_5_19_medi` | 906 | 1106 | 73.3 | 200 | 13.2 |
| `mta_5_19_medi (1)` | 906 | 1106 | 73.3 | 200 | 13.2 |
| `mta_5_19_pla_1'40` | 727 | 471 | 38.9 | 115 | 9.5 |
| `mta_5_19_pla_2'20` | 768 | 377 | 29.4 | 143 | 11.2 |
| `mta_5_21_medi` | 768 | 823 | 64.3 | 113 | 8.8 |
| `mta_5_21_pla_2` | 595 | 606 | 61.1 | 184 | 18.5 |
| `mta_5_21_pla_2'30(1)` | 650 | 553 | 51.0 | 132 | 12.2 |
| `nvt_5_21_medi` | 863 | 929 | 64.6 | 159 | 11.0 |
| `nvt_5_21_pla_2(1)` | 691 | 343 | 29.8 | 181 | 15.7 |
| `oyj_5_22_medi_posiECG` | 750 | 893 | 71.4 | 147 | 11.8 |
| `oyj_5_22_pla_1'50_posiECG` | 590 | 786 | 79.9 | 94 | 9.6 |
| `oyj_5_22_pla_2'15_posiECG` | 611 | 596 | 58.5 | 77 | 7.6 |

### 3.2 Per-phase R-peak rate — `pla` recordings (where the problem lives)

Computed by slicing the detected R-peak series at `rest = [0, 300)`, `stress = [300, 300 + stress_dur)`, `recovery = [300 + stress_dur, end]`. Per-phase HR is `60 × n_peaks / phase_duration`.

| Recording | rest bpm | stress bpm | **recovery bpm** | notes |
|---|---:|---:|---:|---|
| `mta-5-17-pla-1'26` | 76.0 | 13.3 | **0.5** | recovery collapsed |
| `mta-5-17-pla-2` | 75.2 | 12.0 | **0.2** | recovery collapsed |
| `mta_5_19_pla_1'40` | 71.6 | 59.4 | **2.6** | recovery collapsed, plank OK |
| `mta_5_19_pla_2'20` | 74.0 | 2.1 | **0.4** | plank + recovery both collapsed (electrode failure) |
| `mta_5_21_pla_2'30(1)` | 64.2 | 86.0 | **5.1** | recovery collapsed, plank tachycardia fine |
| `mta_5_21_pla_2` | 67.0 | 20.0 | **79.0** | **clean recovery** (control case) |
| `nvt_5_21_pla_2(1)` | 63.4 | 12.0 | **0.4** | recovery collapsed |
| `oyj_5_22_pla_1'50_posiECG` | 73.2 | 92.7 | **83.1** | **clean recovery** (control case) |
| `oyj_5_22_pla_2'15_posiECG` | 73.8 | 88.4 | **9.5** | recovery degraded |

**Observations:**

- **Medi recordings are clean end-to-end** (64–75 bpm averages). R-peak detection works through rest, breathing exercise, and recovery without issue.
- **The dominant failure mode is post-plank recovery, not the plank itself.** 6 of 9 `pla` recordings drop to ≤ 9.5 bpm in recovery despite the recovery signal often looking visually normal (see [outputs/ecg_full_mta_5_19_pla_1_40.png](outputs/ecg_full_mta_5_19_pla_1_40.png) — clean signal, sparse R-peak markers).
- **Disabling kubios `iterative=True` (2026-05-23) did NOT fix this.** Per-phase counts before vs. after the change shifted by ≤ 2 R-peaks per phase in every recording — well within noise. The hypothesis that kubios was rejecting legitimate post-plank R-peaks as global outliers is **falsified**. The missing R-peaks were never proposed by Pan-Tompkins in the first place. The root cause is upstream — either `neurokit2.ecg_clean`'s adaptive baseline shifts during plank and never re-adapts, or Pan-Tompkins's own adaptive threshold drifts during the noisy plank phase and stays mis-calibrated through recovery.
- **Two `pla` recordings recover cleanly** (`mta_5_21_pla_2`: 79.0 bpm in recovery; `oyj_5_22_pla_1'50_posiECG`: 83.1 bpm). In both, the plank phase itself was less noisy (signal not saturated) — corroborating that the failure is *triggered* by plank-noise contamination of the detector's adaptive state.
- **`mta_5_19_pla_2'20`** is a hardware case — the signal stays saturated through recovery (electrode displacement that doesn't reseat). No detector change can recover what isn't in the signal.
- **Breath rates fall in 7.6–18.5 /min**, broadly physiological. Outliers: `mta_5_21_pla_2` at 18.5 /min (plausible plank hyperventilation) and `oyj_5_22_pla_2'15_posiECG` at 7.6 /min (BR dropout — visible in §5 plot). The BR pipeline was not changed in this revision.
- **The four `posiECG` recordings** report HR of 58.5–79.9 bpm whole-recording. The negative-polarity Pan-Tompkins still picks up R-peaks on positive-polarity hardware; the §5 plots will show inverted apices, which is the documented diagnostic view.

### 3.3 Next step (not yet implemented)

Since `iterative=False` did not help, the next investigation should target the per-phase failure of `ecg_clean` and Pan-Tompkins's adaptive threshold. Two reasonable candidate fixes, in order of expected payoff:

1. **Phase-aware detection** — split the signal at `phase_boundaries(rec)` and run `detect_ecg_rpeaks` independently on each phase. Both `ecg_clean` and Pan-Tompkins re-initialise per phase, so plank-noise can't contaminate recovery. ~30-line wrapper.
2. **Replace `nk.ecg_peaks(method="pantompkins1985")`** with `method="neurokit"` (the library's own default, generally more robust on noisy short-window data) and compare per-phase counts on the affected recordings.

---

## 4. The BR-detector fix (and why it mattered)

### Symptom (before fix)

Running [`detect_br_peaks`](src/preprocess.py) on the raw filtered BR signal
produced wildly inconsistent counts:

| Recording | peaks (before) | rate /min (before) | Diagnosis |
|---|---:|---:|---|
| `mta-5-17-medi` | 112 | 7.2 | works |
| `mta-5-17-pla-2` | 19 | **1.5** | giant 1.7e6 transient at the rest→stress boundary |
| `mta_5_19_medi` | 1 | **0.1** | giant 1.4e5 transient at t = 0 (electrode settling) |
| `mta_5_19_pla_1'40` | similar | broken | similar boundary spike |

### Cause

The original detector implements `features.pdf`'s adaptive threshold rule:
each candidate breath peak is accepted only if its AC amplitude ≥
`mean(last 8 accepted) / 3`. The first peak is always accepted because the
"last 8" list is empty. On recordings with a large motion-artefact transient
early in the signal, *that transient becomes the first accepted peak* with
amplitude 10⁵–10⁶. Then `mean(...) / 3 ≈ 10⁵`, and every real breath
(amplitude ~10³) falls below the threshold and is silently rejected — sometimes
for the entire rest of the recording.

### Fix

Pre-clip the filtered signal to **±5·MAD** (median absolute deviation × 1.4826)
before `find_peaks`. MAD is robust to outliers, so even a single 10⁶ transient
in an otherwise quiet signal sets the clip limit at a reasonable ±50000
instead of being scaled by the artefact itself. Once the artefact is bounded,
it can no longer dominate the adaptive threshold.

```python
# src/preprocess.py:detect_br_peaks
if clip_mad > 0:
    med = float(np.median(br))
    mad = float(np.median(np.abs(br - med)))
    sigma_robust = 1.4826 * mad
    if sigma_robust > 1e-12:
        limit = clip_mad * sigma_robust
        br = np.clip(br, med - limit, med + limit)
```

### Result (after fix)

All 9 recordings produce physiologically plausible breath rates (9.5–13.2 /min).
The corresponding ML scores improved across the board — see [`report.md`](report.md).

---

## 5. Per-recording peak plots

Each pair shows the full-duration filtered signal with the detected peaks
overlaid, exactly as produced by the current pipeline. Phases are shaded
(rest blue, stress orange, recovery green). For BR, the y-axis is auto-zoomed
to the 2–98 percentile so that motion-artefact transients go off-chart
instead of crushing the breath waveform onto zero. ECG is auto-zoomed to
1–99 percentile and shown **flipped** (so the R-wave apex points up — see
§2.1).

> All plots are regenerated by `uv run python -m src.plots`.

### 5.1 `mta-5-17-medi (1)` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta-5-17-medi_(1).png) | ![ECG](outputs/ecg_full_mta-5-17-medi_(1).png) |

### 5.2 `mta-5-17-medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta-5-17-medi.png) | ![ECG](outputs/ecg_full_mta-5-17-medi.png) |

### 5.3 `mta-5-17-pla-1'26` — plank 1 m 26 s

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta-5-17-pla-1_26.png) | ![ECG](outputs/ecg_full_mta-5-17-pla-1_26.png) |

### 5.4 `mta-5-17-pla-2` — plank 2 m

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta-5-17-pla-2.png) | ![ECG](outputs/ecg_full_mta-5-17-pla-2.png) |

### 5.5 `mta2_5_19_medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta2_5_19_medi.png) | ![ECG](outputs/ecg_full_mta2_5_19_medi.png) |

### 5.6 `mta_5_19_medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta_5_19_medi.png) | ![ECG](outputs/ecg_full_mta_5_19_medi.png) |

### 5.7 `mta_5_19_medi (1)` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta_5_19_medi_(1).png) | ![ECG](outputs/ecg_full_mta_5_19_medi_(1).png) |

### 5.8 `mta_5_19_pla_1'40` — plank 1 m 40 s

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta_5_19_pla_1_40.png) | ![ECG](outputs/ecg_full_mta_5_19_pla_1_40.png) |

### 5.9 `mta_5_19_pla_2'20` — plank 2 m 20 s

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta_5_19_pla_2_20.png) | ![ECG](outputs/ecg_full_mta_5_19_pla_2_20.png) |

### 5.10 `mta_5_21_medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta_5_21_medi.png) | ![ECG](outputs/ecg_full_mta_5_21_medi.png) |

### 5.11 `mta_5_21_pla_2` — plank 2 m

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta_5_21_pla_2.png) | ![ECG](outputs/ecg_full_mta_5_21_pla_2.png) |

### 5.12 `mta_5_21_pla_2'30(1)` — plank 2 m 30 s

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_mta_5_21_pla_2_30%281%29.png) | ![ECG](outputs/ecg_full_mta_5_21_pla_2_30%281%29.png) |

### 5.13 `nvt_5_21_medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_nvt_5_21_medi.png) | ![ECG](outputs/ecg_full_nvt_5_21_medi.png) |

### 5.14 `nvt_5_21_pla_2(1)` — plank 2 m

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_nvt_5_21_pla_2%281%29.png) | ![ECG](outputs/ecg_full_nvt_5_21_pla_2%281%29.png) |

### 5.15 `ljh_5_21_medi_posiECG` — medi (positive-polarity ECG)

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_ljh_5_21_medi_posiECG.png) | ![ECG](outputs/ecg_full_ljh_5_21_medi_posiECG.png) |

### 5.16 `oyj_5_22_medi_posiECG` — medi (positive-polarity ECG)

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_oyj_5_22_medi_posiECG.png) | ![ECG](outputs/ecg_full_oyj_5_22_medi_posiECG.png) |

### 5.17 `oyj_5_22_pla_1'50_posiECG` — plank 1 m 50 s (positive-polarity ECG)

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_oyj_5_22_pla_1_50_posiECG.png) | ![ECG](outputs/ecg_full_oyj_5_22_pla_1_50_posiECG.png) |

### 5.18 `oyj_5_22_pla_2'15_posiECG` — plank 2 m 15 s (positive-polarity ECG)

| BR | ECG |
|---|---|
| ![BR](outputs/br_full_oyj_5_22_pla_2_15_posiECG.png) | ![ECG](outputs/ecg_full_oyj_5_22_pla_2_15_posiECG.png) |

---

## 6. Known limitations

1. **ECG dropout during plank.** Chest motion saturates or briefly disconnects the ECG electrode in every plank recording. Pan–Tompkins produces zero or near-zero peaks for those segments. Windows whose ECG-derived features (hr, hrv_*, csi) are mostly NaN are not dropped — the median imputer fills them and the model learns to deal with it. Collecting plank ECG with a tighter electrode mount would be the most direct improvement to plank-class recall.
2. **BR motion-artefact transients are bounded, not removed.** The MAD clip prevents them from poisoning peak detection, but the underlying motion still corrupts the signal during plank stress. The plots in §5 make this visible — within plank stress phases the BR signal is often dominated by ringing rather than clean breath cycles.
3. **Temperature is sampled at ~1 Hz with frequent gaps** (intervals of 1–10 s). Not currently used as a feature in the production model per the `features.pdf` spec; the loader keeps it for ablations.
4. **Boundary-window skip cost.** Excluding windows that straddle the 300 s / 600 s protocol transitions costs ~12 windows per recording. The remaining 112 windows are what every model is trained and evaluated on.

---

## 7. Reproduce

```bash
# Regenerate features.csv + model artefacts after any preprocessing change:
uv run python -m src.local_eval
uv run python -m src.dl_train

# Regenerate all signal-quality plots in this report:
uv run python -m src.plots
# -> outputs/br_full_*.png, outputs/ecg_full_*.png, outputs/signals_*.png
```
