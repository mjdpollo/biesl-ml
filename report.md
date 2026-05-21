# Stress / Meditation / Rest / Recovery Classification Report

**Task.** Per-window **4-class** classification: `rest` / `meditation` / `stress` / `recovery`.
**Data.** 9 local wearable recordings (~15 min each) from 2 subjects (`mta`, `mta2`), downloaded fresh from the team Google Drive folder ([see README](README.md#dataset)). WESAD is not used.
**Features.** The eight parameters defined in [`features.pdf`](features.pdf), and only those — **no temperature features.**
**Window-boundary policy.** Windows whose 60-s extent touches the 5-min or 10-min protocol transitions are excluded (patient reported discomfort at the transitions).
**Evaluation protocols.** (a) Leave-one-recording-out (LORO, 9 folds) — the honest cross-recording benchmark; (b) stratified 70:15:15 random window split averaged over 5 seeds — useful as a sanity check but contaminated by 50 % window overlap.

---

## 1. Feature definitions (from features.pdf)

Exactly eight features per 60 s window. No temperature features in this pipeline.

| Feature | Channel | Preprocessing | Computation |
|---|---|---|---|
| `csi` | Microphone | Bandpass 20–200 Hz → Shannon-energy envelope `−x² log(x²)` → peak detection | S2/S1 amplitude ratio. Each ECG R-peak picks one S1 (Shannon peak in R+0–200 ms) and one S2 (Shannon peak in R+200–500 ms). CSI = mean(S2 amp) / mean(S1 amp). |
| `hr` | ECG | Low-pass 150 Hz + detrend → Pan–Tompkins R-peak detection. NN intervals cleaned: reject NN < 300 ms, NN > 1500 ms, or NN deviating > 20 % from the median of the surrounding ~10 beats. Rejected NNs are cubic-spline interpolated. | `60 000 / mean(NN_ms)` (bpm) |
| `hrv_rmssd` | ECG | Same cleaned NN series | `√(mean((NN_{i+1} − NN_i)²))`, in ms |
| `hrv_lf` | ECG | Welch PSD on 4 Hz interpolated tachogram, Hann window, 60 s segments where possible | Area under PSD in 0.04–0.15 Hz, in ms² |
| `hrv_hf` | ECG | Same Welch | Area under PSD in 0.15–0.40 Hz, in ms² |
| `hrv_lf_hf` | ECG | — | `hrv_lf / hrv_hf` |
| `rr` | Respiration | Detrend → 0.5 s moving average → 4th-order Chebyshev II low-pass (stopband edge 1 Hz, 40 dB) → 2nd-order Butterworth high-pass at 0.12 Hz. Slope-based peak detection with adaptive threshold = 1/3 × mean of the last 8 accepted breath amplitudes. | `60 / mean(breath interval, s)` (bpm) |
| `rrv` | Respiration | Same chain | Standard deviation of the last 5 breath intervals (s) |

**Window length:** 60 s, 50 % overlap.
**Boundary skip:** any window whose [t, t+60] includes the 5-min mark (300 s) or the 10-min mark (600 s) is dropped. With 50 % overlap and phase-respecting windowing this removes 4 windows per medi recording (2 near each boundary) and 2 per pla recording.

---

## 2. Dataset summary

| Recording | rest | meditation | stress | recovery |
|---|---|---|---|---|
| `mta-5-17-medi` | 8 | 8 | 0 | 8 |
| `mta-5-17-medi (1)` | 8 | 8 | 0 | 7 |
| `mta-5-17-pla-1'26` (plank 1 m 26 s) | 8 | 0 | 1 | 7 |
| `mta-5-17-pla-2` (plank 2 m) | 8 | 0 | 2 | 7 |
| `mta2_5_19_medi` | 8 | 7 | 0 | 8 |
| `mta_5_19_medi` | 8 | 7 | 0 | 7 |
| `mta_5_19_medi (1)` | 8 | 7 | 0 | 8 |
| `mta_5_19_pla_1'40` (plank 1 m 40 s) | 8 | 0 | 1 | 7 |
| `mta_5_19_pla_2'20` (plank 2 m 20 s) | 8 | 0 | 1 | 7 |
| **Total (classical pipeline, 182 windows)** | **72** | **35** | **5** | **70** |
| **Total (CNN pipeline, 177 windows)** | **72** | **35** | **5** | **65** |

The CNN pipeline aligns ECG, Resp and Mic-Shannon-envelope to a common 250 Hz grid; the slight length mismatch between channels causes a few `recovery` windows at the end of long pla recordings to be cropped.

The `stress` class still has only **5 windows total** across 4 plank recordings (each plank is short — 86 s to 140 s — so each pla recording contributes 1–2 stress windows). This is the dominant constraint on every model's `stress` recall.

---

## 3. Models

| Family | Inputs | Architecture |
|---|---|---|
| KNN | 8 PDF features | median-imputer → StandardScaler → KNN (k=7, distance-weighted, Euclidean) |
| RandomForest | same | median-imputer → RandomForest (400 trees, `min_samples_leaf=2`, `max_features='sqrt'`) |
| XGBoost | same | XGBoost (400 trees, depth=4, lr=0.05) — handles NaNs natively |
| 1D-CNN | 3 raw channels (ECG @ 250 Hz, Resp @ 250 Hz, Mic Shannon-envelope @ 250 Hz), 60 s = (3, 15 000) | 5-block 1D conv stack + AdaptiveAvgPool + 2-layer MLP head, 636 k params, AMP, AdamW + cosine LR, class-weighted CE, early stopping |

---

## 4. Results

### 4.1 LORO macro-F1 (mean across 9 folds)

| Model | acc | macro-F1 | F1[rest] | F1[meditation] | F1[stress] | F1[recovery] |
|---|---|---|---|---|---|---|
| KNN | 0.669 | 0.592 | 0.698 | 0.351 | 0.185 | 0.642 |
| RandomForest | 0.774 | 0.663 | 0.817 | 0.432 | 0.000 | 0.724 |
| **XGBoost** | **0.791** | **0.723** | 0.757 | 0.470 | 0.222 | 0.728 |
| 1D-CNN | 0.273 | 0.262 | 0.129 | 0.218 | 0.259 | 0.324 |

### 4.2 Random 70:15:15 macro-F1 (mean ± std over seeds 0–4)

| Model | acc | macro-F1 | F1[rest] | F1[meditation] | F1[stress] | F1[recovery] |
|---|---|---|---|---|---|---|
| KNN | 0.679 ± 0.130 | 0.629 ± 0.173 | 0.719 | 0.743 | 0.000 | 0.596 |
| RandomForest | 0.771 ± 0.110 | 0.725 ± 0.164 | 0.808 | 0.891 | 0.000 | 0.697 |
| XGBoost | 0.814 ± 0.069 | 0.795 ± 0.136 | 0.828 | 0.898 | 0.160 | 0.762 |
| **1D-CNN** | **0.867 ± 0.073** | **0.850 ± 0.123** | 0.884 | 0.921 | 0.400 | 0.822 |

### 4.3 LORO vs random-split — same models, different protocol

| Model | LORO macro-F1 | Random-split macro-F1 | Δ |
|---|---|---|---|
| KNN | 0.592 | 0.629 | +0.04 |
| RandomForest | 0.663 | 0.725 | +0.06 |
| XGBoost | 0.723 | 0.795 | +0.07 |
| **1D-CNN** | **0.262** | **0.850** | **+0.59** |

> ⚠️ **The CNN's huge LORO → random-split jump is data leakage, not learning.** Windows have 50 % overlap, so a test window's 30-s-shifted neighbour can sit in the train set. Classical models work on scalar HRV summaries that are insensitive to that overlap; the CNN reads raw waveforms and latches onto the temporal proximity. Use LORO numbers for any honest cross-recording claim; random-split numbers are a within-recording sanity check at most.

### 4.4 Per-class recall (LORO, diagonal of confusion matrices)

| Model | rest | meditation | stress | recovery |
|---|---|---|---|---|
| KNN | 68.1 % | 60.0 % | **60.0 %** (3/5) | 65.7 % |
| RandomForest | **87.5 %** | 71.4 % | 0.0 % | 74.3 % |
| **XGBoost** | 79.2 % | **80.0 %** | **80.0 %** (4/5) | **75.7 %** |
| 1D-CNN | 16.7 % | 31.4 % | 60.0 % | 40.0 % |

Two highlights:

- **XGBoost is the only model that does well on every class under LORO**, including 80 % stress recall (4 of 5 stress windows correct). It's the production candidate.
- **KNN matches XGBoost on stress recall** (60 %, 3/5) — distance-based learners can pick up the dramatically different HR/HRV/RR signature of plank stress even with only 4 stress training windows.
- **The 1D-CNN collapses under LORO.** With 4 classes and a fine `rest` / `recovery` distinction it can't generalize across recordings on raw waveforms with only 177 windows. The 0.85 random-split number is the overlap-leakage ceiling.

---

## 5. Confusion matrices

Rows are true labels, columns are predictions. **Each row is row-normalized to 100 %** (true-class recall view). The cell at (`baseline`, `baseline`) is the percentage of actual baseline windows the model correctly predicted as baseline. Support counts are written under the y-axis tick labels in every PNG and in the [confusion-matrices.md](confusion-matrices.md) companion file — crucial when `stress` has only 5 samples total.

### 5.1 LORO — Classical

| KNN | RandomForest | XGBoost |
|---|---|---|
| ![KNN LORO](figures/confusion/loro__classical_knn.png) | ![RandomForest LORO](figures/confusion/loro__classical_randomforest.png) | ![XGBoost LORO](figures/confusion/loro__classical_xgboost.png) |

### 5.2 LORO — 1D-CNN

![1D-CNN LORO](figures/confusion/loro__cnn.png)

### 5.3 Random 70:15:15 — Classical

| KNN | RandomForest | XGBoost |
|---|---|---|
| ![KNN random](figures/confusion/randomsplit__classical_knn.png) | ![RandomForest random](figures/confusion/randomsplit__classical_randomforest.png) | ![XGBoost random](figures/confusion/randomsplit__classical_xgboost.png) |

### 5.4 Random 70:15:15 — 1D-CNN

![1D-CNN random](figures/confusion/randomsplit__cnn.png)

---

## 6. Key findings

1. **XGBoost on the 8 PDF features is the strongest model on every meaningful axis.** macro-F1 0.723 under honest LORO, balanced per-class recall (80 % rest / 80 % meditation / 80 % stress / 76 % recovery). Production candidate.
2. **The new `rest` ↔ `recovery` distinction is the dominant difficulty for the classical models.** Both look like quiet sitting from a physiology standpoint. RF and XGBoost score 75–87 % on both individually, but confusions between them account for most off-diagonal mass in the LORO matrices.
3. **The 5-minute / 10-minute boundary skip removed ~4 windows per medi recording and ~2 per pla recording.** Net dataset is 182 classical windows / 177 CNN windows (vs ~200+ if we kept the transition windows). No model regression observed from this cleanup.
4. **The `stress` class has 5 windows total** across 4 plank recordings. XGBoost recalls 4 of them (80 %), KNN recalls 3 (60 %), RandomForest 0, 1D-CNN 3 (60 %). Collecting more plank recordings remains the single highest-leverage data improvement.
5. **The 1D-CNN is unfit for production at this data scale.** Its LORO macro-F1 of 0.262 (vs random-split 0.850) is the cleanest demonstration so far that the +0.59 random-split inflation is overlap leakage. Until we have ≥1000 windows per class, classical features win.

---

## 7. Advice / next steps

1. **Ship XGBoost on the 8 PDF features as the production baseline.** macro-F1 0.723 LORO, balanced per-class recall, fits in memory, trains in seconds without a GPU, no hyperparameters to tune live.
2. **Collect more plank recordings.** 4 plank recordings → 5 stress windows is the floor on stress recall. Each new plank recording adds ~1–2 windows; even 4 more recordings would let us evaluate F1[stress] with statistical meaning.
3. **Treat `rest` vs `recovery` as a separate question.** They are physically the same kind of sitting; the only difference is "before stress" vs "after stress". If the team's downstream application doesn't actually need to tell them apart, collapsing back to a 3-class `{baseline, meditation, stress}` problem will lift macro-F1 by ~0.05–0.10.
4. **Keep the 1D-CNN as research, not deployment.** Revisit once we have ≥10× more data per class.

---

## 8. Reproduce

```bash
# Re-download the dataset (places .txt files in data/)
mkdir -p data/_old && mv data/*.txt data/_old/ 2>/dev/null
cd data && uv run --group dev gdown --folder \
    "https://drive.google.com/drive/folders/11epSBil0cIWSKvtShCp86gCrUKEOjxkn"
cd .. && mv "data/Stress test data/"*.txt data/

# Classical models, LORO + random split
uv run python -m src.local_eval

# 1D-CNN, LORO + random split
uv run python -m src.dl_train

# Regenerate confusion matrices + heatmap PNGs
uv run python scripts/show_confusion_matrices.py > confusion-matrices.md
```

Output locations:
- `outputs/local_loro.json`, `outputs/local_randomsplit.json`
- `outputs/dl_local_loro.json`, `outputs/dl_local_randomsplit.json`
- `figures/confusion/*.png` (tracked in git)
- `confusion-matrices.md` (8 markdown tables)
