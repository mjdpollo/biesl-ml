# Data Processing Report — Phase-Aware R-peak Detection

End-to-end signal preprocessing variant that runs R-peak detection **independently per protocol phase** (rest / stress / recovery). Companion to [`data-processing.md`](data-processing.md) (baseline pipeline) and [`data-processing-using-neurokit.md`](data-processing-using-neurokit.md) (the algorithm-swap variant).

All code is in [`src/preprocess.py`](src/preprocess.py), [`src/features.py`](src/features.py), and [`src/plots.py`](src/plots.py). This variant is selected by `preprocess_recording(rec, phase_aware=True)`.

---

## 1. What changed vs the baseline

| | Baseline ([data-processing.md](data-processing.md)) | Phase-aware (this doc) |
|---|---|---|
| ECG filter | HP 1 → notch 60 → LP 150 (whole signal) | identical |
| BR / PCG filter | identical | identical |
| R-peak detection | `detect_ecg_rpeaks(ecg, fs)` on whole-recording signal | `detect_ecg_rpeaks_per_phase(ecg, fs, t0, phases)` — slice by `phase_boundaries(rec)`, run `ecg_clean → ecg_peaks(pantompkins1985) → signal_fixpeaks(kubios, iterative=False)` per slice, offset indices back, concatenate |
| Detection method | `pantompkins1985` | `pantompkins1985` (unchanged) |

**Motivation.** Both `ecg_clean`'s adaptive baseline and Pan-Tompkins's adaptive threshold are *stateful* over the whole signal they're applied to. When a plank phase introduces high-amplitude motion artefact, those adaptive states get mis-calibrated and never re-adapt to the cleaner recovery segment. Phase-isolated detection forces a fresh detector state per phase.

**Implementation.** [src/preprocess.py — `detect_ecg_rpeaks_per_phase`](src/preprocess.py):

```python
def detect_ecg_rpeaks_per_phase(ecg, fs, t0, phases, *, method="pantompkins1985"):
    out, seen = [], set()
    for _, (s, e) in phases.items():
        i_lo = max(0, int(round((s - t0) * fs)))
        i_hi = min(len(ecg), int(round((e - t0) * fs)))
        if i_hi - i_lo < int(round(2.0 * fs)):    # < 2 s — skip
            continue
        sub_peaks = detect_ecg_rpeaks(ecg[i_lo:i_hi], fs, method=method)
        for p in sub_peaks:
            idx = int(p + i_lo)
            if 0 <= idx < len(ecg) and idx not in seen:
                seen.add(idx); out.append(idx)
    out.sort()
    return np.asarray(out, dtype=int)
```

---

## 2. Whole-recording averages

| Recording | duration (s) | #R-peaks | HR (bpm) | #BR-peaks | BR (/min) |
|---|---:|---:|---:|---:|---:|
| `ljh_5_21_medi_posiECG` | 917 | 1086 | 71.0 | 209 | 13.7 |
| `mta-5-17-medi` | 931 | 1104 | 71.2 | 163 | 10.5 |
| `mta-5-17-medi (1)` | 914 | 1110 | 72.9 | 165 | 10.8 |
| `mta-5-17-pla-1'26` | 721 | 425 | 35.4 | 125 | 10.4 |
| `mta-5-17-pla-2` | 754 | 418 | 33.3 | 139 | 11.1 |
| `mta2_5_19_medi` | 918 | 1075 | 70.3 | 198 | 12.9 |
| `mta_5_19_medi` | 906 | 1108 | 73.4 | 200 | 13.2 |
| `mta_5_19_medi (1)` | 906 | 1108 | 73.4 | 200 | 13.2 |
| `mta_5_19_pla_1'40` | 727 | 523 | 43.2 | 115 | 9.5 |
| `mta_5_19_pla_2'20` | 768 | 818 | 63.9 | 143 | 11.2 |
| `mta_5_21_medi` | 768 | 824 | 64.4 | 113 | 8.8 |
| `mta_5_21_pla_2` | 595 | 797 | 80.3 | 184 | 18.5 |
| `mta_5_21_pla_2'30(1)` | 650 | 772 | 71.2 | 132 | 12.2 |
| `nvt_5_21_medi` | 863 | 930 | 64.6 | 159 | 11.0 |
| `nvt_5_21_pla_2(1)` | 691 | 362 | 31.4 | 181 | 15.7 |
| `oyj_5_22_medi_posiECG` | 750 | 894 | 71.5 | 147 | 11.8 |
| `oyj_5_22_pla_1'50_posiECG` | 590 | 787 | 80.0 | 94 | 9.6 |
| `oyj_5_22_pla_2'15_posiECG` | 611 | 595 | 58.4 | 77 | 7.6 |

## 3. Per-phase R-peak rate — `pla` recordings (delta vs baseline)

| Recording | rest bpm (Δ) | stress bpm (Δ) | recovery bpm (Δ) | verdict |
|---|---:|---:|---:|---|
| `mta-5-17-pla-1'26` | 76.0 (—) | 10.5 (−2.8) | **5.4 (+4.9)** | small recovery lift, still broken |
| `mta-5-17-pla-2` | 75.2 (—) | 10.5 (−1.5) | **3.8 (+3.6)** | tiny lift |
| `mta_5_19_pla_1'40` | 71.6 (—) | 90.0 (+30.6) | **2.8 (+0.2)** | recovery unchanged, plank tachycardia revealed |
| `mta_5_19_pla_2'20` | 74.0 (—) | 2.1 (—) | **81.0 (+80.6)** | **dramatic recovery** (electrode-fail case still has detectable QRS!) |
| `mta_5_21_pla_2'30(1)` | 64.2 (—) | 85.2 (−0.8) | **71.3 (+66.2)** | **dramatic recovery** |
| `mta_5_21_pla_2` | 67.0 (—) | 113.5 (+93.5) | **80.3 (+1.3)** | recovery already fine; massive stress lift |
| `nvt_5_21_pla_2(1)` | 63.4 (—) | 10.0 (−2.0) | **5.5 (+5.1)** | small lift |
| `oyj_5_22_pla_1'50_posiECG` | 73.2 (—) | 92.7 (—) | **83.5 (+0.4)** | already clean; no change |
| `oyj_5_22_pla_2'15_posiECG` | 73.8 (—) | 88.0 (−0.4) | **9.5 (—)** | unchanged |

**Summary.** Phase-isolation **helps about half** the `pla` recordings — `mta_5_19_pla_2'20`, `mta_5_21_pla_2'30(1)`, and `mta_5_21_pla_2` (the last via plank tachycardia) all get large lifts. But four `pla` recordings still show < 6 bpm in recovery, indicating that the post-plank pantompkins failure is not just about adaptive-state contamination — pantompkins fundamentally struggles with the recovery signal in those cases. The [neurokit-method variant](data-processing-using-neurokit.md) resolves the remaining ones.

**Medi recordings** are essentially unchanged (within ±4 R-peaks) — confirming that phase splitting is benign on already-clean recordings.

---

## 4. Per-recording peak plots

Generated by `uv run python -m src.plots` after a one-line patch to forward `phase_aware=True`, or by the inline driver:

```python
from src.plots import plot_br_full, plot_ecg_full
plot_br_full("data/<rec>.txt", "outputs/phase/br_full_<base>.png", phase_aware=True)
plot_ecg_full("data/<rec>.txt", "outputs/phase/ecg_full_<base>.png", phase_aware=True)
```

Phases shaded: rest blue, stress orange, recovery green. ECG always shown flipped (negative-polarity algorithm view).

### 4.1 `mta-5-17-medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta-5-17-medi.png) | ![ECG](outputs/phase/ecg_full_mta-5-17-medi.png) |

### 4.2 `mta-5-17-medi (1)` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta-5-17-medi_(1).png) | ![ECG](outputs/phase/ecg_full_mta-5-17-medi_(1).png) |

### 4.3 `mta-5-17-pla-1'26` — plank 1 m 26 s

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta-5-17-pla-1_26.png) | ![ECG](outputs/phase/ecg_full_mta-5-17-pla-1_26.png) |

### 4.4 `mta-5-17-pla-2` — plank 2 m

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta-5-17-pla-2.png) | ![ECG](outputs/phase/ecg_full_mta-5-17-pla-2.png) |

### 4.5 `mta2_5_19_medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta2_5_19_medi.png) | ![ECG](outputs/phase/ecg_full_mta2_5_19_medi.png) |

### 4.6 `mta_5_19_medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta_5_19_medi.png) | ![ECG](outputs/phase/ecg_full_mta_5_19_medi.png) |

### 4.7 `mta_5_19_medi (1)` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta_5_19_medi_(1).png) | ![ECG](outputs/phase/ecg_full_mta_5_19_medi_(1).png) |

### 4.8 `mta_5_19_pla_1'40` — plank 1 m 40 s

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta_5_19_pla_1_40.png) | ![ECG](outputs/phase/ecg_full_mta_5_19_pla_1_40.png) |

### 4.9 `mta_5_19_pla_2'20` — plank 2 m 20 s

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta_5_19_pla_2_20.png) | ![ECG](outputs/phase/ecg_full_mta_5_19_pla_2_20.png) |

### 4.10 `mta_5_21_medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta_5_21_medi.png) | ![ECG](outputs/phase/ecg_full_mta_5_21_medi.png) |

### 4.11 `mta_5_21_pla_2` — plank 2 m

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta_5_21_pla_2.png) | ![ECG](outputs/phase/ecg_full_mta_5_21_pla_2.png) |

### 4.12 `mta_5_21_pla_2'30(1)` — plank 2 m 30 s

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_mta_5_21_pla_2_30%281%29.png) | ![ECG](outputs/phase/ecg_full_mta_5_21_pla_2_30%281%29.png) |

### 4.13 `nvt_5_21_medi` — medi

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_nvt_5_21_medi.png) | ![ECG](outputs/phase/ecg_full_nvt_5_21_medi.png) |

### 4.14 `nvt_5_21_pla_2(1)` — plank 2 m

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_nvt_5_21_pla_2%281%29.png) | ![ECG](outputs/phase/ecg_full_nvt_5_21_pla_2%281%29.png) |

### 4.15 `ljh_5_21_medi_posiECG` — medi (positive-polarity ECG)

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_ljh_5_21_medi_posiECG.png) | ![ECG](outputs/phase/ecg_full_ljh_5_21_medi_posiECG.png) |

### 4.16 `oyj_5_22_medi_posiECG` — medi (positive-polarity ECG)

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_oyj_5_22_medi_posiECG.png) | ![ECG](outputs/phase/ecg_full_oyj_5_22_medi_posiECG.png) |

### 4.17 `oyj_5_22_pla_1'50_posiECG` — plank 1 m 50 s (positive-polarity ECG)

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_oyj_5_22_pla_1_50_posiECG.png) | ![ECG](outputs/phase/ecg_full_oyj_5_22_pla_1_50_posiECG.png) |

### 4.18 `oyj_5_22_pla_2'15_posiECG` — plank 2 m 15 s (positive-polarity ECG)

| BR | ECG |
|---|---|
| ![BR](outputs/phase/br_full_oyj_5_22_pla_2_15_posiECG.png) | ![ECG](outputs/phase/ecg_full_oyj_5_22_pla_2_15_posiECG.png) |

---

## 5. Verdict

Phase-isolation is **a principled fix that recovers some recordings** (notably `mta_5_19_pla_2'20` and `mta_5_21_pla_2'30(1)`) but does not solve the full post-plank degradation problem. Four `pla` recordings still report < 6 bpm in recovery despite each phase running its own detector instance, meaning pantompkins on the recovery segment alone — uncontaminated by plank — still fails.

The recommended production change is the [neurokit-method variant](data-processing-using-neurokit.md), which resolves all remaining cases. Phase-isolation can be stacked on top of neurokit for a small additional defensive benefit but is not strictly necessary.

## 6. Reproduce

```bash
uv run python -c "
import os
from src.io import list_recordings
from src.plots import plot_br_full, plot_ecg_full
os.makedirs('outputs/phase', exist_ok=True)
for p in list_recordings('data'):
    base = os.path.splitext(os.path.basename(p))[0].replace(' ', '_').replace(chr(39), '_')
    plot_br_full(p,  f'outputs/phase/br_full_{base}.png',  phase_aware=True)
    plot_ecg_full(p, f'outputs/phase/ecg_full_{base}.png', phase_aware=True)
"
```
