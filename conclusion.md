# Conclusions and Next Steps — Window-size Sweep

Synthesises three sibling experiments — same 18-recording dataset, same neurokit R-peak detector, same 8 PDF features, same 4 models — differing only in window length and the resulting per-window sample size:

- **60 s window / 30 s overlap** — [report-60window-30overlap.md](report-60window-30overlap.md)
- **40 s window / 20 s overlap** — [report-40window-20overlap.md](report-40window-20overlap.md)
- **30 s window / 15 s overlap** — [report-30window-15overlap.md](report-30window-15overlap.md)

Signal preprocessing is documented in [data-processing-using-neurokit.md](data-processing-using-neurokit.md) (production-aligned variant).

---

## 1. Headline numbers at a glance

### 1.1 Dataset size

| Window / overlap | total windows | rest | meditation | plank |
|---|---:|---:|---:|---:|
| 60 s / 30 s | 222 | 144 | 63 | **15** |
| **40 s / 20 s** | 376 | 234 | 108 | **34** |
| 30 s / 15 s | **529** | 324 | 153 | **52** |

Shrinking the window roughly doubles the example count in each step; plank, which is the rare class, scales **3.5×** between 60 s and 30 s.

### 1.2 LORO (honest cross-recording) — best classical model per window

| Window | Best model | Acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---|---:|---:|---:|---:|---:|
| 60 s / 30 s | XGBoost | 0.927 | 0.864 | 0.943 | 0.447 | 0.426 |
| **40 s / 20 s** | XGBoost | **0.945** | **0.879** | **0.958** | 0.458 | **0.484** |
| 30 s / 15 s | XGBoost | 0.904 | 0.809 | 0.919 | 0.419 | 0.486 |

**40 s / 20 s wins on every classical LORO metric except F1[plank], where it's essentially tied with 30 s.**

### 1.3 LORO — 1D-CNN

| Window | Acc | macro-F1 | F1[rest] | F1[meditation] | F1[plank] |
|---|---:|---:|---:|---:|---:|
| 60 s / 30 s | 0.753 | 0.639 | 0.696 | 0.243 | 0.315 |
| **40 s / 20 s** | 0.820 | 0.740 | 0.754 | 0.340 | **0.481** |
| 30 s / 15 s | 0.842 | 0.748 | 0.790 | 0.362 | 0.453 |

The CNN gains far more from the extra examples than the classical models — +0.11 macro-F1 from 60 s to 40 s, vs +0.015 for XGBoost. With 60 s windows the CNN had only 222 training examples; 376 (40 s) and 529 (30 s) are the first sample sizes at which it begins to compete.

### 1.4 LORO confusion — XGBoost plank performance

| Window | plank recall | plank precision | non-plank false positives |
|---|---:|---:|---:|
| 60 s / 30 s | 14 / 15 = **93.3 %** | 14 / 14 = **100 %** | 0 of 207 |
| **40 s / 20 s** | 32 / 34 = **94.1 %** | 32 / 32 = **100 %** | 0 of 342 |
| 30 s / 15 s | 50 / 52 = **96.2 %** | 50 / 52 = 96.2 % | 2 of 477 |

At 30 s the recall is highest but precision starts to slip — 2 non-plank windows now bleed into the plank prediction column.

### 1.5 Random-split LORO gap (overlap-leakage diagnostic)

| Window | XGBoost LORO | XGBoost random-split | Δ | CNN LORO | CNN random-split | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 60 s / 30 s | 0.864 | 0.926 | **+0.06** | 0.639 | 0.899 | +0.26 |
| 40 s / 20 s | 0.879 | 0.958 | +0.08 | 0.740 | 0.928 | +0.19 |
| 30 s / 15 s | 0.809 | 0.950 | **+0.14** | 0.748 | 0.959 | +0.21 |

The classical gap roughly doubles between 60 s (+0.06) and 30 s (+0.14). At 30 s the random-split numbers are mostly measuring the leakage between 15-s-shifted neighbours, not generalisation. Use LORO numbers for any cross-subject claim — at **all** window sizes, but emphatically at 30 s.

---

## 2. Why the inverted-U?

Three competing effects determine the LORO score as window shrinks:

1. **More training examples** (good): each halving of step roughly doubles the windows. With only 18 recordings the model is sample-starved at 60 s; more windows help every model.
2. **HRV LF/HF Welch quality** (bad as window shrinks): `features.pdf` specifies 60 s Welch segments. At 40 s the LF (0.04–0.15 Hz) band sits at the spectral resolution limit and at 30 s many windows return NaN (see `_lf_hf`'s early-return at `t[-1] - t[0] < 30.0`). The median imputer fills these NaNs with the global median, which is uninformative for those windows.
3. **Within-recording label persistence** (bad as window shrinks): smaller windows mean the same physiological state is sampled more times, which can over-represent within-recording quirks and hurt LORO generalisation while inflating random-split scores.

Effect (1) dominates from 60 s → 40 s (XGBoost LORO 0.864 → 0.879). Effects (2) and (3) start to dominate from 40 s → 30 s (LORO drops to 0.809). 40 s appears to be the sweet spot for **this** dataset, model, and feature schema.

---

## 3. Conclusions

1. **40 s / 20 s is the new production candidate.** XGBoost LORO macro-F1 0.879 vs 0.864 (60 s) and 0.809 (30 s). Plank recall 94.1 % with zero false positives. The HRV LF/HF feature noise from shrinking 60 s → 40 s is more than paid for by the +70 % training-set growth.
2. **30 s / 15 s overshoots.** The plank-recall gain (96.2 % vs 94.1 %) is real but small, and it costs ~+0.07 macro-F1 elsewhere because the LF/HF features turn NaN-heavy. Not recommended.
3. **60 s / 30 s remains the most spec-faithful** (the only setting where the PDF's 60 s Welch is exactly satisfied) and is still a strong baseline. Keep it as the reference for any HRV-only analysis where feature quality is paramount.
4. **The 1D-CNN at 40 s and 30 s windows is finally competitive** on LORO macro-F1 (~0.74) but **still inferior to XGBoost** (0.879 at 40 s). Its random-split leap is a leakage signal, not generalisation. Don't ship the CNN until either (a) the data scales 5–10×, or (b) overlap is removed for CNN training. Classical XGBoost on PDF features remains the recommendation.
5. **Plank classification is no longer the bottleneck.** With 34 (40 s) or 52 (30 s) plank windows, XGBoost gets ≥ 94 % recall on every variant. The next bottleneck is the **meditation vs rest** confusion — F1[meditation] sits at 0.42–0.46 across all three window sizes, dragged by recordings (`mta-5-17-medi`, `mta-5-17-medi (1)`, `oyj_5_22_medi_posiECG`) where rest-baseline HR is indistinguishable from breathing-exercise HR. This is the next thing to attack.

---

## 4. Next steps

1. **Adopt 40 s / 20 s as the production default.** Change `features.WINDOW_S = 40.0` in [src/features.py](src/features.py) and re-run downstream consumers. Keep `OVERLAP = 0.5`. Document the change in [data-processing-using-neurokit.md](data-processing-using-neurokit.md) §2.1.
2. **Target the meditation/rest boundary.** With plank now handled, F1[meditation] is the loss leader. Two cheap experiments:
   - Add a respiratory-cycle-coherence feature (most `medi` recordings have a paced breathing rhythm; rest is unstructured). RR/RRV alone miss the rhythmic structure.
   - Try a per-subject HR z-score normalization: each subject's rest-baseline HR is captured in the first 5 minutes; meditation differs from rest by HR shift relative to that baseline rather than absolute bpm.
3. **Collect more `pla-*` recordings** to reach ≥ 100 plank windows at the production 40 s setting. The current 34 is enough to score F1[plank] = 0.48 but still single-recording-sensitive — `mta_5_19_pla_2'20` alone contributes 5/34.
4. **Re-evaluate temperature ablation** at the production 40 s window. The previous report retired temperature because it hurt classical macro-F1 by 2–6 points at the old 60 s/old detector. Under the new R-peak detector and shorter window the finding may flip.
5. **Visually verify the post-plank R-peaks** that neurokit recovered on `mta_5_19_pla_2'20` (the recording originally labelled an electrode-failure case) — if they turn out to be motion-artefact false positives, plank-class HRV features are noisier than they appear and the F1[plank] = 0.48 number is inflated.
6. **For the 1D-CNN**: either restrict to a no-overlap protocol (window step = window length) and accept the smaller dataset, or wait for ~5× more recordings before re-investing. The current 0.74–0.75 LORO is below the classical floor.

---

## 5. Reproduce all three experiments

```bash
# 60 s baseline (current default in src/features.py):
uv run python -m src.local_eval
uv run python -m src.dl_train
uv run python scripts/show_confusion_matrices.py > confusion-matrices.md

# 40 s and 30 s variants in one go (driver edits src/features.py and restores
# WINDOW_S=60.0 on exit, even on failure):
bash scripts/run_window_variants.sh

# Per-variant confusion tables + PNGs:
uv run python scripts/show_confusion_matrices.py \
    --inputs-dir outputs/win40_ov20 \
    --figures-dir figures/confusion/win40_ov20 \
  > confusion-matrices-40window-20overlap.md
uv run python scripts/show_confusion_matrices.py \
    --inputs-dir outputs/win30_ov15 \
    --figures-dir figures/confusion/win30_ov15 \
  > confusion-matrices-30window-15overlap.md
```
