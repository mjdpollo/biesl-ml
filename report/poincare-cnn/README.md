# Poincaré-image 2D-CNN — report bundle

Self-contained bundle: ECG RR (NN) Poincaré plots → 64×64 log-count images → small 2D-CNN, evaluated with leave-one-recording-out (LORO).

## Contents

| Document | What |
|---|---|
| [poincare-cnn-window-comparison.md](poincare-cnn-window-comparison.md) | **Start here** — 60 s vs 2 min head-to-head + takeaway |
| [poincare-cnn-report.md](poincare-cnn-report.md) | Full 60-s run (recommended setting) |
| [poincare-cnn-report_2min.md](poincare-cnn-report_2min.md) | Full 2-min run (original spec; plank collapses) |

## Headline (LORO macro-F1)

| window | windows | acc | macro-F1 | F1[plank] |
|---|---:|---:|---:|---:|
| **60 s** | 382 | 0.562 | **0.249** | 0.090 |
| 2 min | 271 | 0.571 | 0.199 | 0.000 |

60 s is the better setting; the 2-min window leaves plank with only 5 windows. See the comparison doc for details.

## Figures

![comparison](figures/window_comparison.png)

| 60 s confusion | 2 min confusion |
|---|---|
| ![](figures/confusion_loro.png) | ![](figures/confusion_loro_2min.png) |

### Sample Poincaré images (60 s)

![samples](figures/samples_by_class.png)
