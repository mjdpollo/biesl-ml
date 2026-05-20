# Confusion matrices — all models × protocols × feature configs

Companion file to [summary-before-data-processing.md](summary-before-data-processing.md) and [report.md](report.md). Generated from the run JSONs under `outputs/`.

**Reading the tables.** Rows are true labels, columns are predictions. **Each row is normalized to 100 %** (true-class recall view): the cell at (`stress`, `stress`) is the percentage of actual `stress` windows the model correctly predicted as `stress`. A `support` column shows the raw count of true samples per row — crucial when `stress` has only 4 samples total under LORO and a 100 % means "1 out of 1 got it".

**Reading the PNGs.** Same row-normalization, fixed colour scale 0–100 %, so heatmaps are directly comparable across (model × protocol × config). The y-axis tick labels include the support count.

Heatmap PNGs of every matrix live in [figures/confusion/](figures/confusion/). Regenerate with:

```bash
uv run python scripts/show_confusion_matrices.py > confusion-matrices.md
```

## LORO confusion matrices (sum across 7 folds)


### Classical — PDF features only

#### KNN — LORO — PDF features only  *(n_test=180)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 91.6% | 8.4% | 0.0% | 131 |
| **meditation** | 24.4% | 75.6% | 0.0% | 45 |
| **stress** | 100.0% | 0.0% | 0.0% | 4 |

#### RANDOMFOREST — LORO — PDF features only  *(n_test=180)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 95.4% | 4.6% | 0.0% | 131 |
| **meditation** | 17.8% | 82.2% | 0.0% | 45 |
| **stress** | 100.0% | 0.0% | 0.0% | 4 |

#### XGBOOST — LORO — PDF features only  *(n_test=180)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 96.2% | 3.8% | 0.0% | 131 |
| **meditation** | 24.4% | 75.6% | 0.0% | 45 |
| **stress** | 75.0% | 0.0% | 25.0% | 4 |

### Classical — PDF + temperature

#### KNN — LORO — PDF + temperature  *(n_test=180)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 93.1% | 6.9% | 0.0% | 131 |
| **meditation** | 40.0% | 60.0% | 0.0% | 45 |
| **stress** | 100.0% | 0.0% | 0.0% | 4 |

#### RANDOMFOREST — LORO — PDF + temperature  *(n_test=180)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 94.7% | 5.3% | 0.0% | 131 |
| **meditation** | 22.2% | 77.8% | 0.0% | 45 |
| **stress** | 100.0% | 0.0% | 0.0% | 4 |

#### XGBOOST — LORO — PDF + temperature  *(n_test=180)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 96.2% | 3.8% | 0.0% | 131 |
| **meditation** | 11.1% | 88.9% | 0.0% | 45 |
| **stress** | 100.0% | 0.0% | 0.0% | 4 |

### 1D-CNN — LORO

#### 1D-CNN — LORO — PDF channels (3 ch)  *(n_test=174)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 66.4% | 26.4% | 7.2% | 125 |
| **meditation** | 42.2% | 57.8% | 0.0% | 45 |
| **stress** | 50.0% | 50.0% | 0.0% | 4 |

#### 1D-CNN — LORO — PDF + temperature (4 ch)  *(n_test=174)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 84.0% | 8.8% | 7.2% | 125 |
| **meditation** | 31.1% | 68.9% | 0.0% | 45 |
| **stress** | 50.0% | 0.0% | 50.0% | 4 |


## Random-split confusion matrices (sum across 5 seeds)


### Classical — PDF features only

#### KNN — random 70:15:15 — PDF features only  *(n_test=135)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 90.9% | 9.1% | 0.0% | 99 |
| **meditation** | 29.4% | 70.6% | 0.0% | 34 |
| **stress** | 100.0% | 0.0% | 0.0% | 2 |

#### RANDOMFOREST — random 70:15:15 — PDF features only  *(n_test=135)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 93.9% | 6.1% | 0.0% | 99 |
| **meditation** | 17.6% | 82.4% | 0.0% | 34 |
| **stress** | 100.0% | 0.0% | 0.0% | 2 |

#### XGBOOST — random 70:15:15 — PDF features only  *(n_test=135)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 94.9% | 5.1% | 0.0% | 99 |
| **meditation** | 14.7% | 85.3% | 0.0% | 34 |
| **stress** | 100.0% | 0.0% | 0.0% | 2 |

### Classical — PDF + temperature

#### KNN — random 70:15:15 — PDF + temperature  *(n_test=135)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 90.9% | 9.1% | 0.0% | 99 |
| **meditation** | 38.2% | 61.8% | 0.0% | 34 |
| **stress** | 100.0% | 0.0% | 0.0% | 2 |

#### RANDOMFOREST — random 70:15:15 — PDF + temperature  *(n_test=135)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 93.9% | 6.1% | 0.0% | 99 |
| **meditation** | 17.6% | 82.4% | 0.0% | 34 |
| **stress** | 100.0% | 0.0% | 0.0% | 2 |

#### XGBOOST — random 70:15:15 — PDF + temperature  *(n_test=135)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 94.9% | 5.1% | 0.0% | 99 |
| **meditation** | 17.6% | 82.4% | 0.0% | 34 |
| **stress** | 100.0% | 0.0% | 0.0% | 2 |

### 1D-CNN — random 70:15:15

#### 1D-CNN — random 70:15:15 — PDF channels (3 ch)  *(n_test=135)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 89.4% | 8.5% | 2.1% | 94 |
| **meditation** | 18.4% | 76.3% | 5.3% | 38 |
| **stress** | 33.3% | 0.0% | 66.7% | 3 |

#### 1D-CNN — random 70:15:15 — PDF + temperature (4 ch)  *(n_test=135)*
| true \\ pred | baseline | meditation | stress | support |
|---|---|---|---|---|
| **baseline** | 95.7% | 0.0% | 4.3% | 94 |
| **meditation** | 5.3% | 94.7% | 0.0% | 38 |
| **stress** | 66.7% | 0.0% | 33.3% | 3 |
