# Confusion matrices — local-only, 3 classes (rest / meditation / plank)

Companion file to [report.md](report.md). Generated from the per-protocol JSONs under `outputs/`.

**Reading the tables.** Rows are true labels, columns are predictions. **Each row is row-normalized to 100 %** (true-class recall view): the cell at (`plank`, `plank`) is the percentage of actual `plank` windows the model correctly predicted as `plank`. The `support` column shows the raw count of true samples per row.

**Reading the PNGs.** Same row-normalization, fixed colour scale 0–100 %. The y-axis tick labels include the per-row support count.

Heatmap PNGs live in [figures/confusion/](figures/confusion/). Regenerate with:

```bash
uv run python scripts/show_confusion_matrices.py > confusion-matrices.md
```

## LORO confusion matrices (sum across recording folds)


### Classical — PDF features only

#### KNN — LORO  *(n_test=112)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 91.7% | 8.3% | 0.0% | 72 |
| **meditation** | 22.9% | 71.4% | 5.7% | 35 |
| **plank** | 60.0% | 0.0% | 40.0% | 5 |

#### RANDOMFOREST — LORO  *(n_test=112)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 91.7% | 8.3% | 0.0% | 72 |
| **meditation** | 20.0% | 80.0% | 0.0% | 35 |
| **plank** | 60.0% | 0.0% | 40.0% | 5 |

#### XGBOOST — LORO  *(n_test=112)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 93.1% | 6.9% | 0.0% | 72 |
| **meditation** | 17.1% | 82.9% | 0.0% | 35 |
| **plank** | 20.0% | 0.0% | 80.0% | 5 |

### 1D-CNN — PDF channels

#### 1D-CNN — LORO  *(n_test=112)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 69.4% | 19.4% | 11.1% | 72 |
| **meditation** | 34.3% | 65.7% | 0.0% | 35 |
| **plank** | 0.0% | 20.0% | 80.0% | 5 |


## Random 70:15:15 confusion matrices (sum across 5 seeds)


### Classical — PDF features only

#### KNN — random 70:15:15  *(n_test=85)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 92.3% | 7.7% | 0.0% | 52 |
| **meditation** | 25.0% | 68.8% | 6.2% | 32 |
| **plank** | 0.0% | 0.0% | 100.0% | 1 |

#### RANDOMFOREST — random 70:15:15  *(n_test=85)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 86.5% | 13.5% | 0.0% | 52 |
| **meditation** | 12.5% | 87.5% | 0.0% | 32 |
| **plank** | 0.0% | 0.0% | 100.0% | 1 |

#### XGBOOST — random 70:15:15  *(n_test=85)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 94.2% | 5.8% | 0.0% | 52 |
| **meditation** | 21.9% | 78.1% | 0.0% | 32 |
| **plank** | 0.0% | 0.0% | 100.0% | 1 |

### 1D-CNN — PDF channels

#### 1D-CNN — random 70:15:15  *(n_test=85)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 88.5% | 9.6% | 1.9% | 52 |
| **meditation** | 3.1% | 96.9% | 0.0% | 32 |
| **plank** | 0.0% | 0.0% | 100.0% | 1 |
