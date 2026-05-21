# Confusion matrices — local-only, 4 classes (rest / meditation / stress / recovery)

Companion file to [report.md](report.md). Generated from the per-protocol JSONs under `outputs/`.

**Reading the tables.** Rows are true labels, columns are predictions. **Each row is row-normalized to 100 %** (true-class recall view): the cell at (`stress`, `stress`) is the percentage of actual `stress` windows the model correctly predicted as `stress`. The `support` column shows the raw count of true samples per row.

**Reading the PNGs.** Same row-normalization, fixed colour scale 0–100 %. The y-axis tick labels include the per-row support count.

Heatmap PNGs live in [figures/confusion/](figures/confusion/). Regenerate with:

```bash
uv run python scripts/show_confusion_matrices.py > confusion-matrices.md
```

## LORO confusion matrices (sum across recording folds)


### Classical — PDF features only

#### KNN — LORO  *(n_test=182)*
| true \\ pred | rest | meditation | stress | recovery | support |
|---|---|---|---|---|---|
| **rest** | 68.1% | 6.9% | 6.9% | 18.1% | 72 |
| **meditation** | 5.7% | 60.0% | 8.6% | 25.7% | 35 |
| **stress** | 20.0% | 0.0% | 60.0% | 20.0% | 5 |
| **recovery** | 22.9% | 10.0% | 1.4% | 65.7% | 70 |

#### RANDOMFOREST — LORO  *(n_test=182)*
| true \\ pred | rest | meditation | stress | recovery | support |
|---|---|---|---|---|---|
| **rest** | 87.5% | 2.8% | 0.0% | 9.7% | 72 |
| **meditation** | 20.0% | 71.4% | 0.0% | 8.6% | 35 |
| **stress** | 20.0% | 0.0% | 0.0% | 80.0% | 5 |
| **recovery** | 22.9% | 2.9% | 0.0% | 74.3% | 70 |

#### XGBOOST — LORO  *(n_test=182)*
| true \\ pred | rest | meditation | stress | recovery | support |
|---|---|---|---|---|---|
| **rest** | 79.2% | 2.8% | 0.0% | 18.1% | 72 |
| **meditation** | 14.3% | 80.0% | 0.0% | 5.7% | 35 |
| **stress** | 20.0% | 0.0% | 80.0% | 0.0% | 5 |
| **recovery** | 22.9% | 1.4% | 0.0% | 75.7% | 70 |

### 1D-CNN — PDF channels

#### 1D-CNN — LORO  *(n_test=177)*
| true \\ pred | rest | meditation | stress | recovery | support |
|---|---|---|---|---|---|
| **rest** | 16.7% | 1.4% | 12.5% | 69.4% | 72 |
| **meditation** | 34.3% | 31.4% | 25.7% | 8.6% | 35 |
| **stress** | 20.0% | 20.0% | 60.0% | 0.0% | 5 |
| **recovery** | 53.8% | 1.5% | 4.6% | 40.0% | 65 |


## Random 70:15:15 confusion matrices (sum across 5 seeds)


### Classical — PDF features only

#### KNN — random 70:15:15  *(n_test=140)*
| true \\ pred | rest | meditation | stress | recovery | support |
|---|---|---|---|---|---|
| **rest** | 73.8% | 6.6% | 0.0% | 19.7% | 61 |
| **meditation** | 8.0% | 84.0% | 0.0% | 8.0% | 25 |
| **stress** | 40.0% | 0.0% | 0.0% | 60.0% | 5 |
| **recovery** | 30.6% | 10.2% | 0.0% | 59.2% | 49 |

#### RANDOMFOREST — random 70:15:15  *(n_test=140)*
| true \\ pred | rest | meditation | stress | recovery | support |
|---|---|---|---|---|---|
| **rest** | 86.9% | 0.0% | 0.0% | 13.1% | 61 |
| **meditation** | 8.0% | 80.0% | 0.0% | 12.0% | 25 |
| **stress** | 40.0% | 0.0% | 0.0% | 60.0% | 5 |
| **recovery** | 26.5% | 2.0% | 0.0% | 71.4% | 49 |

#### XGBOOST — random 70:15:15  *(n_test=140)*
| true \\ pred | rest | meditation | stress | recovery | support |
|---|---|---|---|---|---|
| **rest** | 82.0% | 1.6% | 0.0% | 16.4% | 61 |
| **meditation** | 4.0% | 92.0% | 0.0% | 4.0% | 25 |
| **stress** | 20.0% | 20.0% | 40.0% | 20.0% | 5 |
| **recovery** | 16.3% | 4.1% | 0.0% | 79.6% | 49 |

### 1D-CNN — PDF channels

#### 1D-CNN — random 70:15:15  *(n_test=135)*
| true \\ pred | rest | meditation | stress | recovery | support |
|---|---|---|---|---|---|
| **rest** | 94.6% | 1.8% | 0.0% | 3.6% | 56 |
| **meditation** | 0.0% | 95.7% | 0.0% | 4.3% | 23 |
| **stress** | 0.0% | 0.0% | 66.7% | 33.3% | 3 |
| **recovery** | 18.9% | 5.7% | 0.0% | 75.5% | 53 |
