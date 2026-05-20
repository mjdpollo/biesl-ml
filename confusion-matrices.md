# Confusion matrices — all models × protocols × feature configs

Companion file to [summary-before-data-processing.md](summary-before-data-processing.md). Generated from the run JSONs under `outputs/`. Rows are true labels, columns are predictions.

Heatmap PNGs of every matrix live in [figures/confusion/](figures/confusion/). Regenerate with:

```bash
uv run python scripts/show_confusion_matrices.py > confusion-matrices.md
```

## LORO confusion matrices (sum across 7 folds)


### Classical — PDF features only

#### KNN — LORO — PDF features only  *(n_test=180)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 120 | 11 | 0 |
| **meditation** | 11 | 34 | 0 |
| **stress** | 4 | 0 | 0 |

#### RANDOMFOREST — LORO — PDF features only  *(n_test=180)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 125 | 6 | 0 |
| **meditation** | 8 | 37 | 0 |
| **stress** | 4 | 0 | 0 |

#### XGBOOST — LORO — PDF features only  *(n_test=180)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 126 | 5 | 0 |
| **meditation** | 11 | 34 | 0 |
| **stress** | 3 | 0 | 1 |

### Classical — PDF + temperature

#### KNN — LORO — PDF + temperature  *(n_test=180)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 122 | 9 | 0 |
| **meditation** | 18 | 27 | 0 |
| **stress** | 4 | 0 | 0 |

#### RANDOMFOREST — LORO — PDF + temperature  *(n_test=180)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 124 | 7 | 0 |
| **meditation** | 10 | 35 | 0 |
| **stress** | 4 | 0 | 0 |

#### XGBOOST — LORO — PDF + temperature  *(n_test=180)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 126 | 5 | 0 |
| **meditation** | 5 | 40 | 0 |
| **stress** | 4 | 0 | 0 |

### 1D-CNN — LORO

#### 1D-CNN — LORO — PDF channels (3 ch)  *(n_test=174)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 83 | 33 | 9 |
| **meditation** | 19 | 26 | 0 |
| **stress** | 2 | 2 | 0 |

#### 1D-CNN — LORO — PDF + temperature (4 ch)  *(n_test=174)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 105 | 11 | 9 |
| **meditation** | 14 | 31 | 0 |
| **stress** | 2 | 0 | 2 |


## Random-split confusion matrices (sum across 5 seeds)


### Classical — PDF features only

#### KNN — random 70:15:15 — PDF features only  *(n_test=135)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 90 | 9 | 0 |
| **meditation** | 10 | 24 | 0 |
| **stress** | 2 | 0 | 0 |

#### RANDOMFOREST — random 70:15:15 — PDF features only  *(n_test=135)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 93 | 6 | 0 |
| **meditation** | 6 | 28 | 0 |
| **stress** | 2 | 0 | 0 |

#### XGBOOST — random 70:15:15 — PDF features only  *(n_test=135)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 94 | 5 | 0 |
| **meditation** | 5 | 29 | 0 |
| **stress** | 2 | 0 | 0 |

### Classical — PDF + temperature

#### KNN — random 70:15:15 — PDF + temperature  *(n_test=135)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 90 | 9 | 0 |
| **meditation** | 13 | 21 | 0 |
| **stress** | 2 | 0 | 0 |

#### RANDOMFOREST — random 70:15:15 — PDF + temperature  *(n_test=135)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 93 | 6 | 0 |
| **meditation** | 6 | 28 | 0 |
| **stress** | 2 | 0 | 0 |

#### XGBOOST — random 70:15:15 — PDF + temperature  *(n_test=135)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 94 | 5 | 0 |
| **meditation** | 6 | 28 | 0 |
| **stress** | 2 | 0 | 0 |

### 1D-CNN — random 70:15:15

#### 1D-CNN — random 70:15:15 — PDF channels (3 ch)  *(n_test=135)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 84 | 8 | 2 |
| **meditation** | 7 | 29 | 2 |
| **stress** | 1 | 0 | 2 |

#### 1D-CNN — random 70:15:15 — PDF + temperature (4 ch)  *(n_test=135)*
| true \\ pred | baseline | meditation | stress |
|---|---|---|---|
| **baseline** | 90 | 0 | 4 |
| **meditation** | 2 | 36 | 0 |
| **stress** | 2 | 0 | 1 |
