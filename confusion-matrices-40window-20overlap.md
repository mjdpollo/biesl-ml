  -> figures/confusion/win40_ov20/loro__classical_knn.png
  -> figures/confusion/win40_ov20/loro__classical_randomforest.png
  -> figures/confusion/win40_ov20/loro__classical_xgboost.png
  -> figures/confusion/win40_ov20/loro__cnn.png
  -> figures/confusion/win40_ov20/randomsplit__classical_knn.png
  -> figures/confusion/win40_ov20/randomsplit__classical_randomforest.png
  -> figures/confusion/win40_ov20/randomsplit__classical_xgboost.png
  -> figures/confusion/win40_ov20/randomsplit__cnn.png

All PNGs saved under figures/confusion/win40_ov20/

## LORO confusion matrices (sum across recording folds)


### Classical — PDF features only

#### KNN — LORO  *(n_test=376)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 91.0% | 9.0% | 0.0% | 234 |
| **meditation** | 25.0% | 75.0% | 0.0% | 108 |
| **plank** | 26.5% | 11.8% | 61.8% | 34 |

#### RANDOMFOREST — LORO  *(n_test=376)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 95.7% | 4.3% | 0.0% | 234 |
| **meditation** | 11.1% | 88.9% | 0.0% | 108 |
| **plank** | 8.8% | 2.9% | 88.2% | 34 |

#### XGBOOST — LORO  *(n_test=376)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 97.4% | 2.6% | 0.0% | 234 |
| **meditation** | 13.0% | 87.0% | 0.0% | 108 |
| **plank** | 2.9% | 2.9% | 94.1% | 34 |

### 1D-CNN — PDF channels

#### 1D-CNN — LORO  *(n_test=376)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 75.6% | 24.4% | 0.0% | 234 |
| **meditation** | 13.9% | 81.5% | 4.6% | 108 |
| **plank** | 0.0% | 5.9% | 94.1% | 34 |


## Random 70:15:15 confusion matrices (sum across 5 seeds)


### Classical — PDF features only

#### KNN — random 70:15:15  *(n_test=285)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 96.6% | 3.4% | 0.0% | 175 |
| **meditation** | 10.6% | 89.4% | 0.0% | 85 |
| **plank** | 20.0% | 8.0% | 72.0% | 25 |

#### RANDOMFOREST — random 70:15:15  *(n_test=285)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 96.6% | 3.4% | 0.0% | 175 |
| **meditation** | 5.9% | 94.1% | 0.0% | 85 |
| **plank** | 12.0% | 0.0% | 88.0% | 25 |

#### XGBOOST — random 70:15:15  *(n_test=285)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 97.7% | 2.3% | 0.0% | 175 |
| **meditation** | 4.7% | 95.3% | 0.0% | 85 |
| **plank** | 8.0% | 0.0% | 92.0% | 25 |

### 1D-CNN — PDF channels

#### 1D-CNN — random 70:15:15  *(n_test=285)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 92.0% | 8.0% | 0.0% | 175 |
| **meditation** | 12.9% | 87.1% | 0.0% | 85 |
| **plank** | 0.0% | 0.0% | 100.0% | 25 |
