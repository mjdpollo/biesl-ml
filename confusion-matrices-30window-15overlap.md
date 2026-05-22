  -> figures/confusion/win30_ov15/loro__classical_knn.png
  -> figures/confusion/win30_ov15/loro__classical_randomforest.png
  -> figures/confusion/win30_ov15/loro__classical_xgboost.png
  -> figures/confusion/win30_ov15/loro__cnn.png
  -> figures/confusion/win30_ov15/randomsplit__classical_knn.png
  -> figures/confusion/win30_ov15/randomsplit__classical_randomforest.png
  -> figures/confusion/win30_ov15/randomsplit__classical_xgboost.png
  -> figures/confusion/win30_ov15/randomsplit__cnn.png

All PNGs saved under figures/confusion/win30_ov15/

## LORO confusion matrices (sum across recording folds)


### Classical — PDF features only

#### KNN — LORO  *(n_test=529)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 90.7% | 9.3% | 0.0% | 324 |
| **meditation** | 24.2% | 75.8% | 0.0% | 153 |
| **plank** | 11.5% | 7.7% | 80.8% | 52 |

#### RANDOMFOREST — LORO  *(n_test=529)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 92.3% | 7.7% | 0.0% | 324 |
| **meditation** | 20.9% | 79.1% | 0.0% | 153 |
| **plank** | 7.7% | 3.8% | 88.5% | 52 |

#### XGBOOST — LORO  *(n_test=529)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 93.5% | 6.2% | 0.3% | 324 |
| **meditation** | 20.9% | 78.4% | 0.7% | 153 |
| **plank** | 1.9% | 1.9% | 96.2% | 52 |

### 1D-CNN — PDF channels

#### 1D-CNN — LORO  *(n_test=529)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 78.7% | 21.3% | 0.0% | 324 |
| **meditation** | 12.4% | 86.9% | 0.7% | 153 |
| **plank** | 0.0% | 13.5% | 86.5% | 52 |


## Random 70:15:15 confusion matrices (sum across 5 seeds)


### Classical — PDF features only

#### KNN — random 70:15:15  *(n_test=400)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 93.9% | 6.1% | 0.0% | 245 |
| **meditation** | 15.7% | 84.3% | 0.0% | 115 |
| **plank** | 10.0% | 0.0% | 90.0% | 40 |

#### RANDOMFOREST — random 70:15:15  *(n_test=400)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 93.9% | 6.1% | 0.0% | 245 |
| **meditation** | 13.0% | 87.0% | 0.0% | 115 |
| **plank** | 2.5% | 0.0% | 97.5% | 40 |

#### XGBOOST — random 70:15:15  *(n_test=400)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 95.5% | 4.5% | 0.0% | 245 |
| **meditation** | 8.7% | 90.4% | 0.9% | 115 |
| **plank** | 0.0% | 0.0% | 100.0% | 40 |

### 1D-CNN — PDF channels

#### 1D-CNN — random 70:15:15  *(n_test=400)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 93.1% | 6.9% | 0.0% | 245 |
| **meditation** | 2.6% | 97.4% | 0.0% | 115 |
| **plank** | 0.0% | 0.0% | 100.0% | 40 |
