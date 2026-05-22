  -> figures/confusion/loro__classical_knn.png
  -> figures/confusion/loro__classical_randomforest.png
  -> figures/confusion/loro__classical_xgboost.png
  -> figures/confusion/loro__cnn.png
  -> figures/confusion/randomsplit__classical_knn.png
  -> figures/confusion/randomsplit__classical_randomforest.png
  -> figures/confusion/randomsplit__classical_xgboost.png
  -> figures/confusion/randomsplit__cnn.png

All PNGs saved under figures/confusion/

## LORO confusion matrices (sum across recording folds)


### Classical — PDF features only

#### KNN — LORO  *(n_test=222)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 91.7% | 8.3% | 0.0% | 144 |
| **meditation** | 34.9% | 65.1% | 0.0% | 63 |
| **plank** | 46.7% | 26.7% | 26.7% | 15 |

#### RANDOMFOREST — LORO  *(n_test=222)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 95.8% | 4.2% | 0.0% | 144 |
| **meditation** | 12.7% | 87.3% | 0.0% | 63 |
| **plank** | 6.7% | 26.7% | 66.7% | 15 |

#### XGBOOST — LORO  *(n_test=222)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 94.4% | 5.6% | 0.0% | 144 |
| **meditation** | 12.7% | 87.3% | 0.0% | 63 |
| **plank** | 6.7% | 0.0% | 93.3% | 15 |

### 1D-CNN — PDF channels

#### 1D-CNN — LORO  *(n_test=222)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 76.4% | 21.5% | 2.1% | 144 |
| **meditation** | 38.1% | 61.9% | 0.0% | 63 |
| **plank** | 33.3% | 6.7% | 60.0% | 15 |


## Random 70:15:15 confusion matrices (sum across 5 seeds)


### Classical — PDF features only

#### KNN — random 70:15:15  *(n_test=170)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 94.5% | 4.5% | 0.9% | 110 |
| **meditation** | 24.0% | 76.0% | 0.0% | 50 |
| **plank** | 30.0% | 20.0% | 50.0% | 10 |

#### RANDOMFOREST — random 70:15:15  *(n_test=170)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 98.2% | 1.8% | 0.0% | 110 |
| **meditation** | 14.0% | 86.0% | 0.0% | 50 |
| **plank** | 0.0% | 10.0% | 90.0% | 10 |

#### XGBOOST — random 70:15:15  *(n_test=170)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 97.3% | 2.7% | 0.0% | 110 |
| **meditation** | 14.0% | 86.0% | 0.0% | 50 |
| **plank** | 10.0% | 0.0% | 90.0% | 10 |

### 1D-CNN — PDF channels

#### 1D-CNN — random 70:15:15  *(n_test=170)*
| true \\ pred | rest | meditation | plank | support |
|---|---|---|---|---|
| **rest** | 88.2% | 11.8% | 0.0% | 110 |
| **meditation** | 16.0% | 84.0% | 0.0% | 50 |
| **plank** | 0.0% | 0.0% | 100.0% | 10 |
