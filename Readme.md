# Deep Learning training monitoring lab
## Readmission Risk NN

Neural network that predicts patient readmission risk for a healthcare AI startup (HealthTech Analytics), using training monitoring and callbacks to make training reliable.

## Overview
A baseline dense network overfit quickly: validation loss bottomed out at epoch 3 while training kept improving. An improved model adds regularization and monitoring so it trains efficiently and generalizes better.

## Techniques
- Dropout and batch normalization
- Gradient clipping
- EarlyStopping (monitors validation loss, `patience=5`, restores best weights)
- ModelCheckpoint (saves the best model)
- ReduceLROnPlateau
- TensorBoard for training visualization

## Results
| Model | Test recall |
|---|---|
| Baseline (50 epochs, no callbacks) | 0.5625 |
| Improved (stopped at epoch 23 of 100) | 0.5693 |

Early stopping saved about 77% of the planned training time. Both models perform similarly, which suggests the features, not the model, limit performance.

## Files
- `nn_training_lab.ipynb`: full analysis, training curves, and written reflection
- `readmission_data.csv`: patient readmission dataset (diabetic patients; target is readmitted within or after 30 days vs. not)