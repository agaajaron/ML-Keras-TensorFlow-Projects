# Boston Housing Price Regression — Keras ANN Case Study

Predict median house prices in Boston suburbs using Artificial Neural Networks, exploring regularization, weight initialization, hyperparameter tuning, transfer learning, and SHAP explainability.

## Dataset

- **Source**: `tensorflow.keras.datasets.boston_housing` (StatLib / Carnegie Mellon)
- **Size**: 404 training + 102 test samples
- **Features**: 13 numeric attributes (crime rate, rooms, tax rate, etc.)
- **Target**: MEDV — median house value in $1,000s (regression)
- **Metric**: R² score

See [data/data.md](data/data.md) for full feature descriptions and SHAP findings.

## Models

| Model | Architecture | Key Technique | Notes |
|---|---|---|---|
| Skeleton MSE | 16→8→1 (Glorot) | Loss=MSE | Converges ~epoch 30 |
| Skeleton Huber | 16→8→1 (Glorot) | Loss=Huber | Better gradient stability |
| Wide & Deep | Input→30→30→concat→1 | Skip connection (Functional API) | Residual from raw input |
| He Normal | 16→8→1 (he_normal) | He initialization | Reduces vanishing gradients |
| BatchNorm | 16→BN→8→1 | Batch Normalization | Smoother learning curves |
| **BatchNorm+Dropout** | **16→BN→Drop→8→Drop→1** | **BN + Dropout** | **Best standard model** |
| **Tuned (RCV)** | Flexible n-layer | RandomizedSearchCV | **Best R² overall** |
| Transfer Learning | Frozen RCV base + new head | Weight reuse | Fine-tuning |

## Project Structure

```
casestudy-cnn-housing/
├── data/
│   └── data.md
├── models/
│   ├── functional_nn.h5      # Saved Wide & Deep model
│   └── RCV_model.h5          # Best RandomizedSearchCV model
├── notebooks/
│   └── casestudy-cnn-housing.ipynb
├── src/
│   ├── config.py        # Imports, seeds, GPU env
│   ├── data_loader.py   # boston_housing.load_data()
│   ├── preprocessing.py # StandardScaler on both X and Y
│   ├── utils.py         # plot_loss_curves, evaluate_r2, get_run_logdir
│   ├── model.py         # 7 build_model_* functions
│   ├── train.py         # Train all models, RandomizedSearchCV, transfer learning
│   ├── evaluate.py      # R² comparison table + SHAP analysis
│   └── visualize.py     # Y distribution, loss curves, SHAP summary plot
└── requirements.txt
```

## Key Findings

- **BatchNorm + Dropout together** generalize better than either technique alone
- **RandomizedSearchCV** reliably finds better hyperparameters than manual tuning
- **Transfer Learning** with frozen layers matches or beats training from scratch on this small dataset (404 samples)
- **SHAP analysis**: `LSTAT` (lower-status %) and `RM` (rooms per dwelling) are the strongest predictors
- Standardizing both X **and** Y is important for faster convergence in regression NNs
