# Dataset: Boston Housing (keras built-in)

## Source
- **API**: `tensorflow.keras.datasets.boston_housing.load_data()`
- **Original Source**: StatLib library, Carnegie Mellon University
- Houses in Boston suburbs, late 1970s

## Overview
| Property | Value |
|---|---|
| Training samples | 404 |
| Test samples | 102 |
| Features | 13 (all numeric) |
| Task | Regression — predict median house price |
| Target | MEDV — median value of owner-occupied homes ($1,000s) |
| Metric | MSE (training) / R² (evaluation) |

## Features
| Column | Description |
|---|---|
| CRIM | Per capita crime rate |
| ZN | Proportion of residential land zoned for lots >25,000 sq.ft |
| INDUS | Proportion of non-retail business acres |
| CHAS | Charles River dummy (1 = tract bounds river) |
| NOX | Nitric oxide concentration (parts per 10M) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of units built before 1940 |
| DIS | Weighted distance to Boston employment centres |
| RAD | Accessibility index to radial highways |
| TAX | Property tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk − 0.63)² — proportion of Black residents |
| LSTAT | % lower-status population |

## Preprocessing Applied
- `StandardScaler` applied to **both X and Y** (required for regression NNs to converge faster)
- Train/test split: 80/20 (`test_split=0.2, seed=113`)

## SHAP Feature Importance (from best model)
Top predictors by SHAP value magnitude:
1. **LSTAT** — % lower status population (negative impact on price)
2. **RM** — number of rooms (positive impact)
3. **DIS** — distance to employment centres
4. **NOX** — nitric oxide concentration

## Models Trained

### Model 1: Skeleton MSE — `build_model_skeleton_mse`
- `Dense(16,relu,Glorot) → Dense(8,relu,Glorot) → Dense(1,relu)`
- RMSprop(0.001), loss=MSE, 100 epochs
- Converges ~epoch 30; R² ≈ 0.70

### Model 2: Skeleton Huber — `build_model_skeleton_huber`
- Same architecture, loss=Huber() — less sensitive to outliers
- Converges ~epoch 20; slightly better R²

### Model 3: Functional Wide & Deep — `functional_nn`
- Functional API with skip/residual connection:
  `Input → Dense(30) → Dense(30) → concatenate(input, out) → Dense(1)`
- Huber loss; R² comparable to Model 2

### Model 4: He Normal Init — `build_model_he_normal`
- `Dense(16,relu,he_normal) → Dense(8,relu,he_normal) → Dense(1,relu,he_normal)`
- Two convergence points (vanishing gradient partially solved by He init)
- Some overfitting observed

### Model 5: BatchNormalization — `build_model_batch`
- `Dense(16,he) → BatchNorm → Dense(8,he) → Dense(1,he)`
- Smoother learning curves; BatchNorm addresses vanishing gradients

### Model 6: BatchNorm + Dropout — `build_model_dropout` ← **Best Standard Model**
- `Dense(16,he) → BatchNorm → Dropout(0.03) → Dense(8,he) → Dropout(0.01) → Dense(1,he)`
- BatchNorm + small Dropout together give best generalization

### Model 7: Flexible + RandomizedSearchCV — `build_model_flexible`
- Parameterized: `n_hidden_layer`, `n_neurons`, `active`, `learning_rate`
- L1 regularization; pyramid neuron halving per layer
- Wrapped in `KerasRegressor` for sklearn `RandomizedSearchCV`
- Best params found: typically 3–4 layers, 32–64 neurons, relu/selu
- **Best R² on test set** among all models

### Transfer Learning
- Loaded `RCV_model.h5` (best RandomizedSearchCV model)
- Replaced output layer with new `Dense(1, relu)`
- Froze all earlier layers, retrained only the new head
- Demonstrates weight reuse from a trained regression base

### TensorBoard + EarlyStopping
- `EarlyStopping(monitor='val_mse', patience=10)`
- TensorBoard logging via `keras.callbacks.TensorBoard`

## Conclusions
- **RandomizedSearchCV tuned model** achieves best R² by systematically exploring hyperparameters
- **Transfer Learning** can match or exceed from-scratch training when the base model is strong
- **BatchNorm + Dropout together** (Model 6) beat either technique alone
- **SHAP analysis**: RM (rooms) and LSTAT (lower-status %) are the dominant predictors
- Small dataset (404 train) makes regularization critical — all models benefit from it
