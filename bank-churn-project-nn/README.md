# Bank Churn Prediction — Neural Networks

Predict whether a bank customer will leave within the next 6 months using a Keras neural network classifier.

## Dataset

- **File**: `Churn.csv` (10,000 rows × 14 columns)
- **Target**: `Exited` — 0 = stayed, 1 = left (20.4% churn rate — imbalanced)
- **Features**: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

See [data/data.md](data/data.md) for full feature descriptions and EDA findings.

## Models

| Model | Architecture | Test Accuracy | Notes |
|---|---|---|---|
| Model 1 | input → 128 → 1 | 0.86 | Simple baseline, no overfitting |
| Model 2 | input → 128→64→64→32 → 1 | 0.80 | Overfits (train 0.99) |
| Model 3 | 64 → 32 → 1 (he_normal) | 0.84 | Slight overfitting |
| **Model 4** | **64 → Dropout(0.2) → 32 → Dropout(0.1) → 1** | **0.87** | **Best — L2 + LR decay + EarlyStopping** |
| Model 5a | 32→64→32→256→1 | — | Early stopping, rich metrics |
| Model 5b | 64→256→1 | 0.85 | Sigmoid mid-layer |
| Model 6 | 1024→1024→1024 + BN+Dropout | — | Large network, overfits |

Sklearn baselines (GBM, RF, KNN, LR) all score ~0.80 — Model 1 alone beats them.

## Project Structure

```
bank-churn-project-nn/
├── data/
│   └── data.md
├── models/
├── notebooks/
│   └── Bank_churn_project_NN.ipynb
├── src/
│   ├── config.py        # All imports
│   ├── data_loader.py   # Load Churn.csv from Google Drive
│   ├── preprocessing.py # Drop IDs, LabelEncoder, get_dummies, StandardScaler, split
│   ├── utils.py         # histogram_boxplot, labeled_barplot, stratified_cv
│   ├── model.py         # build_model1 … build_model6
│   ├── train.py         # Fit all models + sklearn comparison
│   ├── evaluate.py      # Accuracy, recall, F1, AUC-ROC per model
│   └── visualize.py     # EDA plots, training curves, ROC curves, confusion matrices
└── requirements.txt
```

## Key Findings

- Fewer dense layers generalize better on this tabular dataset
- **Dropout** has a significant regularization effect; BatchNorm less so
- Older customers (~50 yrs) and female customers churn more
- Class imbalance (20% churn) limits recall — SMOTE or class weighting recommended for next steps
