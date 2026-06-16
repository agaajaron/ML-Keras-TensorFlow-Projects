# Dataset: Bank Customer Churn (Churn.csv)

## Source
- **File**: `Churn.csv` (loaded from Google Drive)
- **Context**: Bank churn prediction — identify customers likely to leave within 6 months

## Overview
| Property | Value |
|---|---|
| Rows | 10,000 |
| Columns | 14 (12 features after preprocessing) |
| Task | Binary classification (churn prediction) |
| Class balance | ~79.6% stayed (0), ~20.4% left (1) — imbalanced |

## Features
| Column | Type | Description |
|---|---|---|
| RowNumber | ID | Dropped (row index, no value) |
| CustomerId | ID | Dropped (surrogate key) |
| Surname | Text | Dropped (not predictive) |
| CreditScore | Numeric | Customer credit history score (350–850) |
| Geography | Categorical | Country: France (50%), Germany (25%), Spain (25%) |
| Gender | Categorical | Male / Female |
| Age | Numeric | Customer age in years (18–92); older customers churn more |
| Tenure | Numeric | Years with bank (0–10) |
| Balance | Numeric | Account balance; ~3,500 customers have zero |
| NumOfProducts | Numeric | Number of bank products held (1–4) |
| HasCrCard | Binary | Has credit card (0/1) |
| IsActiveMember | Binary | Uses services regularly (0/1) |
| EstimatedSalary | Numeric | Annual salary estimate |
| **Exited** | **Target** | **0 = stayed, 1 = left within 6 months** |

## Preprocessing Applied
- Dropped: RowNumber, CustomerId, Surname
- `Geography` → LabelEncoder → `pd.get_dummies(drop_first=True)`
- `Gender` → LabelEncoder (Male=1, Female=0)
- 80/20 stratified train-test split (`random_state=0`)
- `StandardScaler` fit on train, transform both sets

## Key EDA Findings
- **56% of churned customers are female** (vs 45% in overall dataset)
- **Older customers (≈50 yrs) churn more** — visible in age distribution
- **Customers with 3–4 products** tend to churn at higher rates
- Zero balance group (~3,500 customers) shows no unusual churn pattern
- No strong pairwise correlations (max: NumOfProducts ↔ Balance)

## Models Trained

### Model 1: Simple (input→128→1) — `build_model1`
- `Sequential: Dense(relu) → Dense(128,relu) → Dense(1,sigmoid)`
- Adam optimizer, binary_crossentropy, 100 epochs, batch=300
- Train acc: 0.86 | Test acc: 0.86 — **no overfitting, beats sklearn baselines**

### Model 2: Deep (input→128→64→64→32→1) — `build_model2`
- 5-layer deep network with he_normal init, 300 epochs
- Train acc: 0.99 | Test acc: 0.80 — **severe overfitting**

### Model 3: Medium (64→32→1) — `build_model3`
- `build_model()` with he_normal, validation_split=0.2, 100 epochs
- Train acc: 0.91 | Test acc: 0.84 — moderate overfitting

### Model 4: Dropout + L2 + LR decay — `build_model4_dropout` ← **Best Model**
- `Dense(64,L2) → Dropout(0.2) → Dense(32,L2) → Dropout(0.1) → Dense(1,sigmoid)`
- InverseTimeDecay learning rate + EarlyStopping(patience=70, monitor=val_accuracy)
- Train ≈ Test ≈ 0.86–0.87 — minimal overfitting, best generalization

### Model 5a: (32→64→32→256→1) — `build_model5a`
- Added sigmoid hidden layer; rich METRICS (TP/FP/TN/FN/AUC/Recall)
- EarlyStopping(patience=10); still overfits

### Model 5b: (64→256→1) — `build_model5b`
- Smaller with sigmoid mid-layer; Train 0.88 / Test 0.85

### Model 6: (1024→1024→1024) BN+Dropout — `build_model6`
- Large network; BatchNorm + Dropout(0.3) × 2
- Still overfits — batch norm alone doesn't outperform Model 4

## Sklearn Baselines (via Stratified K-Fold CV)
| Model | Accuracy |
|---|---|
| Gradient Boosting | ~0.80 |
| Random Forest | ~0.80 |
| KNN (k=11) | ~0.80 |
| Logistic Regression | ~0.80 |

**Conclusion**: Model 1 (simple NN, 0.86) beats all sklearn classifiers (~0.80) without tuning.

## Conclusions
- **Best model**: Model 4 (Dropout + L2 + LR decay) — 0.86–0.87 on both train and test
- **Key insight**: Fewer hidden layers work better for this tabular data
- **Dropout** has significant regularization effect; Batch Normalization less so
- Class imbalance (20% churn) is the main limiting factor — SMOTE or class weights could further improve recall
