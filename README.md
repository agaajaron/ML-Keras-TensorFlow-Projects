# ML-Keras-TensorFlow-Projects

Neural network projects built with Keras and TensorFlow, covering binary classification and regression tasks on tabular data.

## Projects

| Folder | Task | Dataset | Best Model |
|---|---|---|---|
| [bank-churn-project-nn](bank-churn-project-nn/) | Binary classification | Churn.csv (10K customers) | Dropout + L2 + LR decay NN — 0.87 accuracy |
| [casestudy-cnn-housing](casestudy-cnn-housing/) | Regression | Boston Housing (keras built-in) | RandomizedSearchCV tuned ANN — best R² |

## Project Structure

Each project follows a standard ML layout:

```
project-name/
├── data/
│   ├── data.md          # Dataset description, features, model summary
│   └── .gitkeep
├── models/
│   └── .gitkeep         # Saved .h5 model files
├── notebooks/           # Original Jupyter notebooks
├── src/
│   ├── config.py        # Imports, seeds, global settings
│   ├── data_loader.py   # Load raw data
│   ├── preprocessing.py # Encoding, scaling, train/test split
│   ├── utils.py         # Reusable helper functions
│   ├── model.py         # Model builder functions
│   ├── train.py         # Training and hyperparameter tuning
│   ├── evaluate.py      # Metrics and evaluation
│   └── visualize.py     # EDA plots and training curves
└── requirements.txt
```

## Key Techniques

- Sequential and Functional Keras API
- Regularization: Dropout, L2, BatchNormalization
- Learning rate schedules: `InverseTimeDecay`
- Callbacks: `EarlyStopping`, `TensorBoard`, `ReduceLROnPlateau`
- Hyperparameter tuning: `RandomizedSearchCV` with `KerasRegressor`
- Transfer Learning: freeze pretrained layers, replace output head
- Explainability: SHAP `KernelExplainer` and `DeepExplainer`
- Weight initialization: Glorot Normal, He Normal
