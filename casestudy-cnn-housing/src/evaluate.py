# %% [markdown]
# # Model Evaluation — R² Scores and SHAP Explainability

# %%
from train import *
from utils import evaluate_r2
import shap

model_names_h = [
    "Skeleton MSE", "Skeleton Huber", "Wide & Deep (FNN)",
    "He Normal", "BatchNorm", "BatchNorm+Dropout",
    "TensorBoard+Early", "RandomizedSearchCV", "Transfer Learning",
]
models_h = [model1, model2, model_fnn, model4, model5, model6, model_tb, model_tuned, model_B_on_A]

# %%
print("R² scores on test set:")
for name, m in zip(model_names_h, models_h):
    preds = m.predict(scaled_test_x)
    y_pred = scaler.inverse_transform(preds.reshape(-1, 1))
    y_true = scaler.inverse_transform(scaled_test_y.reshape(-1, 1))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:30s}  R² = {r2:.3f}")

# %%
# SHAP explainability — best model (RandomizedSearchCV tuned)
feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                 "DIS", "RAD", "TAX", "PTRATIO", "B100", "LSTAT"]

df_train_summary = shap.kmeans(scaled_train_x, 25)
explainer = shap.KernelExplainer(model_tuned.predict, df_train_summary)
shap_values = explainer.shap_values(scaled_train_x)

data_interim = pd.DataFrame(scaled_train_x, columns=feature_names)
shap.summary_plot(shap_values[0], data_interim)
