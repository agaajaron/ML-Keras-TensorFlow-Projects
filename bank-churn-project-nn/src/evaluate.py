# %% [markdown]
# # Model Evaluation

# %%
from train import *
from utils import threshold_predict

model_names = ["Model1", "Model3", "Model4 (best)", "Model5a", "Model5b", "Model6"]
nn_models   = [model1, model3, model4, model5a, model5b, model6]

# %%
for name, m in zip(model_names, nn_models):
    y_pred = threshold_predict(m, X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(f"{name:20s}  Acc={acc:.3f}  Recall={rec:.3f}  Prec={prec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

# %%
# Confusion matrix — best model (Model 4)
y_pred4 = threshold_predict(model4, X_test_sc)
cm4 = confusion_matrix(y_test, y_pred4)
print("\nModel 4 confusion matrix:\n", cm4)
print(classification_report(y_test, y_pred4, target_names=["Stayed (0)", "Left (1)"]))

# %%
# ROC-AUC comparison
r_probs = [0] * len(y_test)
r_auc = roc_auc_score(y_test, r_probs)

for name, m in zip(model_names, nn_models):
    y_p = threshold_predict(m, X_test_sc)
    auc = roc_auc_score(y_test, y_p)
    print(f"{name:20s}  AUC={auc:.3f}  (random={r_auc:.3f})")
