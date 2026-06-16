# %% [markdown]
# # Visualization — EDA and Training Curves

# %%
from evaluate import *
from utils import histogram_boxplot, labeled_barplot

# ── EDA — numeric distributions ──────────────────────────────────────────────
# %%
for col in ["CreditScore", "Age", "EstimatedSalary", "Balance"]:
    histogram_boxplot(loan, col)

# ── EDA — categorical barplots ───────────────────────────────────────────────
# %%
for col in ["Geography", "Gender", "Exited", "NumOfProducts", "HasCrCard", "IsActiveMember"]:
    labeled_barplot(loan, col, perc=True)

# ── EDA — bivariate ──────────────────────────────────────────────────────────
# %%
sns.displot(loan, x="CreditScore", hue="Exited", bins=30, palette="winter", kde=True, stat="probability")
plt.title("Credit Score by Churn")
plt.show()

sns.displot(loan, x="Age", hue="Exited", bins=20, palette="winter", kde=True)
plt.title("Age by Churn")
plt.show()

sns.boxplot(x="NumOfProducts", y="EstimatedSalary", data=loan)
plt.title("Salary vs Number of Products")
plt.show()

sns.boxplot(x="Exited", y="EstimatedSalary", data=loan)
plt.title("Salary: Stayed vs Exited")
plt.show()

# ── EDA — correlation heatmap ────────────────────────────────────────────────
# %%
plt.figure(figsize=(22, 22))
sns.heatmap(data.corr(), annot=True, cmap="Spectral")
plt.title("Correlation Matrix")
plt.show()

# ── EDA — pairplot ───────────────────────────────────────────────────────────
# %%
sns.pairplot(data, hue="Exited")
plt.show()

# ── Training curves — Model 3 ────────────────────────────────────────────────
# %%
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, 100), history3.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), history3.history["val_loss"], label="val_loss")
plt.title("Model 3 — Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()

plt.plot(history3.history["accuracy"])
plt.plot(history3.history["val_accuracy"])
plt.title("Model 3 — Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(["train", "val"]); plt.show()

# ── Training curves — Model 4 (best) ─────────────────────────────────────────
# %%
plt.plot(history4.history["accuracy"])
plt.plot(history4.history["val_accuracy"])
plt.title("Model 4 — Accuracy (Dropout + L2 + LR decay)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(["train", "val"]); plt.show()

# ── ROC curves ────────────────────────────────────────────────────────────────
# %%
r_probs = [0] * len(y_test)
r_auc = roc_auc_score(y_test, r_probs)
r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)

plt.figure(figsize=(10, 8))
plt.plot(r_fpr, r_tpr, linestyle="--", label=f"Random (AUC={r_auc:.3f})")

for name, m in zip(["Model1", "Model4"], [model1, model4]):
    y_p = threshold_predict(m, X_test_sc)
    fpr, tpr, _ = roc_curve(y_test, y_p)
    auc = roc_auc_score(y_test, y_p)
    plt.plot(fpr, tpr, marker=".", label=f"{name} (AUC={auc:.3f})")

plt.title("ROC Curves"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.show()

# ── Confusion matrix — Model 4 ────────────────────────────────────────────────
# %%
y_pred4 = threshold_predict(model4, X_test_sc)
cm = confusion_matrix(y_test, y_pred4)
labels = np.array([[f"{v}\n{v/cm.sum():.1%}" for v in row] for row in cm])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=labels, fmt="", xticklabels=["Stayed", "Left"],
            yticklabels=["Stayed", "Left"])
plt.title("Model 4 — Confusion Matrix"); plt.show()
