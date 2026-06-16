# %% [markdown]
# # Visualization — Distributions and Training Curves

# %%
from evaluate import *
from utils import plot_loss_curves

# ── Y distribution ────────────────────────────────────────────────────────────
# %%
plt.subplots(figsize=(16, 5))
plt.title("Distribution of Median House Price (raw)")
sns.distplot(Y_train)
plt.show()

plt.subplots(figsize=(16, 5))
plt.title("Distribution of Median House Price (standardized)")
sns.distplot(scaled_train_y)
plt.show()

# ── Training curves — Model 1 (MSE) ─────────────────────────────────────────
# %%
plot_loss_curves(history1, EPOCHS, title="Model 1 — MSE Loss", convergence_epoch=30)

# ── Training curves — Model 2 (Huber) ────────────────────────────────────────
# %%
plot_loss_curves(history2, EPOCHS, title="Model 2 — Huber Loss", convergence_epoch=20)

# ── Training curves — Functional Wide & Deep ─────────────────────────────────
# %%
plot_loss_curves(history_fnn, EPOCHS, title="Functional Wide & Deep — Huber Loss")

# ── Training curves — He Normal ──────────────────────────────────────────────
# %%
N = EPOCHS
plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, N), history4.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history4.history["val_loss"], label="val_loss")
plt.axvline(x=20, color="red", label="convergence 1")
plt.axvline(x=55, color="green", label="convergence 2")
plt.title("Model 4 — He Normal Init")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()

# ── Training curves — BatchNorm ──────────────────────────────────────────────
# %%
plot_loss_curves(history5, EPOCHS, title="Model 5 — BatchNormalization")

# ── Training curves — BatchNorm + Dropout (best) ─────────────────────────────
# %%
plot_loss_curves(history6, EPOCHS, title="Model 6 — BatchNorm + Dropout (best standard)")

# ── Training curves — Transfer Learning ──────────────────────────────────────
# %%
plot_loss_curves(history_tl, EPOCHS, title="Transfer Learning — Frozen layers + new head")

# ── SHAP summary plot ─────────────────────────────────────────────────────────
# %%
feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                 "DIS", "RAD", "TAX", "PTRATIO", "B100", "LSTAT"]
data_interim = pd.DataFrame(scaled_train_x, columns=feature_names)
shap.summary_plot(shap_values[0], data_interim)
