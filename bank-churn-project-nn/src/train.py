# %% [markdown]
# # Model Training

# %%
from preprocessing import *
from model import (
    build_model1, build_model2, build_model3,
    build_model4_dropout, get_callbacks_model4,
    build_model5a, build_model5b, build_model6,
)
from utils import stratified_cv

input_dim = X_train_sc.shape[1]

# ── Model 1: simple (input → 128 → 1) ───────────────────────────────────────
# %%
model1 = build_model1(input_dim)
model1.fit(X_train_sc, y_train.to_numpy(), batch_size=300, epochs=100, verbose=1)
print("Model 1 trained. Test eval:")
model1.evaluate(X_test_sc, y_test.to_numpy())

# ── Model 2: deep overfitting (128→64→64→32→1) ──────────────────────────────
# %%
model2 = build_model2(input_dim)
model2.fit(X_train_sc, y_train.to_numpy(), batch_size=300, epochs=300, verbose=0)
print("Model 2 trained (expect overfitting). Test eval:")
model2.evaluate(X_test_sc, y_test.to_numpy())

# ── Model 3: (64→32→1) he_normal ────────────────────────────────────────────
# %%
model3 = build_model3(input_dim)
history3 = model3.fit(X_train_sc, y_train, epochs=100, validation_split=0.2, verbose=1)
print("Model 3 trained. Test eval:")
model3.evaluate(X_test_sc, y_test)

# ── Model 4: Dropout + L2 + LR decay — best model ───────────────────────────
# %%
model4 = build_model4_dropout(X_train_sc.shape[0])
history4 = model4.fit(
    X_train_sc, y_train,
    validation_data=(X_test_sc, y_test),
    epochs=150, batch_size=32,
    callbacks=get_callbacks_model4(), verbose=0,
)
print("Model 4 trained (best model). Test eval:")
model4.evaluate(X_test_sc, y_test)

# ── Model 5a: (32→64→32→256→1) early stopping ───────────────────────────────
# %%
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", verbose=0, patience=10, mode="max", restore_best_weights=True
)
model5a = build_model5a(input_dim)
history5a = model5a.fit(
    X_train_sc, y_train, epochs=50, batch_size=200,
    validation_split=0.30, shuffle=True, callbacks=[early_stop],
)

# ── Model 5b: (64→256→1) sigmoid ────────────────────────────────────────────
# %%
model5b = build_model5b()
history5b = model5b.fit(
    X_train_sc, y_train, epochs=100, batch_size=200,
    validation_split=0.30, shuffle=True, callbacks=[early_stop],
)

# ── Model 6: (1024→1024→1024) BN+Dropout ────────────────────────────────────
# %%
model6 = build_model6()
history6 = model6.fit(
    X_train_sc, y_train, epochs=100, batch_size=200,
    validation_split=0.30, shuffle=True, callbacks=[early_stop],
)

# ── Sklearn baselines via stratified cross-validation ───────────────────────
# %%
print("Gradient Boosting:  {:.2f}".format(
    metrics.accuracy_score(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print("Random Forest:      {:.2f}".format(
    metrics.accuracy_score(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print("KNN (k=11):         {:.2f}".format(
    metrics.accuracy_score(y, stratified_cv(X, y, neighbors.KNeighborsClassifier, n_neighbors=11))))
print("Logistic Regression:{:.2f}".format(
    metrics.accuracy_score(y, stratified_cv(X, y, linear_model.LogisticRegression))))
