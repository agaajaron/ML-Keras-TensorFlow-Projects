# %% [markdown]
# # Model Training — Boston Housing Regression

# %%
from preprocessing import *
from model import (
    build_model_skeleton_mse, build_model_skeleton_huber,
    functional_nn,
    build_model_he_normal, build_model_batch, build_model_dropout,
    build_model_flexible,
)
from utils import plot_loss_curves, get_run_logdir, evaluate_r2
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

EPOCHS = 100

# ── Model 1: Skeleton MSE ────────────────────────────────────────────────────
# %%
model1 = build_model_skeleton_mse()
history1 = model1.fit(scaled_train_x, scaled_train_y, epochs=EPOCHS,
                      validation_split=0.3, verbose=0)
evaluate_r2(model1, scaler, scaled_test_x, scaled_test_y)

# ── Model 2: Skeleton Huber ──────────────────────────────────────────────────
# %%
model2 = build_model_skeleton_huber()
history2 = model2.fit(scaled_train_x, scaled_train_y, epochs=EPOCHS,
                      validation_split=0.3, verbose=0)
evaluate_r2(model2, scaler, scaled_test_x, scaled_test_y)

# ── Model 3: Functional Wide & Deep ─────────────────────────────────────────
# %%
model_fnn = functional_nn()
history_fnn = model_fnn.fit(scaled_train_x, scaled_train_y, epochs=EPOCHS,
                             validation_split=0.3, verbose=0)
evaluate_r2(model_fnn, scaler, scaled_test_x, scaled_test_y)
model_fnn.save("models/functional_nn.h5")

# ── Model 4: He Normal init ──────────────────────────────────────────────────
# %%
model4 = build_model_he_normal()
history4 = model4.fit(scaled_train_x, scaled_train_y, epochs=EPOCHS,
                      validation_split=0.3, verbose=0)
evaluate_r2(model4, scaler, scaled_test_x, scaled_test_y)

# ── Model 5: BatchNorm ───────────────────────────────────────────────────────
# %%
model5 = build_model_batch()
history5 = model5.fit(scaled_train_x, scaled_train_y, epochs=EPOCHS,
                      validation_split=0.3, verbose=0)
evaluate_r2(model5, scaler, scaled_test_x, scaled_test_y)

# ── Model 6: BatchNorm + Dropout (best standard model) ───────────────────────
# %%
model6 = build_model_dropout()
history6 = model6.fit(scaled_train_x, scaled_train_y, epochs=EPOCHS,
                      validation_split=0.3, verbose=0)
evaluate_r2(model6, scaler, scaled_test_x, scaled_test_y)

# ── TensorBoard + EarlyStopping example ─────────────────────────────────────
# %%
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_mse', min_delta=0.01, patience=10, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False,
)
model_tb = build_model_batch()
model_tb.fit(scaled_train_x, scaled_train_y, epochs=EPOCHS, validation_split=0.3,
             verbose=0, callbacks=[tensorboard_cb, early_stop])
evaluate_r2(model_tb, scaler, scaled_test_x, scaled_test_y)

# ── RandomizedSearchCV — hyperparameter tuning ───────────────────────────────
# %%
model_opt = KerasRegressor(build_fn=build_model_flexible, epochs=50, batch_size=32, verbose=0)

param_distribs = {
    "n_hidden_layer": [2, 3, 4, 5],
    "n_neurons":      [16, 32, 64],
    "active":         ['relu', 'selu'],
    "learning_rate":  reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(
    model_opt, param_distribs, n_iter=5, cv=3, scoring='neg_mean_squared_error'
)
rnd_search_cv.fit(scaled_train_x, scaled_train_y, epochs=50)
print("Best params:", rnd_search_cv.best_params_)

model_tuned = rnd_search_cv.best_estimator_.model
model_tuned.save("models/RCV_model.h5")
evaluate_r2(model_tuned, scaler, scaled_test_x, scaled_test_y)

# ── Transfer Learning — freeze layers, replace output head ──────────────────
# %%
model_A = keras.models.load_model("models/RCV_model.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="relu"))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(optimizer=keras.optimizers.RMSprop(lr=0.01),
                     loss=tf.keras.losses.Huber(), metrics=['mse'])

history_tl = model_B_on_A.fit(scaled_train_x, scaled_train_y,
                               epochs=EPOCHS, validation_split=0.3, verbose=0)
evaluate_r2(model_B_on_A, scaler, scaled_test_x, scaled_test_y)
