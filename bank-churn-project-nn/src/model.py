# %% [markdown]
# # Model Definitions — Bank Churn Neural Networks
#
# Six neural network architectures explored:
#
#  Model 1: (input → 128 → 1)
#    Simple baseline; 100 epochs, batch=300 — 0.86 accuracy, no overfitting
#
#  Model 2: (input → 128→64→64→32 → 1) he_normal
#    Deeper network; 300 epochs — heavy overfitting (0.99 train / 0.80 test)
#
#  Model 3: build_model() — (64 → 32 → 1) he_normal
#    Moderate depth; validation_split=0.2 — slight overfitting
#
#  Model 4: nn_model — (64 → 32 → 1) + L2 + Dropout + InverseTimeDecay
#    Best model: minimal overfitting, 0.86–0.87 on both train and test
#    EarlyStopping(patience=70, monitor=val_accuracy)
#
#  Model 5a: model4 — (32→64→32→256→1) + early stopping + rich METRICS
#    Overfits; early stopping doesn't fully prevent it
#
#  Model 5b: model5 — (64→256→1) sigmoid layers + early stopping
#    Train 0.88 / Test 0.85
#
#  Model 6: (1024→1024→1024) + BatchNorm + Dropout(0.3)
#    Large network; overfits despite regularization
#
# Conclusion: Model 4 (Dropout + L2 + LR decay) is the best performing.
# All sklearn classifiers (GBM, RF, KNN, LR) score ~0.8 — NN Model 1 beats them.

# %%
from preprocessing import *

METRICS_FULL = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),
]


def build_model1(input_dim):
    """Simple 3-layer network: input → 128 relu → 1 sigmoid. Baseline."""
    m = Sequential()
    m.add(Dense(input_dim, activation="relu", input_dim=input_dim))
    m.add(Dense(128, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_model2(input_dim):
    """Deeper: input → 128→64→64→32 → 1, he_normal init. Overfits at 300 epochs."""
    m = Sequential()
    m.add(Dense(input_dim, activation="relu", input_dim=input_dim))
    m.add(Dense(128, activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(64,  activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(64,  activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(32,  activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(1,   activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_model3(input_shape):
    """(64 → 32 → 1) he_normal. Slight overfitting."""
    m = Sequential()
    m.add(Dense(64, input_shape=(input_shape,), activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(32, activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(1,  activation="sigmoid"))
    m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m


def build_model4_dropout(input_shape):
    """
    Best model: (64 → Dropout(0.2) → 32 → Dropout(0.1) → 1)
    L2 regularization + InverseTimeDecay LR schedule + EarlyStopping(patience=70)
    Train≈Test≈0.86–0.87 accuracy.
    """
    m = Sequential()
    m.add(Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                activation="relu", kernel_initializer="he_normal"))
    m.add(Dropout(0.2))
    m.add(Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                activation="relu", kernel_initializer="he_normal"))
    m.add(Dropout(0.1))
    m.add(Dense(1, activation="sigmoid"))

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=(input_shape / 32) * 50,
        decay_rate=1,
        staircase=False,
    )
    m.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(lr_schedule),
              metrics=["accuracy"])
    return m


def get_callbacks_model4():
    return [tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=70, restore_best_weights=True
    )]


def build_model5a(input_shape):
    """(32→64→32→256→1) with early stopping and METRICS. Overfits."""
    m = Sequential()
    m.add(Dense(32, activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(64, input_shape=(input_shape,), activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(32, activation="relu", kernel_initializer="he_normal"))
    m.add(Dense(256, activation="sigmoid"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS_FULL)
    return m


def build_model5b():
    """(64→256→1) sigmoid mid-layer. Train 0.88 / Test 0.85."""
    m = Sequential()
    m.add(Dense(64, activation="relu", use_bias=True,
                bias_initializer="zeros", kernel_initializer="he_normal"))
    m.add(Dense(256, activation="sigmoid"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS_FULL)
    return m


def build_model6():
    """(1024→BN→1024→Dropout→BN→1024→Dropout→BN→1) Large deep network. Overfits."""
    m = keras.Sequential()
    m.add(Dense(1024, activation="relu", kernel_initializer="he_normal"))
    m.add(BatchNormalization())
    m.add(Dense(1024, activation="relu", kernel_initializer="he_normal"))
    m.add(Dropout(0.3))
    m.add(BatchNormalization())
    m.add(Dense(1024, activation="relu", kernel_initializer="he_normal"))
    m.add(Dropout(0.3))
    m.add(BatchNormalization())
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS_FULL)
    return m
