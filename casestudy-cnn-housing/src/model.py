# %% [markdown]
# # Model Definitions — Boston Housing Regression
#
# Six ANN architectures + one functional API model:
#
#  Model 1: build_model_Skeleton (MSE)
#    Dense(16,relu,Glorot) → Dense(8,relu,Glorot) → Dense(1,relu)
#    Optimizer: RMSprop(0.001), Loss: MSE
#    Converges ~epoch 30; R² ≈ 0.70
#
#  Model 2: build_model_Skeleton (Huber)
#    Same architecture, Loss: Huber() — smoother gradient, better convergence
#    Converges ~epoch 20; R² slightly better
#
#  Model 3: functional_NN (Wide & Deep / residual)
#    Input → Dense(30,relu) → Dense(30,relu) → concatenate(input, hidden2) → Dense(1)
#    Skip connection from raw input to output; Huber loss
#
#  Model 4: build_model_HE_Normal
#    Dense(16,relu,he_normal) → Dense(8,relu,he_normal) → Dense(1,relu,he_normal)
#    Huber loss; vanishing gradient partially solved by He init
#    Two convergence points; some overfitting
#
#  Model 5: build_model_Batch
#    Dense(16,relu,he_normal) → BatchNorm → Dense(8,relu,he_normal) → Dense(1,relu,he_normal)
#    BatchNorm reduces vanishing gradients → smoother learning curve
#
#  Model 6: build_model_dropout ← Best standard model
#    Dense(16,relu,he_normal) → BatchNorm → Dropout(0.03)
#    → Dense(8,relu,he_normal) → Dropout(0.01) → Dense(1,relu,he_normal)
#    BatchNorm + Dropout together give best generalization
#
#  Model 7: build_model (flexible, for RandomizedSearchCV)
#    Parameterized: n_hidden_layer, n_neurons, active, learning_rate
#    L1 regularization; Pyramid neuron reduction each layer
#    Wrapped in KerasRegressor for sklearn compatibility

# %%
from preprocessing import *

initializer_glorot = tf.keras.initializers.GlorotNormal()


def build_model_skeleton_mse():
    """Dense(16,relu,Glorot) → Dense(8,relu,Glorot) → Dense(1,relu). Loss=MSE."""
    model = Sequential()
    model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu',
                    kernel_initializer=initializer_glorot))
    model.add(Dense(8, activation='relu', kernel_initializer=initializer_glorot))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(0.001), metrics=['mse'])
    return model


def build_model_skeleton_huber():
    """Same architecture as skeleton but with Huber loss — better convergence."""
    model = Sequential()
    model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu',
                    kernel_initializer=initializer_glorot))
    model.add(Dense(8, activation='relu', kernel_initializer=initializer_glorot))
    model.add(Dense(1, activation='relu'))
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=keras.optimizers.RMSprop(0.001), metrics=['mse'])
    return model


def functional_nn():
    """
    Wide & Deep / residual architecture (Functional API):
    Input → Dense(30,relu) → Dense(30,relu) → concatenate(input, out) → Dense(1)
    Skip connection passes raw features directly to output layer.
    """
    inp = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(inp)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([inp, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.models.Model(inputs=[inp], outputs=[output])
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=keras.optimizers.RMSprop(0.001), metrics=['mse'])
    return model


def build_model_he_normal():
    """Dense(16→8→1) all he_normal init + Huber. He init reduces vanishing gradients."""
    model = Sequential()
    model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu',
                    kernel_initializer='he_normal'))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='relu', kernel_initializer='he_normal'))
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=keras.optimizers.RMSprop(0.001), metrics=['mse'])
    return model


def build_model_batch():
    """Dense(16,he) → BatchNorm → Dense(8,he) → Dense(1,he). Smoother learning curves."""
    model = Sequential()
    model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu',
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='relu', kernel_initializer='he_normal'))
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=keras.optimizers.RMSprop(0.001), metrics=['mse'])
    return model


def build_model_dropout():
    """
    Best standard model:
    Dense(16,he) → BatchNorm → Dropout(0.03) → Dense(8,he) → Dropout(0.01) → Dense(1,he)
    BatchNorm + Dropout together generalize best on Boston Housing.
    """
    model = Sequential()
    model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu',
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.03))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.01))
    model.add(Dense(1, activation='relu', kernel_initializer='he_normal'))
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=keras.optimizers.RMSprop(0.001), metrics=['mse'])
    return model


def build_model_flexible(n_hidden_layer=1, n_neurons=10, input_shape=None,
                          learning_rate=0.01, active="relu", drop=0.01):
    """
    Flexible parameterized model for RandomizedSearchCV.
    - n_hidden_layer: number of hidden layers
    - n_neurons: neurons in first hidden layer (halved each subsequent layer)
    - active: activation function
    - learning_rate: for RMSprop
    - L1 regularization applied to all hidden layers
    - Dropout added when n_hidden_layer is even
    """
    if input_shape is None:
        input_shape = X_train.shape[1]
    model = Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    n = n_neurons
    for _ in range(n_hidden_layer):
        model.add(Dense(n, activation=active,
                        kernel_regularizer=tf.keras.regularizers.L1(0.01)))
        if n_hidden_layer % 2 == 0:
            model.add(Dropout(drop))
        n = int(round(n * 0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  loss=tf.keras.losses.Huber(), metrics=['mse'])
    return model
