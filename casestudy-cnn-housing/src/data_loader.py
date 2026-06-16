# %% [markdown]
# # Data Loading — Boston Housing Dataset
#
# Source: tensorflow.keras.datasets.boston_housing
# 404 training + 102 test samples, 13 features
# Target: MEDV — median house value in $1,000s (regression)

# %%
from config import *

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)

print("Train size:", X_train.shape[0], " | Features:", X_train.shape[1])
print("Test size: ", X_test.shape[0])
print("Y_train shape:", Y_train.shape)
print("Sample row:", X_train[0])
