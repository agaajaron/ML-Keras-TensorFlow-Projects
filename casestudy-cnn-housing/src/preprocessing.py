# %% [markdown]
# # Data Preprocessing — Standardization
#
# For regression NNs, both X and Y are standardized to speed convergence.
# (Do NOT standardize Y for classification tasks.)

# %%
from data_loader import *

# %%
scaler = StandardScaler()

scaled_train_x = scaler.fit_transform(X_train)
scaled_test_x = scaler.transform(X_test)

scaled_train_y = scaler.fit_transform(Y_train.reshape(-1, 1))
scaled_test_y = scaler.transform(Y_test.reshape(-1, 1))

print("X_train scaled:", scaled_train_x.shape)
print("Y_train scaled:", scaled_train_y.shape)
