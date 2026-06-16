# %% [markdown]
# # Data Preprocessing

# %%
from data_loader import *

# %%
# Drop identifier columns — no predictive value
data = loan.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# %%
# Split features and target
X = data.drop(labels=["Exited"], axis=1)
y = data["Exited"]

# %%
# Encode Geography using LabelEncoder then one-hot (to avoid ordinal assumption)
label_geo = LabelEncoder()
X["Geography"] = label_geo.fit_transform(X["Geography"])
X = pd.get_dummies(X, drop_first=True, columns=["Geography"])

# Encode Gender
label_gen = LabelEncoder()
X["Gender"] = label_gen.fit_transform(X["Gender"])

print("Features after encoding:", list(X.columns))
X.head()

# %%
# Stratified 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
print("Train:", X_train.shape, "  Test:", X_test.shape)
print("Churn rate:", y.mean().round(3))

# %%
# Standardize — fit on train only
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
