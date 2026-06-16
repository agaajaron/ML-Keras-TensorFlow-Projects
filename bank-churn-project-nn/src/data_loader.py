# %% [markdown]
# # Data Loading — Bank Customer Churn Dataset
#
# Source: Churn.csv (Google Drive)
# 10,000 rows × 14 columns
# Target: Exited (0 = stayed, 1 = left within 6 months)

# %%
from config import *

from google.colab import drive
drive.mount('/content/drive')

loan = pd.read_csv("/content/drive/MyDrive/Churn.csv")
data = loan.copy()

print("Shape:", data.shape)
print(data.dtypes)
data.describe()
