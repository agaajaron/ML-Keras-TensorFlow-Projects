# %%
import warnings
warnings.filterwarnings("ignore")

import os
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as plt2
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import boston_housing

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.base import clone

import shap
from scipy.stats import reciprocal

# Reproducibility
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tf.random.set_seed(1234)
np.random.seed(1234)
rn.seed(1254)

sns.set(color_codes=True)
