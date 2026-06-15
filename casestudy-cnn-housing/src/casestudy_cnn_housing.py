# %% [markdown]
# # Boston housing price regression dataset
# 
# Dataset taken from the StatLib library which is maintained at Carnegie Mellon University.
# 
# Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s. Targets are the median values of the houses at a location (in k$).
# 
# 
# ## Variables in order:
# 
#  **CRIM**     per capita crime rate by town
# 
#  **ZN**       proportion of residential land zoned for lots over 25,000 sq.ft.
# 
#  **INDUS**    proportion of non-retail business acres per town
# 
#  **CHAS**     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
#  **NOX**      nitric oxides concentration (parts per 10 million)
# 
#  **RM**       average number of rooms per dwelling
# 
#  **AGE**      proportion of owner-occupied units built prior to 1940
# 
#  **DIS**      weighted distances to five Boston employment centres
# 
#  **RAD**      index of accessibility to radial highways
# 
#  **TAX**      full-value property-tax rate per $10,000
# 
#  **PTRATIO**  pupil-teacher ratio by town
# 
#  **B**        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 
#  **LSTAT**    % lower status of the population
# 
#  **MEDV**     Median value of owner-occupied homes in $1000's

# %%
import tensorflow
tensorflow.__version__

# %%
import random as rn

import numpy as np

import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
PYTHONHASHSEED=0

tf.random.set_seed(1234)
np.random.seed(1234)
rn.seed(1254)

# %%
# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ### Import dataset
# - This dataset can be imported
# - High level API Keras has some datasets available
# - You can look at all the datasets available here https://keras.io/datasets/
# 

# %%
from tensorflow.keras.datasets import boston_housing

# boston_housing.load_data() function returns 2 tuples, one for train data and 
# other for test data. We will take only train data here.

(X_train, Y_train),(X_test, Y_test)  = boston_housing.load_data(path="boston_housing.npz", test_split=0.2, seed=113)

# %% [markdown]
# ### Getting details of dataset
# - We will see how many rows are there in the data
# - We will check how many features are there

# %%
print('Number of examples in train set: ', X_train.shape[0])
print('Number of features for each example: ', X_train.shape[1])
print('Shape of actual prices in training data: ', Y_train.shape)

# %%
print('Number of examples in test set: ', X_test.shape[0])
print('Shape of actual prices in test data: ', Y_test.shape)

# %% [markdown]
# Let's see some values of features and labels from the dataset

# %%
X_train[0]

# %% [markdown]
# ### Distribution of Dependent Variable

# %%
import matplotlib.pyplot as plt
# %matplotlib inline #Making plots visible in the Jupyter Notebook

import seaborn as sns
sns.set(color_codes=True)

# %%
plt.subplots(figsize=(16,5));
plt.title("Distribution of Median Price of House");
sns.distplot(Y_train);

# %% [markdown]
# ### We must normalize our data for neural networks to perform optimally
# Convergence is faster for normalized data.

# %%
"""
1. Regression Based Task - We can standardize both dependent and independent variables. We has a faster/quicker convergence. If we you are batchnormalization, make sure not standardize the data.

Batch normalization has a parameter called momentum. 


2. Classification - Avoid standardizing the dependent variable.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Scale both the training inputs and outputs
scaled_train_x = scaler.fit_transform(X_train)
scaled_test_x = scaler.transform(X_test)

scaled_train_y = scaler.fit_transform(Y_train.reshape(-1, 1))
scaled_test_y = scaler.transform(Y_test.reshape(-1, 1))

# %%
plt.subplots(figsize=(16,5));
plt.title("Distribution of Median Price of House");
sns.distplot(scaled_train_y);

# %%
# scaled_train_y=scaled_train_y.reshape(-1)
# scaled_train_y.shape

# %% [markdown]
# ### Build the model
# 

# %%
# !pip install tensorflow_addons

# import tensorflow_addons as tfa

import tensorflow as tf

# r2_metric = tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(1,))

# %% [markdown]
# ### Artifical Neural Network

# %%
from tensorflow import keras # importing keras library
from tensorflow.keras.models import Sequential  # importing the Sequential Model
from tensorflow.keras.layers import Dense       # importing Dense layer

initializer = tf.keras.initializers.GlorotNormal()

def build_model_Skeleton():
    
    ## Initializing the ANN
    model = Sequential() 
    
    # This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
    input_layer = Dense(16, input_shape=(X_train.shape[1],),activation='relu', kernel_initializer=initializer)
    model.add(input_layer) # 
    
    #Adding the hidden layer
    # Notice that we do not need to specify input dim. 
    hidden_layer = Dense(8, activation='relu', kernel_initializer=initializer); 
    model.add(hidden_layer) 
    
    #Adding the output layer
    # Notice that we do not need to specify input dim. 
    # we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
    # We use the sigmoid because we want probability outcomes
    
    output_layer = Dense(1,activation='relu') 

    model.add(output_layer)


    optimizer = keras.optimizers.RMSprop(0.001)  # Defining the optimizer #0.01 #0.001 #0.0001
    
    #Loss function 
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics='mse')  # Defining the loss function, optimizer and metrices 
    return model

# %%
#Getting the model summary. We are uisng standalone keras to build our model
model = build_model_Skeleton()
model.summary()

# %%
#Defining the number of epochs
EPOCHS = 100

#fitting the model
history = model.fit(scaled_train_x , scaled_train_y, epochs=EPOCHS, validation_split = 0.3, verbose=0)

# %%
import pylab as plt

import numpy as np

N = 100 #Similar to Size of Epoch

plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.axvline(x=30, color="red", label="convergence");

plt.title("Training Loss and Validation loss on the dataset using MSE")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="middle")
plt.show()

# %%
#Lets Print the predicted prices 

test_predictionsk=model.predict(scaled_test_x)

y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

from sklearn.metrics import r2_score
print("R2:", round(r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1),2))

# %%
from tensorflow import keras # importing keras library
from tensorflow.keras.models import Sequential  # importing the Sequential Model
from tensorflow.keras.layers import Dense       # importing Dense layer

initializer = tf.keras.initializers.GlorotNormal()

def build_model_Skeleton():
    
    ## Initializing the ANN
    model = Sequential() 
    
    # This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
    input_layer = Dense(16, input_shape=(X_train.shape[1],),activation='relu', kernel_initializer=initializer)
    model.add(input_layer) # 
    
    #Adding the hidden layer
    # Notice that we do not need to specify input dim. 
    hidden_layer = Dense(8, activation='relu', kernel_initializer=initializer); 
    model.add(hidden_layer) 
    
    #Adding the output layer
    # Notice that we do not need to specify input dim. 
    # we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
    # We use the sigmoid because we want probability outcomes
    
    output_layer = Dense(1,activation='relu') 

    model.add(output_layer)


    optimizer = keras.optimizers.RMSprop(0.001)  # Defining the optimizer #0.01 #0.001 #0.0001
    
    #Loss function 
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics='mse')  # Defining the loss function, optimizer and metrices 
    return model

# %% [markdown]
# Build the model and view the summary

# %%
#Getting the model summary. We are uisng standalone keras to build our model
model = build_model_Skeleton()
model.summary()

# %% [markdown]
# Letâ€™s now train the model for 100 epochs, and record the training and validation accuracy in â€˜historyâ€™.

# %%
#Definign the number of epochs
EPOCHS = 100

#fitting the model
history = model.fit(scaled_train_x , scaled_train_y, epochs=EPOCHS, validation_split = 0.3, verbose=0)

# %% [markdown]
# Let's plot the  validation and training loss

# %%
import pylab as plt

import numpy as np

N = 100 #Similar to Size of Epoch

plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.axvline(x=20, color="red", label="convergence");

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="middle")
plt.show()

# %%
model.evaluate(scaled_test_x , scaled_test_y)

# %%
#Lets Print the predicted prices 

test_predictionsk=model.predict(scaled_test_x)

y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

# %%
from sklearn.metrics import r2_score
print("R2:", round(r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1),2))

# %%
def functional_NN():

  input = keras.layers.Input(shape=X_train.shape[1:])

  hidden1 = keras.layers.Dense(30, activation="relu")(input)
  
  hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
  
  concat = keras.layers.concatenate([input, hidden2])
  
  output = keras.layers.Dense(1)(concat)
  
  model = keras.models.Model(inputs=[input], outputs=[output])

  optimizer = keras.optimizers.RMSprop(0.001)  # Defining the optimizer #0.01 #0.001 #0.0001
    
  #Loss function 
  model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics='mse')  # Defining the loss function, optimizer and metrices 
  return model

# %%
#Getting the model summary. We are uisng standalone keras to build our model
model_FNN = functional_NN()
model_FNN.summary()

# %%
#Definign the number of epochs
EPOCHS = 100

#fitting the model
history1 = model_FNN.fit(scaled_train_x , scaled_train_y, epochs=EPOCHS, validation_split = 0.3, verbose=0)

# %%
import pylab as plt

import numpy as np

N = 100 #Similar to Size of Epoch

plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history1.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history1.history["val_loss"], label="val_loss")

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="middle")
plt.show()

# %%
#Lets Print the predicted prices 

test_predictionsk=model_FNN.predict(scaled_test_x)

y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

from sklearn.metrics import r2_score
print("R2:", round(r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1),2))

# %%
model.save("my_keras_model.h5")

model = keras.models.load_model("my_keras_model.h5")

# %% [markdown]
# # Model-2 
# 
# Let's do the weight initialization.
# 
# he_normal. It draws samples from a truncated normal distribution centered on $0$ with $stddev$ = $\sqrt {\dfrac{2}{fan_{in}}}$ where $fan_{in}$ is the number of input units in the weight tensor.

# %%
def build_model_HE_Normal():
  ## Initializing the ANN
    

    model = Sequential() 
    # This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
    input_layer = Dense(16, input_shape=(X_train.shape[1],),activation='relu',kernel_initializer='he_normal')
    
    model.add(input_layer) # 
    #Adding the hidden layer
    # Notice that we do not need to specify input dim. 
    hidden_layer = Dense(8, activation='relu',kernel_initializer='he_normal'); # defining the weight initialiazer 
    
    model.add(hidden_layer) 
    #Adding the output layer
    # Notice that we do not need to specify input dim. 
    # we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
    # We use the sigmoid because we want probability outcomes
    
    output_layer = Dense(1,activation='relu',kernel_initializer='he_normal') 

    model.add(output_layer)


    optimizer = keras.optimizers.RMSprop(0.001)  # Defining the optimizer 
    #Loss function 
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=[ 'mse'])  # Defining the loss function, optimizer and metrices 
    return model

# %%
#Getting the model summary. We are uisng standalone keras to build our model
model2 = build_model_HE_Normal()
model2.summary()

# %%
#Definign the number of epochs
import numpy as np
EPOCHS = 100
#fitting the model
history2 = model2.fit(scaled_train_x , scaled_train_y, epochs=EPOCHS, validation_split = 0.3, verbose=0,)

# %%
N = 100

plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history2.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history2.history["val_loss"], label="val_loss")
plt.axvline(x=20, color="red", label="convergence point 1");
plt.axvline(x=55, color="green", label="convergence point 2");

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="middle")
plt.show()

# %% [markdown]
# As you can see , the above model is overfitting since it is not able to generalize on the training data. Here, the model performance slightly improves for Training when compared to the previous model.

# %% [markdown]
# Testing the model
# 
# 

# %%
model2.evaluate(scaled_test_x , scaled_test_y)

# %%
#Lets Print the predicted prices 
test_predictionsk=model2.predict(scaled_test_x)
y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

# %%
from sklearn.metrics import r2_score
r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1)

# %% [markdown]
# ### Model-**3**
# 
# Adding Batch norm **layer**

# %%
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization       # importing Dense layer

def build_model_Batch():  

    model = Sequential() 
    # This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
    input_layer = Dense(16, input_shape=(X_train.shape[1],),activation='relu',kernel_initializer='he_normal')
    model.add(input_layer) 
    
    #Adding the hidden layer
    model.add(BatchNormalization())# defining the batchnorm
    # Notice that we do not need to specify input dim. 
    hidden_layer = Dense(8, activation='relu',kernel_initializer='he_normal'); # defining the weight initialiazer 
    model.add(hidden_layer)
    
    #Adding the output layer
    # Notice that we do not need to specify input dim. 
    # we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
    # We use the sigmoid because we want probability outcomes
    output_layer = Dense(1,activation='relu',kernel_initializer='he_normal') 
    model.add(output_layer)


    optimizer = keras.optimizers.RMSprop(0.001)  # Defining the optimizer 
    #Loss function 
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=[ 'mse'])  # Defining the loss function, optimizer and metrices 
    return model

# %%
#Getting the model summary. We are uisng standalone keras to build our model
model3 = build_model_Batch()
model3.summary()

# %%
EPOCHS = 100
#fitting the model
history = model3.fit(scaled_train_x , scaled_train_y, epochs=EPOCHS, validation_split = 0.3, verbose=0,)

# %%
N = 100
plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="middle")
plt.show()

# %% [markdown]
# As you can see the model is learning now and learning curves are decreasing but there is some noise in training but previously for model-2, the curves were constant and that might be due to vanishing gradients. Batch norm reduces the vanishing gradient problem

# %%
model3.evaluate(scaled_test_x , scaled_test_y)

# %%
#Lets Print the predicted prices 
test_predictionsk=model3.predict(scaled_test_x )
y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

# %%
from sklearn.metrics import r2_score
r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1)

# %% [markdown]
# # Model-4

# %%
def build_model_dropout():
  ## Initializing the ANN
    

    model = Sequential() 
    # This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
    input_layer = Dense(16, input_shape=(X_train.shape[1],),activation='relu',kernel_initializer='he_normal')
    model.add(input_layer) #
    
    #Adding the hidden layer
    model.add(BatchNormalization())# defining the batchnorm
    model.add(Dropout(0.03))# defining the dropout 
    
    # Notice that we do not need to specify input dim. 
    hidden_layer = Dense(8, activation='relu',kernel_initializer='he_normal'); # defining the weight initialiazer 
    model.add(hidden_layer) 
    model.add(Dropout(0.01))# defining the dropout 
    
    
    #Adding the output layer
    # Notice that we do not need to specify input dim. 
    # we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
    # We use the sigmoid because we want probability outcomes
    output_layer = Dense(1,activation='relu',kernel_initializer='he_normal') 

    model.add(output_layer)


    optimizer = keras.optimizers.RMSprop(0.001)  # Defining the optimizer 
    
    #Loss function 
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=[ 'mse'])  # Defining the loss function, optimizer and metrices 
    return model

# %%
#Getting the model summary. We are uisng standalone keras to build our model
model4 = build_model_dropout()
model4.summary()

# %%
#Definign the number of epochs
import numpy as np
EPOCHS = 100
#fitting the model
history = model4.fit(scaled_train_x , scaled_train_y, epochs=EPOCHS, validation_split = 0.3, verbose=0,)

# %%

N = 100

plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="middle")
plt.show()

# %% [markdown]
# As you see, when we have used batchnorm and dropout together then the model has generalized well.

# %%
model4.evaluate(scaled_test_x , scaled_test_y)

# %%
#Lets Print the predicted prices 
test_predictionsk=model4.predict(scaled_test_x )
y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

# %%
from sklearn.metrics import r2_score
r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1)

# %% [markdown]
# ### Call backs - For Early Exit of NN 

# %%
import os
root_logdir = os.path.join(os.curdir, "my_logs") #Creates a folder called my_log in the current dierctly

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") #creates a log file using a time stamp
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir() # e.g., './my_logs/run_2019_06_07-15_15_22'

# %%
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir) #Create a log for loss function or any metric for training and validation

# %%
early_stop=tf.keras.callbacks.EarlyStopping(
    monitor='val_mse', min_delta=0.01, patience=10, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)

# %%
#Getting the model summary. We are uisng standalone keras to build our model
model_tb = build_model_Batch()
model_tb.summary()

# %%
history=model_tb.fit(scaled_train_x , scaled_train_y, epochs=EPOCHS, validation_split = 0.3, verbose=0,\
                  callbacks=[tensorboard_cb, early_stop])

# %%
# %load_ext tensorboard -6008, 6009 or 6007
# %reload_ext tensorboard
# %tensorboard --logdir=my_logs --port=6008

#--6009 and 6007

# %%
# %tensorboard

# %%
model_tb.evaluate(scaled_test_x , scaled_test_y)

# %%
#Lets Print the predicted prices 
test_predictionsk=model_tb.predict(scaled_test_x )
y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

# %%
from sklearn.metrics import r2_score
r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1)

# %% [markdown]
# ### Grid Search CV/ Randomized Search CV
# 
# to implement a Grid or a Randomized Search, we need to define a model function that can take in the hyper parameters. The below function is one such example. Note we will use a Wrapper Class to integrate a NN model with Grid Search.

# %%
# ------------------- L2 Regularizer added below-----------------------

# %%
from tensorflow import keras # importing keras library
from tensorflow.keras.models import Sequential  # importing the Sequential Model
from tensorflow.keras.layers import Dense       # importing Dense layer
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  

def build_model(n_hidden_layer=1, n_neurons=10, input_shape=X_train.shape[1], learning_rate=0.01, active="relu", drop=0.01):
    
    # 1. Sequential model: create model; Initializes the model
    model = Sequential()
    
    model.add(keras.Input(shape=(input_shape,))) #2. Specify the shape of input Layer

    
#     temp=n_neurons
#     i=0
    
    #---------------------The number of neurons in layer n+1 is 2/3 * Number of neurons in layer n---------------
    #hideen layers - Hyper Parameter
    #Number of Neurons is a hyper Parameter
    
    for layer in range(n_hidden_layer):
        
        model.add(Dense(n_neurons, activation=active, kernel_regularizer=tf.keras.regularizers.L1(0.01)))

        if n_hidden_layer%2==0:
          
          model.add(Dropout(drop)) #-----Regularization parameters
        
        n_neurons=int(round(n_neurons*1/2)) #-----Pyramid process
    

    
    model.add(Dense(1, activation = 'relu')) #---sigmoid
    
    # Compile model
    
    optimizer=keras.optimizers.RMSprop(lr=learning_rate)

    model.compile(optimizer = optimizer, loss = tf.keras.losses.Huber(), metrics=['mse']) 

    # model.compile(optimizer = optimizer, loss = 'binary_categorical_crossentropy', metrics=['accuarcy', 'recall','precision', 'f1_score']) 
    
    return model

# %%
from sklearn.base import clone

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor #----KerasClassifier
 
model_opt = KerasRegressor(build_fn=build_model, epochs=100, batch_size=32, verbose=0)

# %%
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden_layer": [2,3,4,5],
    "n_neurons": [16, 32, 64],
    "active": ['relu', 'selu'],
    "learning_rate": reciprocal(3e-4, 3e-2)
#     "opt":['Adam']
}

#---dt=DecisionTreeClassifier(random_state=7)

#---Randomized Search CV - K fold cross validation to decide the best hyper parameters

rnd_search_cv = RandomizedSearchCV(model_opt, param_distribs, n_iter=5, cv=3, scoring='neg_mean_squared_error')

rnd_search_cv.fit(scaled_train_x, scaled_train_y, epochs=50)

# %%
print("Best: Paramter %s" % (rnd_search_cv.best_params_)) #---This combination produces the lowest average cross validation error

# %%
model_tuned = rnd_search_cv.best_estimator_.model

# %%
#Lets Print the predicted prices 
test_predictionsk=model_tuned.predict(scaled_test_x )
y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

from sklearn.metrics import r2_score
r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1)

# %% [markdown]
# ### Explainability using SHAP (Shapely Additive)
# 
# Alternatively there is another package called LIME.
# 
# https://github.com/marcotcr/lime
# 
# Widely used with Image Processing.

# %%
# !pip install shap

# %%
import shap

# we use the first 100 training examples as our background dataset to integrate over
explainer = shap.DeepExplainer(model_tuned, scaled_train_x)

# %%
explainer

# %%
# KernelExplainer is a general approach that can work with any ML framework
# Its inputs are the predictions and training data

#Step 1:
# Summarize the training set to accelerate analysis
df_train_normed_summary = shap.kmeans(scaled_train_x, 25) #Cluster data points into n groups (n can be any value, hypothetically I used 25 here)

#Step 2
# Instantiate an explainer with the model predictions and training data summary; Tree based models like Random forest, XGBoost replace kernel explainer with Tree Explainer 
explainer = shap.KernelExplainer(model_tuned.predict, df_train_normed_summary)

#step 3:
# Extract Shapley values from the explainer
shap_values = explainer.shap_values(scaled_train_x) #Explainability

# %%
import pandas as pd

data_interim=pd.DataFrame(scaled_train_x) #Given smote is an array or a numpy array
data_interim.columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B100", "LSTAT"]
data_interim.head()

# %%
# Summarize the Shapley values in a plot
shap.summary_plot(shap_values[0], data_interim)

# %% [markdown]
# Each dot has three characteristics:
# 
# - Vertical location shows what feature it is depicting
# - Color shows whether that feature was high or low for that row of the dataset
# - Horizontal location shows whether the effect of that value caused a higher or lower prediction.

# %%
model_tuned.save("RCV_model.h5")

# %% [markdown]
# ### Transfer Learning

# %%
model_A = keras.models.load_model("RCV_model.h5")

model_B_on_A = keras.models.Sequential(model_A.layers[:-1])

model_B_on_A.add(keras.layers.Dense(1, activation="relu"))

# %%
model_A_clone = keras.models.clone_model(model_A)

model_A_clone.set_weights(model_A.get_weights())

# %%
for layer in model_B_on_A.layers[:-1]:
  layer.trainable = False

# %%
optimizer=keras.optimizers.RMSprop(lr=0.01)

model_B_on_A.compile(optimizer = optimizer, loss = tf.keras.losses.Huber(), metrics=['mse']) 

#Definign the number of epochs
import numpy as np
EPOCHS = 100
#fitting the model
history = model_B_on_A.fit(scaled_train_x , scaled_train_y, epochs=EPOCHS, validation_split = 0.3, verbose=0,)

# %%
N = 100

plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="middle")
plt.show()

# %%
#Lets Print the predicted prices 

test_predictionsk=model_B_on_A.predict(scaled_test_x)

y_pred1=scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

from sklearn.metrics import r2_score
print("R2:", round(r2_score(scaler.inverse_transform(scaled_test_y.reshape(-1, 1)),y_pred1),2))

# %%
from tensorflow import keras # importing keras library
from tensorflow.keras.models import Sequential  # importing the Sequential Model
from tensorflow.keras.layers import Dense       # importing Dense layer

initializer = tf.keras.initializers.GlorotNormal()

def build_model_Skeleton():
    
    ## Initializing the ANN
    model = Sequential() 
    
    # This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
    input_layer = Dense(16, input_shape=(X_train.shape[1],),activation='relu', kernel_initializer=initializer)
    model.add(input_layer) # 
    
    #Adding the hidden layer
    # Notice that we do not need to specify input dim. 
    hidden_layer = Dense(8, activation='relu', kernel_initializer=initializer); 
    model.add(hidden_layer) 
    
    #Adding the output layer
    # Notice that we do not need to specify input dim. 
    # we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
    # We use the sigmoid because we want probability outcomes
    
    output_layer = Dense(1,activation='sigmoid')

    # output_layer = Dense(1,activation='softmax')  

    model.add(output_layer)


    optimizer = keras.optimizers.RMSprop(0.001)  # Defining the optimizer #0.01 #0.001 #0.0001
    
    #Loss function 
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy', Recall(), Precision()])  # Defining the loss function, optimizer and metrices 

    # model.compile(loss='categorical_crossentropy',
    #             optimizer=optimizer,
    #             metrics='mse')  # Defining the loss function, optimizer and metrices 
    return model

