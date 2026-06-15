# %% [markdown]
# ## Bank Churn Prediction Project on Neural Networks
# Background and Context
# 
# Businesses like banks that provide service have to worry about the problem of 'Churn' i.e. customers leaving and joining another service provider. It is important to understand which aspects of the service influence a customer's decision in this regard. Management can concentrate efforts on the improvement of service, keeping in mind these priorities.
# 
# Objective
# 
# Given a Bank customer, build a neural network-based classifier that can determine whether they will leave or not in the next 6 months.

# %% [markdown]
# Data Dictionary
# 
#     CustomerId: Unique ID which is assigned to each customer
#     Surname: Last name of the customer 
#     CreditScore: It defines the credit history of the customer.  
#     Geography: A customerâ€™s location    
#     Gender: It defines the Gender of the customer   
#     Age: Age of the customer     
#     Tenure: Number of years for which the customer has been with the bank
#     NumOfProducts: It refers to the number of products that a customer has purchased through the bank.
#     Balance: Account balance
#     HasCrCard: It is a categorical variable that decides whether the customer has a credit card or not.
#     EstimatedSalary: Estimated salary 
#     isActiveMember: It is a categorical variable that decides whether the customer is an active member of the bank or not ( Active member in the sense, using bank products regularly, making transactions, etc )
#     Excited: It is a categorical variable that decides whether the customer left the bank within six months or not. It can take two values 
# 
#                     0=No ( Customer did not leave the bank )
# 
#                     1=Yes ( Customer left the bank 

# %%
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import warnings

warnings.filterwarnings("ignore")
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# %%
#%load_ext nb_black

# Library to suppress warnings or deprecation notes
import warnings

warnings.filterwarnings("ignore")


# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# Libraries to build decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV

# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    plot_confusion_matrix,
    make_scorer,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)


# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score



# %%


# %%
# Mount Google drive to access the dataset (monkeys_dataset.zip)
from google.colab import drive
drive.mount('/content/drive')


# %%
loan = pd.read_csv("/content/drive/MyDrive/Churn.csv")

# %%
loan

# %% [markdown]
# ### 10 000 rows of data with 14 columns/features

# %%
loan.describe()

# %% [markdown]
# ### Conclusions 
# 
# Mean credit score is 650.5min. 350 max 850.  
# 
# Mean age 38.9 years with range from 18 to 92 years.
# 
# Mean tenure is 5 years with minim 0 and Maximum 10 years.
# 
# Balance mean 76000 with min. 0 and max. approximately 250000. 
# 
# Number of products mean 1.5 with minimum 1 and max 4.
# 
# Estimated salary mean 100000 min 11 max 200000.
# 

# %%
loan.isna().any()

# %% [markdown]
# ###  Nomissing values. 

# %%
loan.info()

# %% [markdown]
# ### Note: Geography and gender will need to be encoded. 

# %% [markdown]
# ### Checking for numbers ofunique values for each of the feature

# %%
for col in loan.columns:
    print("Number of unique values in ", col, len(loan[col].unique()))


# %%
# customized boxplot+histogram with mean and median values
def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):

    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="yellow")
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")

# %%
histogram_boxplot(loan, "CreditScore")

# %% [markdown]
# ### Conlcusions
# Mean is close to median so distribution is close to normal distibtin in that respect, for a given dataset. 
# 
# The max credit score available is 850 so the distribtion cannot have the same tail as on the left side. Therefore the weird behavior of histogram at 850 the max credit score. 
# 
# Notice few outliers with very low score.

# %%
histogram_boxplot(loan, "Age")

# %% [markdown]
# ### Conclusion: Several outliers above age 60. 

# %%
histogram_boxplot(loan, "EstimatedSalary")

# %%
histogram_boxplot(loan, "Balance")

# %% [markdown]
# ### Note
# There is a relatively large grop of customers who have zero balance.This causes the change of the overall distrbution. But also since these is a 3500+ cusstomers gop I will later check this group separately. I 
# do not tihnk I can drop this group as it is around 1/3 of data set and the infomration about this customers might help understand the data set. 

# %% [markdown]
# ### Barplots and detailed distributions

# %%
def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

# %%
labeled_barplot(loan, "Geography", perc=True)

# %% [markdown]
# ### Conclusion:
# Location of customers: 50 % from France 25% from Germany 25 % from Spain.

# %% [markdown]
# ### Note
# But in my opinion the data set is not large enough to model separately each of the subgroup. 
# 
# According to literature the more important factor is the number of data points. 
# (Smaller dataset would only be enough if I have reason to reduce features). 

# %%
labeled_barplot(loan, "Gender", perc=True)

# %% [markdown]
# ### Conclusion: Dataset is approximately balanced with respect to gender

# %%
labeled_barplot(loan, "Exited", perc=True)

# %% [markdown]
# ### 20.4% of customers left the bank. It is an imbalanced dataset and might require the augmentation of the data subset for theminority class in order to imporve the precision.

# %%
labeled_barplot(loan, "NumOfProducts", perc=True)

# %% [markdown]
# ### Conclusion about the number of products:
# Less than 3% of customers have 3 or 4 products.

# %%
labeled_barplot(loan, "HasCrCard", perc=True)

# %% [markdown]
# ### Conclusion  about the Credit card possession
# 30% of customers have no credit card.

# %%
labeled_barplot(loan, "IsActiveMember", perc=True)

# %% [markdown]
# ### Conclusion about the customers service use habits
# 51.5% of customers are regularly using the services. 

# %%
import seaborn as sns

# %%
sns.displot(
    loan,
    x="CreditScore",
    hue="Exited",
    bins=30,
    palette="winter",
    kde=True,
    stat="probability",
)

# %% [markdown]
# ### Conclusion about the Credit scores for the customers who stay and those who exit. 
# Credit score distribution among the customers who stay or leave is similar. No striking differences.

# %%
sns.displot(
    loan, x="Age", hue="Exited", bins=20, palette="winter", kde=True,
)

# %% [markdown]
# ### Conclusion about the Age for the groups of customers who stay and exit:
# On the fist view the distribution of age for customers who left the bank seems to suggest that more mature customers leave the bank. A  question is if it is related only wiht age or also with the tenure of the customers. 

# %%
sns.boxplot(x="NumOfProducts", y="EstimatedSalary", data=loan)

# %% [markdown]
# ### Conclusion for the salary of customers who have access to diferent bank products
# Number of products a customers have does not seem to strongly depend on salary, except for 4 products (average salary higher). 

# %%
sns.boxplot(x="Exited", y="EstimatedSalary", data=loan)

# %% [markdown]
# ### Conclusion about salary of customers who stay and exit 
# The customers who stay and who leave have similar salary range/mean.

# %% [markdown]
# Before proceeding to more detailed study of the data I plot the pairplot and correlations

# %%


sns.pairplot(loan, hue="Exited")



# %% [markdown]
# ### Conclusion: Dropping the first 3 columns as they do not bring anything to the analysis or modeling of the data.

# %%
data = loan.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# %%
sns.pairplot(data, hue="Exited")

# %% [markdown]
# ### Conclusion 
# 
# One can see the concentration of the age distribution of "exited" customers around the age  of 50 years.
# 
# A large number of excited custmers have 3 or 4 products.
# 
# One unusual thing is that there is a group of exited customers that seems to have low credit score (the lowest) 
# with relatively high salary. 

# %%
plt.figure(figsize=(22, 22))
sns.heatmap(data.corr(), annot=True, cmap="Spectral")

# %% [markdown]
# ### Conclusion: 
# 
# there does not seemtobe large correlation. The only possible - weak correlaiton is for number of products and balance. It is understandable because the more money a person has themore products (for example credit cards, loans, stock investing options) one can be offered by the banking institution. 
# 

# %% [markdown]
# ### Now I will check the mean and median for different subgroups:  

# %%
data.groupby("NumOfProducts")["Age"].mean()

# %% [markdown]
# ### Conclusion: 
# those customers who have 3 and 4 products - are older. 

# %%
data.groupby("NumOfProducts")["Tenure"].mean()

# %% [markdown]
# ### Conclusion:
# Customers for different product number group have similar tenure length.

# %%
data.groupby("NumOfProducts")["EstimatedSalary"].mean()

# %% [markdown]
# ### Conclusion:
# Customers for different product number group have similar average salary

# %%
data.groupby("NumOfProducts")["Balance"].mean()

# %% [markdown]
# ### Conclusion:
# There are some differences in Balance for different customers whohave different number of products but i am not sure what it might mean.

# %%
data.groupby("NumOfProducts")["CreditScore"].mean()

# %% [markdown]
# ### Conclusion:
# Customers for different product number groups have no significat differences for average credit score.

# %%
data.groupby("NumOfProducts")["CreditScore"].median()

# %% [markdown]
# ### Conclusion:
# Slightly decreasing trend for median of credit score wih increasing number of products. I do not think it is significant at this point

# %%
data.groupby("NumOfProducts")["Balance"].median()

# %% [markdown]
# ### Note: clients who have 2 products have median value for Balance -zero. 
# This group had already the lowers average balance and is visible on the histogram plot above.

# %%
data.groupby("NumOfProducts")["EstimatedSalary"].median()

# %% [markdown]
# ### Conclusion: 
# Increasing trend for median Estimated salary with respect to number of products.

# %% [markdown]
# ## Deep dive:
# 
# 1) I will check for the customers who have exited and those who - stay  separately. 
# 
# 2) I will check the correlations between the high salary low credit score group. 
# 
# 3) I will check number of products (1, 2 or 3 and 4)
#     

# %%
exited = data[data["Exited"] == 1]

# %% [markdown]
# ### Note: I have checked pairplot of the "Exite" group with respect to number ofproducts and there was nothing unusual.I deleted the plot as these are big plots that take up space.

# %%
sns.pairplot(exited, hue="Gender")

# %% [markdown]
# ### Plot of the data for the exit customers does not reveal anything interesting with respect to Gender. 
# I will study it in a littlemore detail.

# %%
g = sns.jointplot(
    data=exited, x="Balance", y="CreditScore", hue="Gender"
)



# %%
g = sns.jointplot(
    data=exited, x="Balance", y="EstimatedSalary", hue="Gender"
)


# %%
labeled_barplot(exited, "Gender", perc=True)

# %% [markdown]
# ### Conclusion
# 56% percent of exit customers are female customers. the difference is 11 % for the excit customer group when for the over all group the minority are the femalecustomer (45%).
# 
# 

# %%
g = sns.jointplot(
    data=exited, x="CreditScore", y="EstimatedSalary", hue="Gender"
)


# %%
g = sns.jointplot(
    data=exited, x="Age", y="EstimatedSalary", hue="Gender"
)


# %% [markdown]
# I do not see any visible differences in the female and male customers for the exit group.

# %% [markdown]
# I move on to check for bivariate distributions with respect to other categorical variablesfor the exit group.

# %%
g = sns.jointplot(
    data=exited, x="CreditScore", y="EstimatedSalary", hue="NumOfProducts"
)


# %%
g = sns.jointplot(
    data=exited, x="CreditScore", y="EstimatedSalary", hue="IsActiveMember"
)


# %%
g = sns.jointplot(
    data=exited, x="CreditScore", y="EstimatedSalary", hue="HasCrCard"
)


# %%
g = sns.jointplot(
    data=exited, x="CreditScore", y="Age", hue="IsActiveMember"
)


# %%
plt.figure(figsize=(22, 22))
sns.heatmap(exited.corr(), annot=True, cmap="Spectral")

# %% [markdown]
# ### Conclusion on correlation between variables:
# I really do not see anything here. I will quicl check the zero balance group of customers as this one is the onethat stood out.

# %%
zerob = data[data["Balance"] == 0]

# %%
g = sns.jointplot(
    data=zerob, x="CreditScore", y="Age", hue="IsActiveMember"
)


# %%
g = sns.jointplot(
    data=zerob, x="CreditScore", y="EstimatedSalary", hue="Exited"
)


# %% [markdown]
# ### Conclusion:
# Those with zero balance do not have anything unusual in the distributions - I have checked several fetaures and nothing stands out.Except maybe that I would assume zero balance means some different salary distribution. But it seems tobe not the case. 

# %% [markdown]
# ### Model preparation: I split the data into features ( matrix X) and  vector y will contain the target variable.

# %%
X = data.drop(labels=["Exited"], axis=1)
y = data["Exited"]

# %% [markdown]
# ### Model preparation: I encode the categorical data using Label encoder from sklearn and get dummies functions.

# %%
label1 = LabelEncoder()
X["Geography"] = label1.fit_transform(X["Geography"])

# %%
label = LabelEncoder()
X["Gender"] = label.fit_transform(X["Gender"])
X.head()

# %%
X = pd.get_dummies(X, drop_first=True, columns=["Geography"])
X.head()

# %% [markdown]
# Split the data into test and train

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# %% [markdown]
# ### Model preparation: Scaling data using standard scaler:

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ## Building Neural networks using Keras
# ### Model 1 (128-1) 

# %%
import keras  # importing keras library
from keras.models import Sequential  # importing the Sequential Model
from keras.layers import Dense  # importing Dense layer

# %%
model = Sequential()
model.add(Dense(X.shape[1], activation="relu", input_dim=X.shape[1]))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# %%
model.fit(X_train, y_train.to_numpy(), batch_size=300, epochs=100, verbose=1)

# %% [markdown]
# ### Conclusion:
# Accuracy on train set 0.86

# %%

model.evaluate(X_test, y_test.to_numpy())


# %% [markdown]
# ### Comment: 
# Accuracy on test set 0.86

# %%
predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)

# %%
confusion_matrix(y_test, classes_x)

# %%
y_predicted = model.predict(X_test)
# y_predicted for test transformation
y_pred_trans = []
for pred in y_predicted:
    if pred > 0.5:
        y_pred_trans.append(1)
    else:
        y_pred_trans.append(0)

y_pred_trans[0:10]

# %%
from sklearn.metrics import roc_curve, roc_auc_score

r_probs = [0 for _ in range(len(y_test))]
# Calculate the AUROC
r_auc = roc_auc_score(y_test, r_probs)
on_auc = roc_auc_score(y_test, y_pred_trans)
print("Random Chance Prediction AUROC: " + str(r_auc))
print("One Neuron Network Prediction AUROC: " + str(on_auc))
r_fpr, r_tpr, threshold = roc_curve(y_test, r_probs)
on_fpr, on_tpr, threshold = roc_curve(y_test, y_pred_trans)
threshold[0:10]

from matplotlib.pyplot import figure

plt.figure(figsize=(10, 8))
plt.plot(
    r_fpr, r_tpr, linestyle="--", label="Random Chance Prediction AUROC: %0.3f" % r_auc
)
plt.plot(on_fpr, on_tpr, marker=".", label="Model 5:  %0.3f" % on_auc)

# TITLE
plt.title("ROC Plot")
# Axis Label
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# Legend
plt.legend()
# show plot
plt.show()

# %% [markdown]
# ### Conclusions:
# 
# I have plotted the ROC curve and the confusion matrix.
# 
# There is no overfitting.  

# %% [markdown]
# I do not consider this a good model as accuracy on 0.86 meanswe do not perform mch better than if one throws away all of the relevant information and the minority class.
# 
# Accuracy on both the test and train is 0.86.

# %% [markdown]
# ## Model 2  (128-64-64-32-1) - with added more  dense layers.

# %%
model2 = Sequential()
model2.add(Dense(X.shape[1], activation="relu", input_dim=X.shape[1]))
model2.add(Dense(128, activation="relu", kernel_initializer="he_normal"))
model2.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
model2.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
model2.add(Dense(32, activation="relu", kernel_initializer="he_normal"))
model2.add(Dense(1, activation="sigmoid"))

# %%
model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# %%
model2.fit(X_train, y_train.to_numpy(), batch_size=300, epochs=300, verbose=1)

# %%
model2.evaluate(X_test, y_test.to_numpy())


# %% [markdown]
# ### Conclusion:
# Example of overfitting - accuracy on train set 0.99 but on the train test 0.8. So this model is worse than a simplemodel above. 

# %% [markdown]
# ## For comparison to neural networks precision - several simpler classifiers from sklearn

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

# %%
def stratified_cv(X, y, clf_class, shuffle=True, **kwargs):
    stratified_k_fold = StratifiedKFold().split(X, y)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        Xtrain, Xtest = X.iloc[ii], X.iloc[jj]
        ytrain = y.iloc[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred.iloc[jj] = clf.predict(Xtest)
    return y_pred

# %%
print(
    "Gradient Boosting Classifier:  {:.2f}".format(
        metrics.accuracy_score(
            y, stratified_cv(X, y, ensemble.GradientBoostingClassifier)
        )
    )
)


print(
    "Random Forest Classifier:      {:.2f}".format(
        metrics.accuracy_score(y, stratified_cv(X, y, ensemble.RandomForestClassifier))
    )
)

print(
    "K Nearest Neighbor Classifier: {:.2f}".format(
        metrics.accuracy_score(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))
    )
)

print(
    "Logistic Regression:           {:.2f}".format(
        metrics.accuracy_score(y, stratified_cv(X, y, linear_model.LogisticRegression))
    )
)
#print(
#    "XGBoost Classifier:           {:.2f}".format(
#        metrics.accuracy_score(y, stratified_cv(X, y, XGBClassifier))
#    )
#)

# %%
print(
    "Gradient Boosting Classifier:\n {}\n".format(
        metrics.classification_report(
            y, stratified_cv(X, y, ensemble.GradientBoostingClassifier)
        )
    )
)

print(
    "Random Forest Classifier:\n {}\n".format(
        metrics.classification_report(
            y, stratified_cv(X, y, ensemble.RandomForestClassifier)
        )
    )
)

print(
    "K Nearest Neighbor Classifier:\n {}\n".format(
        metrics.classification_report(
            y, stratified_cv(X, y, neighbors.KNeighborsClassifier, n_neighbors=11)
        )
    )
)

print(
    "Logistic Regression:\n {}\n".format(
        metrics.classification_report(
            y, stratified_cv(X, y, linear_model.LogisticRegression)
        )
    )
)

#print(
#    "XGBoost Classifier:\n {}\n".format(
#        metrics.classification_report(y, stratified_cv(X, y, XGBClassifier))
#    )
#)

# %% [markdown]
# ### Conclusion 
# standard classifiersincluding XGBoost - get accuracy around 0.8 so the basic neural network model (model 1) performed better. As one can see the sklearn classifiers without hypertuning do not perform very well on this data set. 

# %% [markdown]
# ##  Study the model properties  in detail and check more carefully for other metrics.
# 
# ### Model 3 (64-32-1) without dropout layers

# %%
import keras  # importing keras library
from keras.models import Sequential  # importing the Sequential Model
from keras.layers import Dense  # importing Dense layer


def build_model():
    ## Initializing the ANN

    model = Sequential()
    input_layer = Dense(
        64,
        input_shape=(X_train.shape[1],),
        activation="relu",
        kernel_initializer="he_normal",
    )
    model.add(input_layer)  #
    hidden_layer = Dense(32, activation="relu", kernel_initializer="he_normal")
    model.add(hidden_layer)
    output_layer = Dense(1, activation="sigmoid")
    model.add(output_layer)

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )  
    return model

# %%
model2 = build_model()
model2.summary()

# %%
# Definign the number of epochs
EPOCHS = 100
# fitting the model
history = model2.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=1,)

# %% [markdown]
# Accuracy on the train test 0.91

# %%
N = 100
import pylab as plt

plt.figure(figsize=(8, 6))
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
# plt.legend(loc="middle")
plt.show()

# %%
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# %% [markdown]
# ### Conclusion -
# On the train set one can achieve higher accuracy but the accuracy on the test set is max 0.86.

# %%
model2.evaluate(X_test, y_test)

# %% [markdown]
# ### Conclusion: 
# Accuracy on test set 0.84 and on the train set 0.9 so we have overfitting.

# %% [markdown]
# I move to the confusion matrix

# %%
y_predicted2 = model2.predict(X_test)
# y_predicted for test transformation
y_pred_trans2 = []
for pred in y_predicted2:
    if pred > 0.5:
        y_pred_trans2.append(1)
    else:
        y_pred_trans2.append(0)

y_pred_trans2[0:10]

# %%
from sklearn.metrics import roc_curve, roc_auc_score

r_probs = [0 for _ in range(len(y_test))]
# Calculate the AUROC
r_auc = roc_auc_score(y_test, r_probs)
on_auc = roc_auc_score(y_test, y_pred_trans2)
print("Random Chance Prediction AUROC: " + str(r_auc))
print("Model 2 Prediction AUROC: " + str(on_auc))
r_fpr, r_tpr, threshold = roc_curve(y_test, r_probs)
on_fpr, on_tpr, threshold = roc_curve(y_test, y_pred_trans)
threshold[0:10]

from matplotlib.pyplot import figure

plt.figure(figsize=(10, 8))
plt.plot(
    r_fpr, r_tpr, linestyle="--", label="Random Chance Prediction AUROC: %0.3f" % r_auc
)
plt.plot(on_fpr, on_tpr, marker=".", label="Model 5:  %0.3f" % on_auc)

# TITLE
plt.title("ROC Plot")
# Axis Label
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# Legend
plt.legend()
# show plot
plt.show()

# %% [markdown]
# ### Conclusion:
# 
# Model has some overfitting. It performs better than "random". 
# But it shows that on the test set it does not perform very well. 
# 
# In another model I will add dropout.

# %% [markdown]
# ### Model 4 (64-32-1) with dropout and early stopping 

# %%
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm

# %%
# Model 3
nn_model = Sequential()
nn_model.add(
    Dense(
        64,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        activation="relu",
        kernel_initializer="he_normal",
    )
)
nn_model.add(Dropout(rate=0.2))
nn_model.add(
    Dense(
        32,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        activation="relu",
        kernel_initializer="he_normal",
    )
)
nn_model.add(Dropout(rate=0.1))
nn_model.add(Dense(1, activation="sigmoid"))
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001, decay_steps=(X_train.shape[0] / 32) * 50, decay_rate=1, staircase=False
)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=70, restore_best_weights=True
        )
    ]


nn_model.compile(
    loss="binary_crossentropy", optimizer=get_optimizer(), metrics=["accuracy"]
)


historynn = nn_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32,
    callbacks=get_callbacks(),
    verbose=0,
)

plt.plot(historynn.history["accuracy"])
plt.plot(historynn.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# %% [markdown]
# ## Conclusion: Adding dropout greately increased convergence between the train and test sets. It does not improve overall accuracy compared to the simplest neural network   above.
# 
# Dropout and simple few layer network has little overfitting.

# %%
from sklearn.metrics import roc_curve, roc_auc_score

r_probs = [0 for _ in range(len(y_test))]
# Calculate the AUROC
r_auc = roc_auc_score(y_test, r_probs)
on_auc = roc_auc_score(y_test, y_pred_trans_nn)
print("Random Chance Prediction AUROC: " + str(r_auc))
print("Model 2 Prediction AUROC: " + str(on_auc))
r_fpr, r_tpr, threshold = roc_curve(y_test, r_probs)
on_fpr, on_tpr, threshold = roc_curve(y_test, y_pred_trans)
threshold[0:10]

from matplotlib.pyplot import figure

plt.figure(figsize=(10, 8))
plt.plot(
    r_fpr, r_tpr, linestyle="--", label="Random Chance Prediction AUROC: %0.3f" % r_auc
)
plt.plot(on_fpr, on_tpr, marker=".", label="Model 5:  %0.3f" % on_auc)

# TITLE
plt.title("ROC Plot")
# Axis Label
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# Legend
plt.legend()
# show plot
plt.show()

# %%
y_predicted_train_nn = nn_model.predict(X_train)
y_pred_train_nn = []
for pred in y_predicted_train_nn:
    if pred > 0.5:
        y_pred_train_nn.append(1)
    else:
        y_pred_train_nn.append(0)

# %%
y_predicted_nn = nn_model.predict(X_test)
# y_predicted for test transformation
y_pred_trans_nn = []
for pred in y_predicted_nn:
    if pred > 0.5:
        y_pred_trans_nn.append(1)
    else:
        y_pred_trans_nn.append(0)

y_pred_trans_nn[0:10]

# %%
confusion_matrix(y_test, y_pred_trans_nn)

# %%
cm_matrix = confusion_matrix(y_test, y_pred_trans_nn)


# %%
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cm_matrix,
    classes=["0", "1"],
    normalize=False,
    title="Neuron Network Confusion matrix",
)

# %% [markdown]
# ### Conclusion
# We have more of False Negative - so the customers who would be predicted to stay but who would leave the bank. 
# So from business pointof view this model is worse. 

# %% [markdown]
# ### Possibilitles for improvement:

# %% [markdown]
# One can increase precision on the train set but but then the overfitting is a problem. 

# %% [markdown]
# One can improve the imbalance of the classes as we start with 20% and by misclassifing at this level - the model performs rather poorly. 

# %% [markdown]
# ### Model 5 (64-32+256-1) added another sigmoid layer

# %%
model4 = Sequential()
model4.add(Dense(32, activation="relu", kernel_initializer="he_normal"))
input_layer = Dense(
    64,
    input_shape=(X_train.shape[1],),
    activation="relu",
    kernel_initializer="he_normal",
)
model4.add(input_layer)  #
hidden_layer = Dense(32, activation="relu", kernel_initializer="he_normal")
model4.add(hidden_layer)
model4.add(Dense(256, activation="sigmoid"))
model4.add(Dense(1, activation="sigmoid"))

# %%
METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", verbose=4, patience=10, mode="max", restore_best_weights=True
)

model4.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=METRICS,
)

# %%
history4 = model4.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=200,
    validation_split=0.30,
    shuffle=True,
    callbacks=[early_stopping],
)

# %%

history4.history['loss']


# %%
# Plot training and validation accuracy values
plt.plot(history4.history["accuracy"])
plt.plot(history4.history["val_accuracy"])
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Plot training and validation accuracy values
plt.plot(history4.history["loss"])
plt.plot(history4.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# %% [markdown]
# ### Conclusion:
# For the model here we have overfitting.  This model representsearly stopping. 
# It doesnot perform as well as drop out.

# %% [markdown]
# But I will plot consusion matrix for this one too.

# %%
y_predicted_train4 = model4.predict(X_train)
y_pred_train4 = []
for pred in y_predicted_train4:
    if pred > 0.5:
        y_pred_train4.append(1)
    else:
        y_pred_train4.append(0)
y_pred_train[0:5]

# %%
y_predicted4 = model4.predict(X_test)
# y_predicted for test transformation
y_pred_trans4 = []
for pred in y_predicted4:
    if pred > 0.5:
        y_pred_trans4.append(1)
    else:
        y_pred_trans4.append(0)

y_pred_trans[0:10]

# %%
# from sklearn.metrics import fbeta_score

# fbeta_score(y_test, y_pred_trans, average="weighted", beta=0.5)
# Compute confusion matrix
cm_matrix4 = confusion_matrix(y_test, y_pred_trans4)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cm_matrix4,
    classes=["0", "1"],
    normalize=False,
    title="Neuron Network Confusion matrix",
)

# %% [markdown]
# ### Conclusion:
# As expected from overfitting and lower accuracy on the test set - this confusion matrix has rather poor FN,FT.

# %% [markdown]
# ### Model 5 (64+256+1) Smaller neural  network  with 2 sigmoid layers

# %%
model5 = Sequential()
model5.add(
    Dense(
        64,
        activation="relu",
        use_bias=True,
        bias_initializer="zeros",
        kernel_initializer="he_normal",
    )
)
model5.add(Dense(256, activation="sigmoid"))
model5.add(Dense(1, activation="sigmoid"))

# %%
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy', 
    verbose=4,
    patience=10,
    mode='max',
    restore_best_weights=True)

model5.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)

history5 = model5.fit(X_train, y_train, 
                    epochs = 100, batch_size = 200, 
                    validation_split = 0.30, 
                    shuffle = True, callbacks=[early_stopping],)



# %%
y_predicted5 = model5.predict(X_train)

# %%
# y_predicted for test transformation
y_pred_trans5 = []
for pred in y_predicted5:
    if pred > 0.5:
        y_pred_trans5.append(1)
    else:
        y_pred_trans5.append(0)

y_pred_trans5[0:10]

# %%
# Plot training and validation accuracy values
plt.plot(history5.history["accuracy"])
plt.plot(history5.history["val_accuracy"])
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Plot training and validation accuracy values
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# %% [markdown]
# ### Conclusion 
# diference in accuracy between train and test -0.88 vs 0.85 respectively.

# %% [markdown]
# this was test to see if additional sigmoid layer  causes anything (not much for this data set)

# %%
from sklearn.metrics import roc_curve, roc_auc_score

r_probs = [0 for _ in range(len(y_test))]
# Calculate the AUROC
r_auc = roc_auc_score(y_test, r_probs)
on_auc = roc_auc_score(y_test, y_pred_trans5)
print("Random Chance Prediction AUROC: " + str(r_auc))
print("One Neuron Network Prediction AUROC: " + str(on_auc))
r_fpr, r_tpr, threshold = roc_curve(y_test, r_probs)
on_fpr, on_tpr, threshold = roc_curve(y_test, y_pred_trans)
threshold[0:10]

from matplotlib.pyplot import figure

plt.figure(figsize=(10, 8))
plt.plot(
    r_fpr, r_tpr, linestyle="--", label="Random Chance Prediction AUROC: %0.3f" % r_auc
)
plt.plot(on_fpr, on_tpr, marker=".", label="Model 5:  %0.3f" % on_auc)

# TITLE
plt.title("ROC Plot")
# Axis Label
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# Legend
plt.legend()
# show plot
plt.show()

# %% [markdown]
# ## Model 6 (1024-1024-1024) with dropout and batch normalization

# %%
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization

model6 = keras.Sequential()
model6.add(Dense(1024, activation="relu", kernel_initializer="he_normal"))
model6.add(BatchNormalization())
model6.add(Dense(1024, activation="relu", kernel_initializer="he_normal"))
model6.add(Dropout(0.3))
model6.add(BatchNormalization())
model6.add(Dense(1024, activation="relu", kernel_initializer="he_normal"))
model6.add(Dropout(0.3))
model6.add(BatchNormalization())
model6.add(Dense(1, activation="sigmoid"))

# %%
ETRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", verbose=4, patience=10, mode="max", restore_best_weights=True
)

model6.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)

history6 = model6.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=200,
    validation_split=0.30,
    shuffle=True,
    callbacks=[early_stopping],
)

# %%
y_predicted6 = model6.predict(X_train)

# %%
# y_predicted for test transformation
y_pred_trans6 = []
for pred in y_predicted6:
    if pred > 0.5:
        y_pred_trans6.append(1)
    else:
        y_pred_trans6.append(0)



# %%
# Plot training and validation accuracy values
plt.plot(history6.history["accuracy"])
plt.plot(history6.history["val_accuracy"])
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Plot training and validation accuracy values
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# %% [markdown]
# ### Conclusion: 
# I have used many layers - because I wanted to see one bigger claulaction result. 
# Another model that performs well on train test but is overfitting. 
# 
# Batch normalization and dropout did not improve the accuracy. 
# 

# %% [markdown]
# ## Final conclusions:
# 
# 1) fewer dense /hidden lares  leads to better result that exceeds XGBoost.
# 
# 2) Batch renormalization has small effect.
# 
# 3) Dropout has rather significant effect on the results. 
# 
# The best fitted model is achieving accuracy of 0.86-0.87 on both train and test sets and uses dropout. 
# 
# ## More can be done for exmaple using methods for imbalanced data sets.
# 
# 

# %% [markdown]
# ## Maybe PCA or SHAP can help to select most significant features and in this way remove the noise tgether with less significant features.

# %%


# %%


# %%


# %%


# %%


