# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


# read the data
df = pd.read_csv('diabetes.csv') 
print(df.shape)

# summary statistics
df.describe().transpose()

# have the label in one column
target_column = ['diabetes'] 

# feature list
predictors = list(set(list(df.columns))-set(target_column))

# scale features to be in [0,1]
df[predictors] = df[predictors]/df[predictors].max()

# summary stats of scaled features
df.describe().transpose()

# divide data into test and training samples
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

# next we use MLPClassifier from sklearn.neural_network

# set up the model with two hidden layers where the first layer has 10 nodes and the second has 5
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state = 4)

# fit it to the training data
mlp.fit(X_train,y_train.ravel())

# apply it to the test data and compute the fitted values
y_pred = mlp.predict(X_test)


# asses model performance
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# plot the confusion matrix
mat = confusion_matrix(y_pred, y_test)
names = np.unique(y_test)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()


# let's see if the regularized network does better
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state = 4, alpha=0.001)

# fit it to the training data
mlp.fit(X_train,y_train.ravel())

# apply it to the test data and compute the fitted values
y_pred = mlp.predict(X_test)


# asses model performance
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# plot the confusion matrix
mat = confusion_matrix(y_pred, y_test)
names = np.unique(y_test)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()

# yes!