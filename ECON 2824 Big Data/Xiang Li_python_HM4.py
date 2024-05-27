import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics
from matplotlib import pyplot

# pip install xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


## Load the data
path = 'E:/Pitts/ECON 2824 Big Data'
df = pd.read_csv(path+'/assignment4_HRemployee_attrition.csv')

## Clean the data 
selected_columns = ['Age', 'Attrition', 'DailyRate', 'Education', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance', 'TotalWorkingYears']
df = df[selected_columns]

df['Gender'] = df['Gender'].map({'Male': True, 'Female': False})
df['Gender'] = df['Gender'].astype(int)
# Male = 1, Female = 0

df['Attrition'] = df['Attrition'].map({'Yes': True, 'No': False})
df['Attrition'] = df['Attrition'].astype(int)
# Yes = 1, No = 0

## summary statistics
df.describe().transpose()

## have the label in one column
target_column = ['Attrition'] 

## feature list
predictors = list(set(list(df.columns))-set(target_column))

## scale features to be in [0,1]
df[predictors] = df[predictors]/df[predictors].max()

## summary stats of scaled features
df.describe().transpose()

## divide data into test and training samples
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


#### MLPClassifier 
## set up the model with two hidden layers where the first layer has 10 nodes and the second has 5
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state = 4)

## fit it to the training data
mlp.fit(X_train,y_train.ravel())

## apply it to the test data and compute the fitted values
y_pred = mlp.predict(X_test)

## asses model performance
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
# Accuracy = 0.866

## plot the confusion matrix
mat = confusion_matrix(y_pred, y_test)
names = np.unique(y_test)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()

## let's see if the regularized network does better
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state = 4, alpha=0.001)

## fit it to the training data
mlp.fit(X_train,y_train.ravel())

## apply it to the test data and compute the fitted values
y_pred = mlp.predict(X_test)

## asses model performance
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
# Accuracy = 0.864

## plot the confusion matrix
mat = confusion_matrix(y_pred, y_test)
names = np.unique(y_test)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()


#### ensemble

## random forest with 100 and 1000 trees
regressor100 = RandomForestRegressor(n_estimators=100, # this is the number of trees in the forest
                                  random_state=0)  # this sets the seed to make this replicable

regressor1000 = RandomForestRegressor(n_estimators=1000, 
                                  random_state=0)



## fit it to the training data
regressor100.fit(X_train, y_train)
regressor1000.fit(X_train, y_train)

## compute the prediction
y_pred100 = regressor100.predict(X_test)
y_pred1000 = regressor1000.predict(X_test)

## evaluate
print('Root Mean Squared Error w 100 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred100)))
print('Root Mean Squared Error w 1000 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1000)))
# Root Mean Squared Error w 100 trees: 0.34215536644130035
# Root Mean Squared Error w 1000 trees: 0.33973578836456364

## now we also change m
regressor100 = RandomForestRegressor(n_estimators=100,
                                     max_features = 1, # m (max number of features used in a given tree)
                                     random_state=0)

regressor100.fit(X_train, y_train)
y_pred100 = regressor100.predict(X_test)

print('Root Mean Squared Error w 100 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred100)))
## Root Mean Squared Error w 100 trees: 0.3279753046397163

## feature importance plot
importance = regressor100.feature_importances_

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
## plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# Feature: 0, Score: 0.01314
# Feature: 1, Score: 0.04251
# Feature: 2, Score: 0.04752
# Feature: 3, Score: 0.11501
# Feature: 4, Score: 0.01734
# Feature: 5, Score: 0.09751
# Feature: 6, Score: 0.04701
# Feature: 7, Score: 0.05009
# Feature: 8, Score: 0.05182
# Feature: 9, Score: 0.04220
# Feature: 10, Score: 0.09791
# Feature: 11, Score: 0.10007
# Feature: 12, Score: 0.04655
# Feature: 13, Score: 0.12782
# Feature: 14, Score: 0.10351

# Boosting example code
from sklearn.ensemble import AdaBoostClassifier

boosting = AdaBoostClassifier(n_estimators=1000, random_state=0, learning_rate=0.01)

boosting.fit(X_train, y_train)  
y_predboosting = boosting.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predboosting)))
# Root Mean Squared Error: 0.3719166512336502

## Gradient descent boosting example
model = GradientBoostingRegressor(n_estimators=1000, random_state=0, learning_rate=0.01)

model.fit(X_train, y_train)  
y_pred = model.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Root Mean Squared Error: 0.33903642780106086

# XGBoost example code
model = XGBClassifier(n_estimators=1000, random_state=0, learning_rate=0.01)
model.fit(X_train, y_train)

y_predbxg = model.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predbxg)))
# Root Mean Squared Error: 0.392676726249301







