import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# load the data
dataset = pd.read_csv('petrol_consumption.csv')

# have a look at the top lines
dataset.head()

# split the label from the features
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# divide data into test and training samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# random forest with 100 and 1000 trees
regressor100 = RandomForestRegressor(n_estimators=100, # this is the number of trees in the forest
                                  random_state=0)  # this sets the seed to make this replicable

regressor1000 = RandomForestRegressor(n_estimators=1000, 
                                  random_state=0)  


# fit it to the training data
regressor100.fit(X_train, y_train)
regressor1000.fit(X_train, y_train)

# compute the prediction
y_pred100 = regressor100.predict(X_test)
y_pred1000 = regressor1000.predict(X_test)

# evaluate
print('Root Mean Squared Error w 100 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred100)))
print('Root Mean Squared Error w 1000 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1000)))

# now we also change m
regressor100 = RandomForestRegressor(n_estimators=100,
                                     max_features = 1, # m (max number of features used in a given tree)
                                     random_state=0)

regressor100.fit(X_train, y_train)
y_pred100 = regressor100.predict(X_test)

print('Root Mean Squared Error w 100 trees:', np.sqrt(metrics.mean_squared_error(y_test, y_pred100)))

# feature importance plot
importance = regressor100.feature_importances_

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



# Boosting example code
from sklearn.ensemble import AdaBoostClassifier

boosting = AdaBoostClassifier(n_estimators=1000, random_state=0, learning_rate=0.01)

boosting.fit(X_train, y_train)  
y_predboosting = boosting.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predboosting)))


# Gradient descent boosting example
model = GradientBoostingRegressor(n_estimators=1000, random_state=0, learning_rate=0.01)

model.fit(X_train, y_train)  
y_pred = model.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# XGBoost example code
model = XGBClassifier(n_estimators=1000, random_state=0, learning_rate=0.01)
model.fit(X_train, y_train)

y_predbxg = model.predict(X_test)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predbxg)))