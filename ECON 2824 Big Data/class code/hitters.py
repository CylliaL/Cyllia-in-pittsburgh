
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

# set your working directory
path = 'E:/Pitts/ECON 2824 Big Data'
#path = 'D:/Dropbox/Pittsburgh/Teaching/MQE Machine Learning/lecture slides/lecture 6 - Shrinkage methods'

# open the data frame, drop missings, and the Player name column
df = pd.read_csv(path+'/Hitters.csv').dropna().drop('Player', axis = 1)

# describe the data
df.info()

# define binary variables
dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])

# define the label of interest
y = df.Salary

# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')

# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)

X.info()

alphas = 10**np.linspace(10,-2,100)*0.5
alphas

ridge = Ridge(normalize = True)
coefs = []

#normalize = True

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

ridge2 = Ridge(alpha = 4, normalize = True)  #, normalize = True
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred2))          # Calculate the test MSE

ridge3 = Ridge(alpha = 10**10, normalize = True)  #, normalize = True
ridge3.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred3 = ridge3.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge3.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred3))          # Calculate the test MSE

ridge2 = Ridge(alpha = 0, normalize = True)  #, normalize = True
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred = ridge2.predict(X_test)            # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred))           # Calculate the test MSE

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)  ##
ridgecv.fit(X_train, y_train)
ridgecv.alpha_

ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)  #, normalize = True
ridge4.fit(X_train, y_train)
mean_squared_error(y_test, ridge4.predict(X_test))

ridge4.fit(X, y)
pd.Series(ridge4.coef_, index = X.columns)

lasso = Lasso(max_iter = 10000, normalize = True)  #, normalize = True
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)  #, normalize = True
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
mean_squared_error(y_test, lasso.predict(X_test))

pd.Series(lasso.coef_, index=X.columns)