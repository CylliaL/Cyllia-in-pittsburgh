# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 21:31:16 2024

@author: Xiang(Cyllia) Li
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier


## Load the data
path = 'E:/Pitts/ECON 2824 Big Data'
df = pd.read_csv(path+'/assignment3.csv')

## Describe the data
print(df.info())


## Clean the data 
selected_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
data = df[selected_columns]

data['Sex'] = data['Sex'].map({'male': True, 'female': False})
data['Sex'] = data['Sex'].astype(int)

mapping = {'S': 1, 'C': 2, 'Q': 3}
data['Embarked'] = data['Embarked'].map(mapping)

data.dropna(inplace=True)

## Basic summary statistics
summary_sta = data.describe()
print(summary_sta)


## Split the data into test and training sample
X = data.drop('Survived',axis = 1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Fit the tree in the test data
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)


## Evaluate performance in the test data set
print("Accuracy on test set (Classification Tree): {:.3f}".format(tree.score(X_test, y_test)))
# 0.813 So 81.3% accuracy


## Plot the tree
export_graphviz(tree, out_file="survive.dot", class_names=["Not Survived", "Survived"],
                feature_names=X.columns, impurity=True, filled=True)

# conda install -c conda-forge graphviz

import graphviz
with open("survive.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


## Overview of which features were the strongest predictors of death

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
Y_pred_prob = classifier.predict_proba(X_test)[:, 0] 

feature_importances = classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("(Classification Tree) Feature Importances (Strongest Predictors of Death):")
print(sorted_feature_importance_df)

# Sex 0.45 largest, Age 0.2 second


## Pruning the tree

# call the cost complexity command
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# for each alpha, estimate the tree
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    
# drop the last model because that only has 1 node
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# plot accuracy (in test and training) over alpha; first compute accuracy for each alpha
train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))
    
# second, plot it
plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()

# estimate the tree with the optimal alpha and display accuracy
clf_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.005)
clf_.fit(X_train,y_train)

print("Accuracy on test set: {:.3f}".format(clf_.score(X_test, y_test)))
#0.895 accuracy

# plot the pruned tree
export_graphviz(clf_, out_file="Survive.dot", class_names=["Not Survived", "Survived"],
    feature_names=X.columns, impurity=True, filled=True)

with open("survive.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


## Probit model
X_with_const = sm.add_constant(X)
probit_model = sm.Probit(y, X_with_const)
probit_result = probit_model.fit()

pro_death = probit_result.predict(X)

feature_importance_probit = probit_result.get_margeff().summary()

print(feature_importance_probit)


## Random forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=66)
rf_classifier.fit(X_train, y_train)
y_pred_random = rf_classifier.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
print("Accuracy Score (Random forest):", accuracy_random)
#0.85 accuracy

# feature importance 
feature_names = list(X_train.columns)
feature_importances_random = rf_classifier.feature_importances_
print("Feature Importances:")
for i, (feature, importance) in enumerate(zip(feature_names, feature_importances_random)):
    print(f"{feature}: {importance}")

# Sex 0.38 largest, age 0.235 second
