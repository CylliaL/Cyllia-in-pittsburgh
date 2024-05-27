# machine learning library
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

# library to work with data
import pandas as pd

# libraries to plot confusion matrices
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Import data
training = pd.read_csv('D:/Dropbox/Pittsburgh/Teaching/MQE Machine Learning/material/data/naivebayes_train.csv')
test = pd.read_csv('D:/Dropbox/Pittsburgh/Teaching/MQE Machine Learning/material/data/naivebayes_test.csv')

# Create the X, Y, Training and Test
xtrain = training.drop('ontopic', axis=1)
ytrain = training.loc[:, 'ontopic']

# set aside some test data
xtest = test.drop('ontopic', axis=1)
ytest = test.loc[:, 'ontopic']

# Define the naive Bayes Classifier
model = BernoulliNB()

# Train the model 
model.fit(xtrain, ytrain)

# Predict Output using the test data
pred = model.predict(xtest)

# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')

# store an array of predicted and actual labels
d = {'ontopic':ytest, 'prediction':pred}

# turn the array into a data frame and print it
output = pd.DataFrame(data=d)
output