from numpy import mean
from numpy import std
import pandas as pd
from sklearn.datasets import load_wine
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# load data
wine = load_wine()

# define features and label
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Categorical.from_codes(wine.target, wine.target_names)

# split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# define the LDA model
lda = LDA()

# fit the model to the training data
model = lda.fit(X_train, y_train)

# use it for prediction in the test data
y_pred = model.predict(X_test)

# compute accuracy
print(str(accuracy_score(y_test, y_pred)))

# print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# for the SVM example, load the breast cancer data
cancer = datasets.load_breast_cancer()

# split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test

# define the SVM and select a kernel
clf = svm.SVC(kernel='linear')

# train the model
clf.fit(X_train, y_train)

# predict the label and evaluate the prediction
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))