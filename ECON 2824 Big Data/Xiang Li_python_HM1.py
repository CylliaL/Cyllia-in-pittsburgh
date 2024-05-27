# library to work with data
import pandas as pd

# libraries to plot confusion matrices
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import seaborn as sns; sns.set()

# machine learning library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

## Import data
wine = pd.read_csv('E:/Pitts/ECON 2824 Big Data/assignment1_winequality.csv')

## Clean data
print(wine.isnull().sum())
wine.fillna(wine.median(), inplace=True)
wine['fixed acidity'] = wine['fixed acidity'].apply(lambda x: wine['fixed acidity'].median() if x < 0 else x)
wine['pH'] = wine['pH'].clip(lower=2.7, upper=4.0)

## Basic summary statistics
summary_sta = wine.describe()
print(summary_sta)

## Plot the Data
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 15))

for i, column in enumerate(wine.columns):
    row, col = divmod(i, 3)
    wine[column].hist(ax=axes[row, col], bins=15, color='darkred', alpha=0.7)
    axes[row, col].set_title(column)
    axes[row, col].axvline(wine[column].mean(), color='skyblue', linestyle='dashed', linewidth=2, label='Mean')
    axes[row, col].axvline(wine[column].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
    axes[row, col].text(0.665, 0.65, f"Mean: {wine[column].mean():.2f}\nMedian: {wine[column].median():.2f}", 
                        transform=axes[row, col].transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].legend()
    
plt.suptitle('Histograms of Wine Features', y=1)
plt.tight_layout()
plt.show()


## Quality Ratings
plt.figure(figsize=(10, 6))
sns.histplot(wine['quality'], kde=True, bins=range(1, 11), color='darkred')
plt.title('Distribution of Wine Quality Ratings')
plt.xlabel('Quality Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(wine['quality'].mean(), color='skyblue', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(wine['quality'].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.text(2, 1450, f"Mean: {wine['quality'].mean():.2f}", fontsize=11, color='black')
plt.text(2, 1400, f"Median: {wine['quality'].median():.2f}", fontsize=11, color='black')
plt.legend(loc='upper left', fontsize='small')
plt.show()

## Create the X, Y, Training and Test
x = wine.drop('quality', axis=1)
y = wine['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

## kNN
knn_model = KNeighborsClassifier()
knn_scores = cross_val_score(knn_model, x_train_scaled, y_train, cv=5)
print("kNN Cross-Validation Scores:", knn_scores)
print("Mean Accuracy:", knn_scores.mean())

knn_model.fit(x_train, y_train)
knn_predictions = knn_model.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("kNN Test Set Accuracy:", knn_accuracy)

## Naive Bayes
nb_model = GaussianNB()
nb_scores = cross_val_score(nb_model, x_train, y_train, cv=5, scoring='accuracy')
print("Naïve Bayes Cross-Validation Scores:", nb_scores)
print("Mean Accuracy:", nb_scores.mean())

nb_model.fit(x_train, y_train)
nb_predictions = nb_model.predict(x_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naïve Bayes Test Set Accuracy:", nb_accuracy)


           