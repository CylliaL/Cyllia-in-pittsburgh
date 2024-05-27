from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load data set
cancer = load_breast_cancer()

# split it into test and training (target is the label of interest, 
#                                  random_state sets a seed so the code is replicable; 
#                                  without it, the split will be random each time you run this)
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)


# define our basic tree classifier
tree = DecisionTreeClassifier(random_state=0)

# fit it to the training data
tree.fit(X_train, y_train)

# compute accuracy in the test data
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# plot the tree
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
    feature_names=cancer.feature_names, impurity=True, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# apply cost complexity pruning

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
clf_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.01)
clf_.fit(X_train,y_train)

print("Accuracy on test set: {:.3f}".format(clf_.score(X_test, y_test)))

# plot the pruned tree
export_graphviz(clf_, out_file="tree.dot", class_names=["malignant", "benign"],
    feature_names=cancer.feature_names, impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))