# load the mglearn package (you might need to install this first!)
# import mglearn

# this is just to visualize how the decision boundary changes with the number of neighbors
# this is NOT the implementation of the classifier!

# for the first graph with 1 neighbor
# mglearn.plots.plot_knn_classification(n_neighbors=1)

# for the second graph with 3 neighbors
# mglearn.plots.plot_knn_classification(n_neighbors=3)


# now let's do this ourselves with the example from class

# here is some data on users' average rating, descretized to three levels (low, medium, and high)
rating = ['low','low','medium','high','high','high','medium','low','low', 'high','low','medium','medium','high']

# our second feature is their experience on the board
experience = ['2 months','2 months','2 months','4 months','8 months','8 months','8 months','4 months','8 months','4 months','4 months','4 months','2 months','4 months']

# and here is our label
message = ['spam','spam','no spam','no spam','no spam','spam','no spam','spam','no spam','no spam','no spam','no spam','no spam','spam']

# import the preprocessing and KNN packages from sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# define the function that will turn text data into numerical data
le = preprocessing.LabelEncoder()

# now we pre-process our data which the client gave us in text format, so we need to convert it to numerical format
# in other words: we assign numbers to 'low', 'medium', etc.
rating_encoded=le.fit_transform(rating)

# this is what it looks like after the transformation
print(rating_encoded)

# do the same for experience
experience_encoded=le.fit_transform(experience)

# converting string labels into numbers
label=le.fit_transform(message)

# now put our features into a list that the classifier can work with
features=list(zip(rating_encoded,experience_encoded))

# to see what list(zip()) does, type
list(zip(rating,experience))

# define the model and number of neighbors we want to use
model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(features,label)

# now we give the model some input and see what it predicts
# for instance, let's say predict([[0,1]]), where 0 = high user ranking and 1 = 4 months on the board
predicted = model.predict([[0,1]])
print(predicted)

# the prediction 0 means "no spam" from this user