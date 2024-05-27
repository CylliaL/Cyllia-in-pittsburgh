
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

# Create dataset with 3 random cluster centers and 1000 datapoints
x, y = make_blobs(n_samples = 1000, centers = 3, n_features=2, shuffle=True, random_state=31)

# scatter plot
plt.scatter(x[:, 0], x[:, 1], s=50)
plt.show()


# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
  # define an empty object that will keep the sum of squared errors (sse)
  sse = []
  
  # then for each value of k that we specify
  for k in range(1, kmax+1):
      
    # compute the average value of the points in a given cluster by first finding the centroids of the potential clusters
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

M = calculate_WSS(x, 10)
K = [1,2,3,4,5,6,7,8,9,10]

plt.plot(M, K)
plt.show()

# compute the silhouette score
sil = []
kmax = 10
K = [2,3,4,5,6,7,8,9,10]

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(x)
  labels = kmeans.labels_
  sil.append(silhouette_score(x, labels, metric = 'euclidean'))
  
plt.plot(K, sil)
plt.show()


# perform the kmeans clustering with k =3
kmeans = KMeans(n_clusters = 3).fit(x)
labels = kmeans.labels_

plt.scatter(x[:, 0], x[:, 1], c=labels)
plt.show()


# let's look at a less trivial example
centers = [[2, 1], [-1, 1], [-1, -2]]
X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=0.8, random_state=31)

# scatter plot
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

# compute the silhouette score
sil = []
kmax = 10
K = [2,3,4,5,6,7,8,9,10]

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(X)
  labels = kmeans.labels_
  sil.append(silhouette_score(X, labels, metric = 'euclidean'))
  
plt.plot(K, sil)
plt.show()

# perform the kmeans clustering with k =3
kmeans = KMeans(n_clusters = 3).fit(X)
labels = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()

# HIERARCHICAL CLUSTERING EXAMPLE

# load the customer data
customer_data = pd.read_csv('shopping.csv')

# keep relevant columns
data = customer_data.iloc[:, 3:5].values

# plot the dendogram
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))

# generate five hierarchical clusters
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

# what does the result look like?
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')