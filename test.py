#NUMPY
from numpy import unique, where

#DATASETS
from sklearn.datasets import make_classification, make_blobs

#CLUSTER
import sklearn.cluster as skCluster

#MATPLOT
from matplotlib import pyplot as plt

#PICKLE
import pickle as pkl


n_samples = 10
## Creates Dataset
X,y = make_blobs(n_samples=n_samples, n_features=100, centers=3, shuffle=True, random_state=42)

## Plots ^^
# plt.scatter(
#     X[:, 0], X[:, 1],
#     c='white', marker='o',
#     edgecolor='black', s=50
# )
# plt.show()


kmeans = skCluster.KMeans(n_clusters=3)
kmeans.fit(X)

# assign a cluster to each example
yhat = kmeans.predict(X)
# retrieve unique clusters
clusters = unique(yhat)

# print(X)
# print(y)

# SAVING DATA
pkl.dump(kmeans, open("kmean.pkl", "wb"))
pkl.dump(X, open("db.pkl", "wb"))


# for cluster in clusters:
#     # get row indexes for samples with this cluster
#     row_ix = where(yhat == cluster)
#     # create scatter of these samples
#     plt.scatter(X[row_ix, 0], X[row_ix, 1])
# plt.show()

