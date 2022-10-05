#IMPORTS
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

#LOAD DATA
kmeans = pkl.load(open("kmean.pkl", "rb"))
X = pkl.load(open("db.pkl","rb"))

yhat = kmeans.predict(X)
# retrieve unique clusters
clusters = unique(yhat)

for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
plt.show()