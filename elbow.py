from numpy import unique, where
import sklearn.datasets as ds
import sklearn.cluster as skCluster
from matplotlib import pyplot as plt

## Creates Dataset
X,y = ds.make_swiss_roll(n_samples=1500, random_state=0, noise=1)

# taken from *
# https://stackabuse.com/k-means-clustering-with-scikit-learn/

wcss = [] 
for number_of_clusters in range(1, 11): 
    kmeans = skCluster.KMeans(n_clusters = number_of_clusters, random_state = 42)
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)
print(wcss)

ks = [1, 2, 3, 4, 5 , 6 , 7 , 8, 9, 10]
plt.plot(ks, wcss)
plt.show()