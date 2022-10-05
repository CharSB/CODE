#NUMPY
from numpy import unique, where

#DATASETS
import sklearn.datasets as ds

#MATPLOT
from matplotlib import pyplot as plt

## Creates Dataset
X,y = ds.make_swiss_roll(n_samples=1500, random_state=0, noise=1)

## Plots ^^
plt.scatter(
    X[:, 0], X[:, 2],
    c='white', marker='o',
    edgecolor='black', s=50
)
plt.show()