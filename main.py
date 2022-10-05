#IMPORTS
from asyncore import write
import os
from os.path import exists
import random
from turtle import clear
import csv

#NUMPY
from numpy import unique, where

#SciKit-Learn
from sklearn.datasets import make_classification, make_blobs, make_circles, make_moons, make_s_curve, make_swiss_roll
import sklearn.cluster as skCluster
from sklearn.metrics import rand_score

#MATPLOT
from matplotlib import pyplot as plt

#PICKLE
import pickle as pkl


def k_create(k_state:int, db_state=0, n_samples=10):
    X,y = make_classification(n_samples=1500, random_state=10, class_sep=4.5)

    kmeans = skCluster.KMeans(n_clusters=4, random_state=k_state, n_init=1)
    kmeans.fit(X)

    # assign a cluster to each example
    yhat = kmeans.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)


    # SAVING DATA
    pkl.dump(kmeans, open("results/kmeans"+str(k_state)+".pkl", "wb"))
    pkl.dump(X, open("results/kdb"+str(k_state)+".pkl", "wb"))

def k_load(state:int):
    #LOAD DATA
    kmeans = pkl.load(open("results/kmeans"+str(state)+".pkl", "rb"))
    X = pkl.load(open("results/kdb"+str(state)+".pkl","rb"))

    yhat = kmeans.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)

    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()
    
def DB_create(DB_state:int, db_state=0, n_samples=10):
    X, _ = make_blobs(n_samples=1500, n_features=10,random_state=0, cluster_std=1.3)

    # define the model
    dbscan = skCluster.DBSCAN(eps=2, min_samples=4)
    # fit model and predict clusters
    yhat = dbscan.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    

    # SAVING DATA
    pkl.dump(dbscan, open("results/DBSCAN"+str(DB_state)+".pkl", "wb"))
    pkl.dump(X, open("results/db"+str(DB_state)+".pkl", "wb"))

def DB_load(state:int):
    dbscan = pkl.load(open("results/DBSCAN"+str(state)+".pkl", "rb"))
    X = pkl.load(open("results/db"+str(state)+".pkl","rb"))

    yhat = dbscan.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)

    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
    plt.show()

def randIndex(state1:int, state2:int, k=True):
    if k:
        k1 = pkl.load(open("results/kmeans"+str(state1)+".pkl", "rb"))
        k2 = pkl.load(open("results/kmeans"+str(state2)+".pkl", "rb"))
        return rand_score(k1.labels_,k2.labels_)
    else:
        d1 = pkl.load(open("results/DBSCAN"+str(state1)+".pkl", "rb"))
        d2 = pkl.load(open("results/DBSCAN"+str(state2)+".pkl", "rb"))
        return rand_score(d1.labels_,d2.labels_)

def clear_dir(path:str):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

def make_csv(file:str):
    # if(not exists(file)):
        with open(file, 'w', newline='') as csvfile:
            fieldnames = ['s1', 's2', 'rscore']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        csvfile.close()
    # else:
    #     open(file, 'w').close()
    #     print('file already exists :D it has been cleared')

def save_results(s1, s2, rscore, file):
    with open(file, 'a', newline='') as csvfile:
        fieldnames = ['s1', 's2', 'rscore']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'s1': s1, 's2': s2, 'rscore': rscore})
    csvfile.close()

# logic for the experiment
def main():

    trials = 100
    # initialisations for the algorithms
    db_state = 0
    n_samples = 1500
    
    kfile = 'Kswiss_reults.csv'
    make_csv(file=kfile)
    for n in range(0,trials):
        state1 = random.randint(0,100)
        state2 = random.randint(0,100)
        k_create(state1, db_state=db_state, n_samples=n_samples)
        k_create(state2, db_state=db_state, n_samples=n_samples)
        r = randIndex(state1=state1, state2=state2, k=True)
        save_results(s1=state1, s2=state2, rscore=r, file=kfile)
        clear_dir('results')

    dbfile = 'DBswiss_results.csv'
    make_csv(file=dbfile)
    for n in range(0,trials):
        state1 = random.randint(0,100)
        state2 = random.randint(0,100)
        DB_create(state1, db_state=db_state, n_samples=n_samples)
        DB_create(state2, db_state=db_state, n_samples=n_samples)
        r = randIndex(state1=state1, state2=state2, k=False)
        save_results(s1=state1, s2=state2, rscore=r, file=dbfile)
        clear_dir('results')

# Used to test methods without destroying logic of main()
def test():
    k_create(40,db_state=0, n_samples=1500)
    k_load(40)
    k_create(45,db_state=0, n_samples=1500)
    k_load(45)

if __name__ == "__main__":
    test()
