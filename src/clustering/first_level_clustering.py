import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.cluster import KMeans
from clustering.utils.elbow_method import perform_elbow_method

def run_kmeans(data, num_clusters):
    # kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=20, max_iter=400, random_state=0, verbose=1)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0, verbose=1)
    kmeans.fit(data)
    wcss = kmeans.inertia_
    print(wcss)
    # save first level clustering
    dump(kmeans, '../../models/firstlevel_clustering.bin', compress=True)


if __name__ == '__main__':
    # read dataset
    data = pd.read_pickle('../../pickle/trainingset.pkl')
    independent_cols = data.iloc[:, 0:-1].values

    # define the parameters of elbow method
    first_n_cluster = 100
    last_n_cluster = 1000
    step = 100

    # execute elbow method
    # perform_elbow_method(independent_cols, first_n_cluster, last_n_cluster, step)

    # run kmeans on data according to the number of clusters choose through the elbow method
    num_clusters = int(input('Enter the number of clusters: '))
    run_kmeans(independent_cols, num_clusters)