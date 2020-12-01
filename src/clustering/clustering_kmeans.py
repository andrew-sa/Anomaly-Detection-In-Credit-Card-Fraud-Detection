import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.cluster import KMeans
from clustering.utils.elbow_method import perform_elbow_method

def run_kmeans(data, num_clusters):
    '''
    Perform kmeans and store the obtained model

    Parameters:
        data (dataframe): data to cluster
        num_clusters (int): number of clusters
    '''

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0, verbose=1)
    kmeans.fit(data)
    wcss = kmeans.inertia_
    print(wcss)
    # save clustering
    dump(kmeans, '../../models/clustering/kmeans.bin', compress=True)


if __name__ == '__main__':
    # read dataset
    data = pd.read_pickle('../../pickle/trainingset.pkl')
    independent_cols = data.iloc[:, 0:-1].values

    # execute elbow method
    perform_elbow_method(independent_cols)

    # run kmeans on data according to the number of clusters chosen through the elbow method
    num_clusters = int(input('Enter the number of clusters: '))
    run_kmeans(independent_cols, num_clusters)