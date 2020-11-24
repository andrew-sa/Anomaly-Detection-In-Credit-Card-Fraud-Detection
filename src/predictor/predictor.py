import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from data_preparation.preprocessing import min_max_normalization, pca_dimensionality_reduction

def find_knn_centers(data_point, num_neighbors):
    centers = np.load('../../models/secondlevel_clustering/centers.npy')
    knn_model = NearestNeighbors(n_neighbors=num_neighbors, metric='minkowski', p=2, n_jobs=1).fit(centers)
    k_neighbors = knn_model.kneighbors([data_point])
    return k_neighbors[1].flatten()

def get_prediction(data_point, indices_k_nearest_centers):
    bounds = np.load(file='../../models/secondlevel_clustering/bounds.npy', allow_pickle=True)
    prediction = 1
    index = 0
    while 1 == prediction and index < len(bounds): # len(bounds) is equal to bounds.shape[0]
        cluster_info = bounds[index]
        dist = np.linalg.norm(data_point - cluster_info['center'])
        if dist >= cluster_info['lower_bound'] and dist <= cluster_info['upper_bound']:
            prediction = 0
        index += 1
    return prediction

if __name__ == '__main__':
    # load raw test set
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    X_test = test_set.iloc[:, 0:-1]
    y_true = test_set.iloc[:, -1].values
    X_test = min_max_normalization(X_test)
    X_test = pca_dimensionality_reduction(X_test)

    y_pred = []
    num_neighbors = 30
    for data_point in X_test.values:
        indices_k_nearest_centers = find_knn_centers(data_point, num_neighbors)
        y_pred.append(get_prediction(data_point, indices_k_nearest_centers))
    tn, fp, fn, tp = confusion_matrix(y_true, np.array(y_pred)).ravel()
    print('tn: {0}, fp: {1}, fn: {2}, tp: {3}'.format(tn, fp, fn, tp))


    
