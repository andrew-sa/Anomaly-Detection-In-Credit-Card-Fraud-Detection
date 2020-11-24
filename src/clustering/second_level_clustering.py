import math
import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.cluster import KMeans

count = 0

def find_bounds(center, points):
    '''
    This function calculate the bounds of the cluster which its center is center and its data point are points

    Parameters:
        center (array): the center of the cluster
        points (dataframe): the points in the clusters

    Returns:
        (dictionary): a dictionary which contains the lower bound and the upper bound of the cluster   
    '''

    lower_bound = math.nan
    upper_bound = math.nan

    for index, row in points.iterrows():
        dist = np.linalg.norm(row.values - center)
        if math.isnan(lower_bound):
            lower_bound = dist
        elif dist < lower_bound:
            lower_bound = dist
        if math.isnan(upper_bound):
            upper_bound = dist
        elif dist > upper_bound:
            upper_bound = dist

    return {
        'center': center,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    

def perform_second_level_clustering(data, num_clusters):
    '''
    Perform kmeans on data and calculate clusters' bounds

    Parameters:
        data (dataframe): data to divede in clusters
        num_clusters (int): the number of clusters to form

    Returns:
        (list): the centers of the formed clusters
        (list): the bounds of the formed clusters
    '''
    global count

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=20, max_iter=400, random_state=0, verbose=1)
    kmeans.fit(data.values)
    print(kmeans.inertia_)
    # group data by cluster labels
    data['SecondLevel'] = kmeans.labels_
    grouped = data.groupby(data.SecondLevel)
    # for each formed cluster calculate its bounds
    clusters_bounds = []
    center_number = 0
    for center in kmeans.cluster_centers_:
        if len(grouped.get_group(center_number)) > 10: 
            clusters_bounds.append(find_bounds(center, grouped.get_group(center_number).iloc[:, 0:-1]))
            center_number +=1
        else:
            count += 1
    return kmeans.cluster_centers_, clusters_bounds

def convert_list(list_to_convert):
    '''
    Convert a non-flat list in a ndarray

    Parameters:
        list_to_convert (list): the list to convert
    
    Returns:
        (ndarray): the converted list
    '''

    # flat the list
    flat_list = []
    for sublist in list_to_convert:
        for item in sublist:
            flat_list.append(item)
    # convert to numpy ndarray
    return np.array(flat_list)

if __name__ == '__main__':
    # choose number of clusters
    num_clusters = 20

    # load data
    data = pd.read_pickle('../../pickle/trainingset.pkl')
    # load first level clustering
    first_level = load('../../models/firstlevel_clustering.bin')

    count = 0
    centers = []
    clusters_bounds = []
    data = data.iloc[:, 0:-1]
    # group data by first level labels
    data['FirstLevel'] = first_level.labels_
    grouped = data.groupby(data.FirstLevel)
    first_level_centers = first_level.cluster_centers_
    for key, group in grouped.__iter__():
        if len(group) > 10:
            res_bounds = find_bounds(first_level_centers[key], group.iloc[:, 0:-1])
            centers.append(first_level_centers[key])
            clusters_bounds.append(res_bounds)
        else:
            count += 1
    np.save('../../models/secondlevel_clustering/centers.npy', np.array(centers))
    np.save('../../models/secondlevel_clustering/bounds.npy', np.array(clusters_bounds))
    print(np.array(centers))
    print(np.array(clusters_bounds))
    print(count)

    '''
    # for each first level cluster calculate second level clustering and its bounds
    for key, group in grouped.__iter__():
        res_centers, res_bounds = perform_second_level_clustering(group.iloc[:, 0:-1], num_clusters)
        centers.append(res_centers)
        clusters_bounds.append(res_bounds)
    # convert centers and bounds list to numpy array
    ndarray_centers = convert_list(centers)
    ndarray_bounds = convert_list(clusters_bounds)
    # save centers and bounds
    np.save('../../models/secondlevel_clustering/centers.npy', ndarray_centers)
    np.save('../../models/secondlevel_clustering/bounds.npy', ndarray_bounds)
    print(ndarray_centers)
    print(ndarray_bounds)
    print(count)
    '''