import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from models_config import clustering_config
from predictor.predictor import get_tpr_and_tnr_to_evaluate_discarding

logger = logging.getLogger('discarding')
hdlr = logging.FileHandler('../../logs/discarding.log')
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def _convert_list(list_to_convert):
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

def _find_bounds(center, points):
    '''
    This function calculate the bounds of the cluster which its center is center and its data point are points

    Parameters:
        center (array): the center of the cluster
        points (dataframe): the points in the clusters

    Returns:
        (dictionary): a dictionary which contains the center, the lower bound and the upper bound of the cluster   
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

def _calculate_threshold(grouped, len_data, percentage_to_discard):
    '''
    Calculate the maximum length of the cluster to be discarded according to the percentage of data to discard

    Parameters:
        grouped (DataFrameGroupBy): data grouped by cluster
        len_data (int): number of data points
        percentage_to_discard (int): the percentage of data to discard
    
    Returns:
        (int): the maximum length of the cluster to be discarded
    '''

    discard_limit = math.floor(len_data * percentage_to_discard / 100)
    founded = False
    threshold = clustering_config['threshold']['first']
    step_size = clustering_config['threshold']['step_size']
    while False == founded:
        print('Trying threshold {0} ...'.format(threshold))
        discarded_transactions_count = 0
        for key, group in grouped.__iter__():
                if len(group) < threshold:
                    discarded_transactions_count += len(group)
        if (discarded_transactions_count > discard_limit):
            founded = True
        else:
            threshold += step_size

    return threshold - step_size

def delete_small_clustars_and_calculate_bounds(data, clustering, percentage_to_discard):
    '''
    Delete small clusters according to the percentage of data to discard and calculate the bounds of remaining clusters

    Parameters:
        data (DataFrame): data
        clustering (KMeans): the result of kmeans on the data
        percentage_to_discard (int): the percentage of data to discard
    
    Returns:
        (ndarray): centers of remaining clusters
        (list): bounds of remaining clusters
    '''

    result_centers = []
    result_clusters_bounds = []

    data = data.iloc[:, 0:-1]
    # group data by first level labels
    data['Cluster'] = clustering.labels_
    grouped = data.groupby(data.Cluster)
    cluster_centers = clustering.cluster_centers_

    discarded_clusters_count = 0
    discarded_transactions_count = 0

    threshold = _calculate_threshold(grouped, len(data), percentage_to_discard)

    for key, group in grouped.__iter__():
        if len(group) >= threshold:
            bounds = _find_bounds(cluster_centers[key], group.iloc[:, 0:-1])
            result_centers.append(cluster_centers[key])
            result_clusters_bounds.append(bounds)
        else:
            discarded_clusters_count += 1
            discarded_transactions_count += len(group)

    logger.info('Delete {0}% of training set with: threshold = {1}, discarded transactions = {2}, discarded clusters = {3}'.format(percentage_to_discard, threshold, discarded_transactions_count, discarded_clusters_count))
    return np.array(result_centers), result_clusters_bounds

def _plot_tpr_and_tnr(lst_percentage, lst_tpr, lst_tnr):
    '''
    Plot true positive rates and true negative rates

    Parameters:
        lst_percentage (list): x-values (parcentages)
        lst_tpr (list): y1-values (true positive rates)
        lst_tnr (list): y2-values (true negative rates)
    '''

    plt.plot(lst_percentage, lst_tpr, color='blue', marker='o', label='TPR')
    plt.plot(lst_percentage, lst_tnr, color='red', marker='o', label='TNR')
    for x, y1, y2 in zip(lst_percentage, lst_tpr, lst_tnr):
        label_y1 = '{:.3f}'.format(y1)
        plt.annotate(label_y1, (x, y1), textcoords='offset points', xytext=(0, -15), ha='center')
        label_y2 = '{:.3f}'.format(y2)
        plt.annotate(label_y2, (x, y2), textcoords='offset points', xytext=(0, -15), ha='center')
    plt.xticks(lst_percentage)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.xlabel('Percentage of discarded transactions')
    plt.ylabel('Rate')
    plt.show()

def evaluate_percentage_to_discard(data, clustering):
    '''
    Evaluate predictor while percentage changes

    Parameters:
        data (DataFrame): data
        clustering (KMeans): the result of kmeans on the data
    '''

    percentage_to_discard = clustering_config['percentage']['first']
    last_percentage = clustering_config['percentage']['last']
    step_size = clustering_config['percentage']['step_size']
    lst_tpr = []
    lst_tnr = []
    lst_percentage = []
    while percentage_to_discard <= last_percentage:
        centers, bounds = delete_small_clustars_and_calculate_bounds(data, clustering, percentage_to_discard)
        tpr, tnr = get_tpr_and_tnr_to_evaluate_discarding(centers, bounds, 20)
        lst_tpr.append(tpr)
        lst_tnr.append(tnr)
        lst_percentage.append(percentage_to_discard)
        logger.info('Discarding percentage: {}%, TPR: {:.5f}, TNR: {:.5f}'.format(percentage_to_discard, tpr, tnr))
        percentage_to_discard += step_size
    _plot_tpr_and_tnr(lst_percentage, lst_tpr, lst_tnr)

if __name__ == '__main__':
    # load data
    data = pd.read_pickle('../../pickle/trainingset.pkl')
    # load first level clustering
    clustering = load('../../models/clustering/kmeans.bin')
    logger.info('Number of clusters: {0}'.format(len(clustering.cluster_centers_)))

    # result of predictor for each discard's percentage
    evaluate_percentage_to_discard(data, clustering)

    # save centers and bounds with a spefic discard's percentage
    percentage_to_discard = int(input('Enter the percentage of data to discard (from 0 to 10): '))
    centers, bounds = delete_small_clustars_and_calculate_bounds(data, clustering, percentage_to_discard)
    # save centers and bounds
    np.save('../../models/clustering/centers.npy', centers)
    dump(bounds, '../../models/clustering/bounds.bin', compress=True)