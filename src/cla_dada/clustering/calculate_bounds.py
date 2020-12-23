import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from models_config import clustering_config
from cla_dada.predictor.predictor import get_metrics_to_evaluate_discarding, get_metrics_to_evaluate_outliers

logger = logging.getLogger('discarding')
hdlr = logging.FileHandler('../../../logs/discarding.log')
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

def _outlier_bounds(center, points):
    '''
    This function calculate the bounds by which a data point is evaluated as not outlier

    Parameters:
        center (array): the center of the cluster
        points (dataframe): the points in the clusters

    Returns:
        outlier_lower_bound (float): 
        outlier_upper_bound (float)
    '''
    distances = []
    for index, row in points.iterrows():
        dist = np.linalg.norm(row.values - center)
        distances.append(dist)
    sorted(distances)
    # q1 = first quartile, q3 = third quartile
    q1, q3 = np.percentile(distances, [25, 75])
    # iqr = interquartile range
    iqr = q3 - q1
    outlier_lower_bound = q1 - (1.5 * iqr)
    outlier_upper_bound = q3 + (1.5 * iqr)
    return outlier_lower_bound, outlier_upper_bound

def _find_bounds_without_outliers(center, points):
    '''
    This function calculate the bounds of the cluster which its center is center and its data point are points

    Parameters:
        center (array): the center of the cluster
        points (dataframe): the points in the clusters

    Returns:
        (dictionary): a dictionary which contains the center, the lower bound and the upper bound of the cluster
        outliers (int): the number of outliers in the cluster
    '''
    cluster_lower_bound = math.nan
    cluster_upper_bound = math.nan

    outlier_lower_bound, outlier_upper_bound = _outlier_bounds(center, points)
    outliers = 0
    for index, row in points.iterrows():
        dist = np.linalg.norm(row.values - center)
        if dist >= outlier_lower_bound and dist <= outlier_upper_bound:
            if math.isnan(cluster_lower_bound):
                cluster_lower_bound = dist
            elif dist < cluster_lower_bound:
                cluster_lower_bound = dist
            if math.isnan(cluster_upper_bound):
                cluster_upper_bound = dist
            elif dist > cluster_upper_bound:
                cluster_upper_bound = dist
        else:
            outliers += 1

    return {
        'center': center,
        'lower_bound': cluster_lower_bound,
        'upper_bound': cluster_upper_bound
    }, outliers

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

def delete_small_clustars_and_calculate_bounds_without_outliers(data, clustering, percentage_to_discard):
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
    total_outliers = 0

    threshold = _calculate_threshold(grouped, len(data), percentage_to_discard)

    for key, group in grouped.__iter__():
        if len(group) >= threshold:
            bounds, cluster_outliers = _find_bounds_without_outliers(cluster_centers[key], group.iloc[:, 0:-1])
            result_centers.append(cluster_centers[key])
            result_clusters_bounds.append(bounds)
            total_outliers += cluster_outliers
        else:
            discarded_clusters_count += 1
            discarded_transactions_count += len(group)

    logger.info('Delete {0}% of training set with: threshold = {1}, discarded transactions = {2}, discarded clusters = {3}'.format(percentage_to_discard, threshold, discarded_transactions_count, discarded_clusters_count))
    logger.info('Discarded {0} outliers, equals to {1:.2f}% of training set'.format(total_outliers, ((total_outliers * 100) / len(data))))
    return np.array(result_centers), result_clusters_bounds

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
        tpr, tnr = get_metrics_to_evaluate_discarding(centers, bounds)
        lst_tpr.append(tpr)
        lst_tnr.append(tnr)
        lst_percentage.append(percentage_to_discard)
        if percentage_to_discard != last_percentage:
            logger.info('Discarding percentage: {0}%, TPR: {1:.5f}, TNR: {2:.5f}'.format(percentage_to_discard, tpr, tnr))
        else:
            logger.info('Discarding percentage: {0}%, TPR: {1:.5f}, TNR: {2:.5f}\n'.format(percentage_to_discard, tpr, tnr))
        percentage_to_discard += step_size
    _plot_tpr_and_tnr(lst_percentage, lst_tpr, lst_tnr)

def evaluate_outliers(data, clustering):
    '''
    Evaluate predictor without outliers while percentage changes

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
        centers, bounds = delete_small_clustars_and_calculate_bounds_without_outliers(data, clustering, percentage_to_discard)
        tpr, tnr = get_metrics_to_evaluate_outliers(centers, bounds)
        lst_tpr.append(tpr)
        lst_tnr.append(tnr)
        lst_percentage.append(percentage_to_discard)
        if percentage_to_discard != last_percentage:
            logger.info('[WITHOUT outliers] Discarding percentage: {0}%, TPR: {1:.5f}, TNR: {2:.5f}'.format(percentage_to_discard, tpr, tnr))
        else:
            logger.info('[WITHOUT outliers] Discarding percentage: {0}%, TPR: {1:.5f}, TNR: {2:.5f}\n'.format(percentage_to_discard, tpr, tnr))
        percentage_to_discard += step_size
    _plot_tpr_and_tnr(lst_percentage, lst_tpr, lst_tnr)

if __name__ == '__main__':
    # load data
    data = pd.read_pickle('../../../pickle/processed/trainingset.pkl')
    # load clustering
    clustering = load('../../../models/clustering/kmeans.bin')
    logger.info('Number of clusters: {0}\n'.format(len(clustering.cluster_centers_)))
                
    # result of predictor for each discard's percentage
    evaluate_percentage_to_discard(data, clustering)

    # result of predictor for each discard's percentage without outliers
    evaluate_outliers(data, clustering)

    # save centers and bounds with a spefic percentage of discard
    percentage_to_discard = int(input('Enter the percentage of data to discard (from 0 to 10): '))
    to_save = input('Press y or Y to save model WITHOUT outliers (n or N otherwise): ')
    if to_save == 'y' or to_save == 'Y':
        print('Saving model WITHOUT outliers ...')
        model_without_outliers = delete_small_clustars_and_calculate_bounds_without_outliers(data, clustering, percentage_to_discard)
        np.save('../../../models/clustering/centers.npy', model_without_outliers[0])
        dump(model_without_outliers[1], '../../../models/clustering/bounds.bin', compress=True)
        logger.info('Model WITHOUT outliers has been saved\n')
    else:
        print('Saving model WITH outliers ...')
        model_with_outliers = delete_small_clustars_and_calculate_bounds(data, clustering, percentage_to_discard)
        np.save('../../../models/clustering/centers.npy', model_with_outliers[0])
        dump(model_with_outliers[1], '../../../models/clustering/bounds.bin', compress=True)
        logger.info('Model WITH outliers has been saved\n')