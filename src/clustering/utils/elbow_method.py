import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from models_config import clustering_config

def _init_pool(dataset):
    '''
    Pool initializer
    '''
    
    global data
    global logger

    data = dataset
    
    logger = logging.getLogger('clustering')
    hdlr = logging.FileHandler('../../logs/clustering.log')
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)


def _calculate_wcss(n_cluster):
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', n_init=5, max_iter=300, random_state=0, verbose=1)
    kmeans.fit(data)
    wcss = kmeans.inertia_
    logger.info('clusters = {}, wcss = {:.5f}'.format(n_cluster, wcss))
    return wcss


def perform_elbow_method(dataset):
    '''
    Perform elbow method using the numbers of clusters specified in the configuration files and plot the result

    Parameters:
        dataset (ndarray): data to cluster
    '''

    first_n_cluster = clustering_config['elbow_method']['first']
    last_n_cluster = clustering_config['elbow_method']['last']
    step_size = clustering_config['elbow_method']['step_size']
    n_process = int(mp.cpu_count() / (mp.cpu_count() / 2))

    pool = mp.Pool(processes=n_process, initializer=_init_pool, initargs=[dataset,])
    result = pool.map(_calculate_wcss, range(first_n_cluster, last_n_cluster+1, step_size))
    pool.close()
    pool.join()

    plt.plot(range(first_n_cluster, last_n_cluster+1, step_size), result, marker='o')
    for x, y in zip(range(first_n_cluster, last_n_cluster+1, step_size), result):
        label_y = '{:.2f}'.format(y)
        plt.annotate(label_y, (x, y), textcoords='offset points', xytext=(0, -15), ha='center')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.gca().set_ylim([0, None])
    plt.show()

# if __name__ == '__main__':
#     dataset = pd.read_pickle('../pickle/trainingset.pkl')
#     dataset = dataset.iloc[:, 0:-1].values
#     perform_elbow_method(dataset)