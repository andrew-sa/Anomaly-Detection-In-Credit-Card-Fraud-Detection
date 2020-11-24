import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans

def init_pool(dataset):
    global data
    global logger

    data = dataset
    
    logger = logging.getLogger('elbow_method')
    hdlr = logging.FileHandler('../../logs/elbow_method.log')
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)


def calculate_wcss(n_cluster):
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10, max_iter=300, random_state=0, verbose=1)
    # kmeans = MiniBatchKMeans(n_clusters=n_cluster, init='k-means++', n_init=10, max_iter=300, random_state=None, batch_size=n_cluster*10, verbose=1)
    kmeans.fit(data)
    wcss = kmeans.inertia_
    logger.info('clusters = {} is done; wcss = {}'.format(n_cluster, wcss))
    return wcss


def perform_elbow_method(dataset, first_n_cluster, last_n_cluster, step):
    print(dataset)
    print('Number of cpu: ', mp.cpu_count())
    n_process = int(mp.cpu_count() / 4)

    pool = mp.Pool(processes=n_process, initializer=init_pool, initargs=[dataset,])
    result = pool.map(calculate_wcss, range(first_n_cluster, last_n_cluster+1, step))
    pool.close()
    pool.join()

    plt.plot(range(first_n_cluster, last_n_cluster+1, step), result)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# if __name__ == '__main__':
#     dataset = pd.read_pickle('../pickle/trainingset.pkl')
#     dataset = dataset.iloc[:, 0:-1].values
#     perform_elbow_method(dataset, 100, 1000, 100)