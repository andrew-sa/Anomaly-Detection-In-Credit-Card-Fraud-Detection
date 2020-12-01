import logging
import numpy as np
import pandas as pd
import seaborn as sn
import multiprocessing as mp
from matplotlib import colors
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, roc_auc_score
from data_preparation.preprocessing import min_max_normalization
from models_config import predictor_config

logger = logging.getLogger('predictor')
hdlr = logging.FileHandler('../../logs/predictor.log')
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def _find_knn_centers(data_point):
    '''
    Find k nearest centers of the data point

    Parameters:
        data_point (ndarray): the data point
        num_neighbors (int): value of k
        centers (ndarray): clusters centers, if is None the method retrieves the ndarray from the disc
    
    Returns:
        (list): the list contains the indeces of the k nearest centers
    '''

    global num_neighbors
    global centers
    if centers is None:
        centers = np.load('../../models/clustering/centers.npy')

    knn_model = NearestNeighbors(n_neighbors=num_neighbors, metric='minkowski', p=2, n_jobs=1).fit(centers)
    k_neighbors = knn_model.kneighbors([data_point])
    return k_neighbors[1].flatten().tolist()

def get_prediction(row_tuple):
    '''
    Calculate the prediction of the data point

    Parameters:
        row_tuple (tuple): (index, Series) pair, where the Series variable represents the data point
        centers (ndarray): clusters centers, if is None the method retrieves the ndarray from the disc
        bounds (list): clusters bounds, if is None the method retrieves the list from the disc

    Returns:
        prediction (int): the predicted value (0 = negative class, 1 = positive class)
    '''

    global bounds
    if bounds is None:
        bounds = load('../../models/clustering/bounds.bin')
    
    row_index = row_tuple[0]
    data_point = row_tuple[1].values

    indices_k_nearest_centers = _find_knn_centers(data_point)
    prediction = 1
    index = 0
    while 1 == prediction and index < len(bounds): # len(bounds) is equal to bounds.shape[0]
        if index in indices_k_nearest_centers:
            cluster_info = bounds[index]
            dist = np.linalg.norm(data_point - cluster_info['center'])
            if dist >= cluster_info['lower_bound'] and dist <= cluster_info['upper_bound']:
                prediction = 0
        index += 1
    print('Row {0} is done.'.format(row_index))
    return prediction

def _init_pool(number_neighbors, cluster_centers=None, cluster_bounds=None):
    '''
    Pool initializer
    '''

    global num_neighbors
    global centers
    global bounds
    num_neighbors = number_neighbors
    centers = cluster_centers
    bounds = cluster_bounds

def get_tpr_and_tnr_to_evaluate_discarding(cluster_centers, cluster_bounds, number_neighbors):
    '''
    This method is used to evaluate discarding percentages by calculates_bounds.py module.

    Parameters:
        cluster_centers (ndarray): centers of each cluster
        cluster_bounds (list): bounds of each cluster

    Returns:
        tpr (float): true positive rate
        tnr (float): true negative rate
    '''

    # load raw test set
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    X_test = test_set.iloc[:, 0:-1]
    y_true = test_set.iloc[:, -1].values
    X_test = min_max_normalization(X_test)

    # Multiporcesing using Pool
    n_process = mp.cpu_count()
    pool = mp.Pool(processes=n_process, initializer=_init_pool, initargs=[number_neighbors, cluster_centers, cluster_bounds,])
    y_pred = pool.map(get_prediction, X_test.iterrows())
    pool.close()
    pool.join()

    # return true positive rate (TPR) and true negative rate (TNR) 
    tnr, fpr, fnr, tpr = confusion_matrix(y_true=y_true, y_pred=np.array(y_pred), normalize='true').ravel()
    return tpr, tnr

def _plot_tpr_tnr_auc(k_values, tpr_values, tnr_values, roc_auc_values):
    '''
    Plot true positive rates, true negative rates and area under the ROC curve

    Parameters:
        k_values (list): x-values (k values for k-NN)
        tpr_values (list): y1-values (true positive rates)
        tnr_values (list): y2-values (true negative rates)
        roc_auc_values (list): y3-values (rou auc scores)
    '''

    plt.plot(k_values, tpr_values, color='blue', marker='o', label='TPR')
    plt.plot(k_values, tnr_values, color='red', marker='o', label='TNR')
    plt.plot(k_values, roc_auc_values, color='olive', marker='o', label='ROC AUC')
    for x, y1, y2, y3 in zip(k_values, tpr_values, tnr_values, roc_auc_values):
        label_y1 = '{:.3f}'.format(y1)
        plt.annotate(label_y1, (x, y1), textcoords='offset points', xytext=(0, -15), ha='center')
        label_y2 = '{:.3f}'.format(y2)
        plt.annotate(label_y2, (x, y2), textcoords='offset points', xytext=(0, -15), ha='center')
        label_y3 = '{:.3f}'.format(y3)
        plt.annotate(label_y3, (x, y3), textcoords='offset points', xytext=(0, -15), ha='center')
    plt.xticks(k_values)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Number of neighbors (k-NN)')
    plt.legend()
    plt.show()

def _plot_confusion_matrix(conf_matrix, number_neighbors):
    '''
    Plot confusion matrix

    Parameters:
        conf_matrix: ndarray of shape (n_classes, n_classes)
                    (Confusion matrix whose i-th row and j-th column entry indicates the number of samples 
                        with true label being i-th class and prediced label being j-th class.)
        number_neighbors (int): value of k (k-NN) used to obtain this confusion matrix
    '''

    df_conf_matrix = pd.DataFrame(conf_matrix, ['Genuine', 'Fraud'], ['Genuine', 'Fraud'])
    sn.heatmap(df_conf_matrix, annot=True, fmt='d')
    plt.title('k = {0}'.format(number_neighbors))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

def _plot_confusion_matrices(predictions):
    '''
    Plot confusion matrix
    
    Parameters:
        predictions (list): list of list composed of number of neighbors and predicted labels (int, list)
    '''
    labels = ['Genuine', 'Fraud']
    cmap = colors.ListedColormap(['white'])
    plt.figure()
    index = len(predictions)
    for number_neighbors, y_pred in predictions:
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=np.array(y_pred), normalize=None)
        plt.subplot(2, 3, index)
        ax = plt.gca()
        ax.imshow(conf_matrix, cmap=cmap)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, conf_matrix[i, j], ha='center', va='center')
        plt.title('k = {0}'.format(number_neighbors))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        index -= 1
    plt.tight_layout()
    plt.show()

def show_results(predictions):
    '''
    Plot and print results

    Parameters:
        predictions (list): list of list composed of number of neighbors and predicted labels (int, list)
    '''

    k_values = []
    tpr_values = []
    tnr_values = []
    roc_auc_values = []
    for number_neighbors, y_pred in predictions:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=np.array(y_pred), normalize=None).ravel()
        tnr, fpr, fnr, tpr = confusion_matrix(y_true=y_true, y_pred=np.array(y_pred), normalize='true').ravel()
        logger.info('k of k-NN = {0}.'.format(number_neighbors))
        logger.info('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(tn, fp, fn, tp))
        logger.info('TNR: {:.5f}, TPR: {:.5f}'.format(tnr, tpr))
        roc_auc = roc_auc_score(y_true, np.array(y_pred))
        logger.info('ROC AUC: {:.5f}\n'.format(roc_auc))
        k_values.append(number_neighbors)
        tpr_values.append(tpr)
        tnr_values.append(tnr)
        roc_auc_values.append(roc_auc)   
    _plot_tpr_tnr_auc(k_values, tpr_values, tnr_values, roc_auc_values)
    _plot_confusion_matrices(predictions)
    # for number_neighbors, y_pred in predictions:
    #     conf_matrix = confusion_matrix(y_true=y_true, y_pred=np.array(y_pred), normalize=None)
    #     _plot_confusion_matrix(conf_matrix, number_neighbors)


if __name__ == '__main__':
    # load raw test set
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    X_test = test_set.iloc[:, 0:-1]
    y_true = test_set.iloc[:, -1].values
    X_test = min_max_normalization(X_test)

    # calculate predictions for each value of k in k-NN
    predictions = []
    n_process = mp.cpu_count()
    for number_neighbors in predictor_config['n_neighbors']:
        result = []
        pool = mp.Pool(processes=n_process, initializer=_init_pool, initargs=[number_neighbors,])
        y_pred = pool.map(get_prediction, X_test.iterrows())
        result.append(number_neighbors)
        result.append(y_pred)
        predictions.append(result)
        pool.close()
        pool.join()

    show_results(predictions)
    
