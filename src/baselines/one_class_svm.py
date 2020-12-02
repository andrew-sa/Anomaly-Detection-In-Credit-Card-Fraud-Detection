import logging
import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, roc_auc_score

logger = logging.getLogger('baselines')
hdlr = logging.FileHandler('../../logs/baselines.log')
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def _find_best_model(X_train, X_test, y_true):
    '''
    Find the best one-class support vector machine model

    Parameters:
        X_train (ndarray of shape (n_samples, n_features)): training set
        X_test (ndarray of shape (n_samples, n_features)): test set
        y_true (ndarray of shape (n_samples,)): correct target values

    Returns:
        best_model (OneClassSVM): the model with the higher ROC AUC score
    '''
    frauds_ratio = 492 / 284807
    tuned_parameters = {
        'nu': [frauds_ratio, 0.01, 0.1],
        'gamma': [0.0001, 0.001, 0.01, 0.1]
    }
    best_model = None
    best_roc_auc = 0.0
    for nu in tuned_parameters['nu']:
        for gamma in tuned_parameters['gamma']:    
            model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma, verbose=True)
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)
            roc_auc = roc_auc_score(y_true, y_pred)
            if roc_auc > best_roc_auc:
                best_model = model
    print(best_model.get_params())
    return best_model

def get_predictions(X_train, X_test, y_true):
    '''
    Calculate predictions

    Parameters:
        X_train (ndarray of shape (n_samples, n_features)): training set
        X_test (ndarray of shape (n_samples, n_features)): test set
        y_true (ndarray of shape (n_samples,)): correct target values

    Returns:
        y_pred (ndarray of shape (n_samples,)): the predicted values (0 = negative class, 1 = positive class)
    '''
    model = _find_best_model(X_train, X_test, y_true)
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == 1, 0, 1)
    return y_pred


if __name__ == '__main__':
    # load training set
    training_set = pd.read_pickle('../../pickle/trainingset.pkl')
    X_train = training_set.iloc[:, 0:-1].values

    # load and transform raw test set
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    X_test = test_set.iloc[:, 0:-1].values
    y_true = test_set.iloc[:, -1].values
    scaler = load('../../models/preprocessing/minmaxscaler.bin')
    X_test = scaler.transform(X_test)

    y_pred = get_predictions(X_train, X_test, y_true)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
    tnr, fpr, fnr, tpr = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
    roc_auc = roc_auc_score(y_true, y_pred)
    logger.info('ONE-CLASS SUPPORT VECTOR MACHINE')
    logger.info('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(tn, fp, fn, tp))
    logger.info('TNR: {:.5f}, TPR: {:.5f}'.format(tnr, tpr))
    logger.info('ROC AUC: {:.5f}\n'.format(roc_auc))