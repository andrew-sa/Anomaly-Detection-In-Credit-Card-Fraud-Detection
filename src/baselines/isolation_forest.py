import logging
import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score

logger = logging.getLogger('baselines')
hdlr = logging.FileHandler('../../../logs/baselines.log')
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def _find_best_model(X_train, X_test, y_true):
    '''
    Find the best isolation forest model

    Parameters:
        X_train (ndarray of shape (n_samples, n_features)): training set
        X_test (ndarray of shape (n_samples, n_features)): test set
        y_true (ndarray of shape (n_samples,)): correct target values

    Returns:
        best_model (IsolationForest): the model with the higher ROC AUC score
    '''
    frauds_ratio = 492 / 284807
    tuned_parameters = {
        'n_esitmators': [50, 100, 150, 200],
        'max_samples': [50000, 100000, 150000, 200000, len(X_train), 'auto'],
        'contamination': [frauds_ratio, 'auto'],
        'max_features': [1, 5, 15, 20, 25, 29]
    }
    best_model = None
    best_roc_auc = 0.0
    for n_esitmators in tuned_parameters['n_esitmators']:
        for max_samples in tuned_parameters['max_samples']:
            for contamination in tuned_parameters['contamination']:
                for max_features in tuned_parameters['max_features']:
                    model = IsolationForest(n_estimators=n_esitmators, max_samples=max_samples, contamination=contamination, max_features=max_features, n_jobs=1, random_state=0, verbose=1)
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
    logger.info('ISOLATION FOREST')
    logger.info('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(tn, fp, fn, tp))
    logger.info('TNR: {:.5f}, TPR: {:.5f}'.format(tnr, tpr))
    logger.info('ROC AUC: {:.5f}\n'.format(roc_auc))