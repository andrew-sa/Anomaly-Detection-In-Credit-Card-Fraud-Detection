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

def _find_best_model(X_train, X_validate, y_true):
    '''
    Find the best one-class support vector machine model

    Parameters:
        X_train (ndarray of shape (n_samples_train, n_features)): training set
        X_validate (ndarray of shape (n_samples_validate, n_features)): validation set
        y_true (ndarray of shape (n_samples_validate,)): correct target values of validation set

    Returns:
        best_model (OneClassSVM): the model with the higher True Positive Rate
    '''
    frauds_ratio = 492 / 284807
    tuned_parameters = {
        'nu': [frauds_ratio, 0.01, 0.1],
        'gamma': [0.0001, 0.001, 0.01, 0.1]
    }
    best_tnr = 0
    best_tpr = 0
    best_model = None
    for nu in tuned_parameters['nu']:
        for gamma in tuned_parameters['gamma']:    
            model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma, verbose=True)
            model.fit(X_train)
            y_pred = model.predict(X_validate)
            y_score = model.score_samples(X_validate)
            # fraud is negative class
            tnr, _, _, tpr = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
            roc_auc = roc_auc_score(y_true, y_score)
            if tnr > best_tnr:
                best_tnr = tnr
                best_tpr = tpr
                best_model = model
                print('\nNEW BEST')
            elif tnr == best_tnr and tpr > best_tpr:
                best_tnr = tnr
                best_tpr = tpr
                best_model = model
                print('\nNEW BEST')
            print('TPR: {0}\tTNR: {1}\tROC AUC: {2}'.format(tnr, tpr, roc_auc))
    dump(best_model, '../../models/baselines/oc_svm.bin', compress=True)
    print(best_model.get_params())
    return best_model

def get_predictions(X_train, X_validate, y_validate, X_test):
    '''
    Calculate predictions

    Parameters:
        X_train (ndarray of shape (n_samples_train, n_features)): independent columns of training set
        X_validate (ndarray of shape (n_samples_validate, n_features)): independent columns of validation set
        y_validate (ndarray of shape (n_samples_validate,)): labels of validation set
        X_test (ndarray of shape (n_samples_test, n_features)): independent columns of test set

    Returns:
        y_pred (ndarray of shape (n_samples_test,)): the predicted values (0 = negative class, 1 = positive class)
        y_score (ndarray of shape (n_samples_test,)): the (unshifted) scoring function of the samples.
    '''
    model = _find_best_model(X_train, X_validate, np.where(y_validate == 0, 1, -1))
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == 1, 0, 1)
    y_score = model.score_samples(X_test)
    return y_pred, y_score


if __name__ == '__main__':
    # load training set and vaidation set
    training_set = pd.read_pickle('../../pickle/processed/trainingset.pkl')
    X_train = training_set.iloc[:, 0:-1].values
    validation_set = pd.read_pickle('../../pickle/processed/validationset.pkl')
    X_validate = validation_set.iloc[:, 0:-1].values
    y_validate = validation_set.iloc[:, -1].values

    # load and transform raw test set
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    X_test = test_set.iloc[:, 0:-1].values
    y_true = test_set.iloc[:, -1].values
    scaler = load('../../models/preprocessing/minmaxscaler.bin')
    X_test = scaler.transform(X_test)

    y_pred, _ = get_predictions(X_train, X_validate, y_validate, X_test)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
    tnr, fpr, fnr, tpr = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
    # roc_auc = roc_auc_score(np.where(y_true == 0, 1, -1), y_score)
    logger.info('ONE-CLASS SUPPORT VECTOR MACHINE')
    logger.info('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(tn, fp, fn, tp))
    logger.info('TNR: {0:.5f}, TPR: {1:.5f}\n'.format(tnr, tpr))
    # logger.info('ROC AUC: {:.5f}\n'.format(roc_auc))