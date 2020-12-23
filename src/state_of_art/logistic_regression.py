import logging
import numpy as np
import pandas as pd

from joblib import dump, load
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

logger = logging.getLogger('binary-classification')
hdlr = logging.FileHandler('../../logs/binary_classification.log')
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def _find_best_model(X_train, y_train, X_validate, y_true):
    '''
    Find the best logistic regression model

    Parameters:
        X_train (ndarray of shape (n_samples_train, n_features)): training set
        X_train (ndarray of shape (n_samples_train,)): target values of training set
        X_validation (ndarray of shape (n_samples_validate, n_features)): validation set
        y_true (ndarray of shape (n_samples_validate,)): correct target values of validation set

    Returns:
        best_model (LogisticRegression): the model with the higher True Positive Rate
    '''
    TOLERANCE = 0.0001
    MAX_ITER = 10000
    tuned_solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    tuned_C = [0.1, 1, 50, 100, 500, 1000]
    tuned_multi_class = ['ovr', 'multinomial']
    tuned_fit_intercept = [True, False]

    total_model = len(tuned_solver) * len(tuned_C) * len(tuned_multi_class) * len(tuned_fit_intercept)
    discarded_model = 1 * len(tuned_C) * 1 * len(tuned_fit_intercept)
    total_model -= discarded_model
    i_model = 1
    
    best_tnr = 0
    best_tpr = 0
    best_model = None
    for solver_value in tuned_solver:
        for C_value in tuned_C:
            for multi_class_value in tuned_multi_class:
                for fit_intercept_value in tuned_fit_intercept:
                    if (solver_value != 'liblinear' or (solver_value == 'liblinear' and multi_class_value != 'multinomial')):
                        print('Trying model {0}/{1} ...'.format(i_model, total_model))
                        i_model += 1
                        # train the model
                        model = LogisticRegression(
                            solver=solver_value, 
                            C=C_value, 
                            multi_class=multi_class_value, 
                            fit_intercept=fit_intercept_value,
                            tol=TOLERANCE, max_iter=MAX_ITER, dual=False, random_state=0, verbose=1, n_jobs=1)
                        model.fit(X_train, y_train)
                        # predict
                        y_pred = model.predict(X_validate)
                        y_prob = model.predict_proba(X_validate)

                        tnr, _, _, tpr = confusion_matrix(y_true=y_validate, y_pred=y_pred, normalize='true').ravel()
                        roc_auc = roc_auc_score(y_validate, y_prob[:, 1])
                        if tpr > best_tpr:
                            best_tpr = tpr
                            best_tnr = tnr
                            best_model = model
                            print('\nNEW BEST')
                        elif tpr == best_tpr and tnr > best_tnr:
                            best_tpr = tpr
                            best_tnr = tnr
                            best_model = model
                            print('\nNEW BEST')
                        print('TPR: {0}\tTNR: {1}\tROC AUC: {2}'.format(tpr, tnr, roc_auc))
    return best_model

def get_result(X_train, y_train, X_validate, y_validate, X_test, y_true):
    '''
    Predict label of test set and save the result

    Parameters:
        X_train (ndarray of shape (n_samples_train, n_features)): independent columns of training set
        y_train (ndarray of shape (n_samples_train,)): labels of training set
        X_validate (ndarray of shape (n_samples_validate, n_features)): independent columns of validation set
        y_validate (ndarray of shape (n_samples_validate,)): labels of validation set
        X_test (ndarray of shape (n_samples_test, n_features)): independent columns of test set
        y_true (ndarray of shape (n_samples_test,)): correct target values
    '''
    model = _find_best_model(X_train, y_train, X_validate, y_validate)
    y_pred = model.predict(X_test)

    # print result
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
    rates_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
    logger.info('LOGISTIC REGRESSION')
    logger.info('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(conf_matrix[0], conf_matrix[1], conf_matrix[2], conf_matrix[3]))
    logger.info('TNR: {0:.5f}, TPR: {1:.5f}\n'.format(rates_matrix[0], rates_matrix[3]))
    # save the model
    dump(model, '../../models/binary_classifiers/logistic_regression.bin', compress=True)
    print(model.get_params())


def preprocessing(training_set, test_set):
    '''
    Pre-processing raw training set and raw test set

    Parameters:
        training_set (DataFrame): raw training set
        test_set (DataFrame): raw test set

    Returns:
        X_res (ndarray of shape (n_samples_train, n_features)): independent columns of training set after SMOTE oversampling
        y_res (ndarray of shape (n_samples_train,)): labels of training set after SMOTE oversampling
        X_validate (ndarray of shape (n_samples_validate, n_features)): independent columns of validation set
        y_validate (ndarray of shape (n_samples_validate,)): labels of validation set
        X_test (ndarray of shape (n_samples_test, n_features)): independent columns of test set
        y_test (ndarray of shape (n_samples_test,)): labels of test set
    '''
    X_train = training_set.iloc[:, 0:-1].values
    y_train = training_set.iloc[:, -1].values
    X_test = test_set.iloc[:, 0:-1].values
    y_test = test_set.iloc[:, -1].values

    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=10, shuffle=True, stratify=y_train)

    scaler = MinMaxScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    X_validate = scaler.transform(X_validate)
    X_train = scaler.transform(X_train)

    smote = SMOTE(sampling_strategy=1, random_state=0)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    del X_train, y_train

    return X_res, y_res, X_validate, y_validate, X_test, y_test

if __name__ == '__main__':
    # load data
    training_set = pd.read_pickle('../../pickle/raw_trainingset.pkl')
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    # pre-processing
    X_train, y_train, X_validate, y_validate, X_test, y_test = preprocessing(training_set, test_set)
    # result
    get_result(X_train, y_train, X_validate, y_validate, X_test, y_test)