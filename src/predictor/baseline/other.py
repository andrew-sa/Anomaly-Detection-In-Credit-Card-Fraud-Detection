import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import IsolationForest

def get_predictions(X_train, X_test, y_true):
    tuned_parameters = [
        {
            'kernel': ['linear'],
            'nu': [0.1, 0.5, 0.9]
        },
        {
            'kernel': ['poly'],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto'],
            'nu': [0.1, 0.5, 0.9]
        },
        {
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto'],
            'nu': [0.1, 0.5, 0.9]
        }
    ]
    print(1 / (29 * X_train.var()))
    exit()
    # train the model on train set 
    model = IsolationForest(n_estimators=100, contamination=0.0017, max_features=0.1, n_jobs=1, random_state=0, verbose=1)
    model.fit(X_train) 
    
    # print prediction results 
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == 1, 0, 1)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
    tnr, fpr, fnr, tpr = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
    roc_auc = roc_auc_score(y_true, y_pred)
    print('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(tn, fp, fn, tp))
    print('TNR: {:.5f}, TPR: {:.5f}'.format(tnr, tpr))
    print('ROC AUC: {:.5f}\n'.format(roc_auc))

if __name__ == '__main__':
    # load training set
    training_set = pd.read_pickle('../../../pickle/trainingset.pkl')
    X_train = training_set.iloc[:, 0:-1].values

    # load raw test set
    test_set = pd.read_pickle('../../../pickle/raw_testset.pkl')
    X_test = test_set.iloc[:, 0:-1].values
    y_true = test_set.iloc[:, -1].values
    scaler = load('../../../models/preprocessing/minmaxscaler.bin')
    X_test = scaler.transform(X_test)

    get_predictions(X_train, X_test, y_true)