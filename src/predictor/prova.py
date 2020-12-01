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
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold, GridSearchCV

if __name__ == '__main__':
    # load data
    data = pd.read_pickle('../../pickle/trainingset.pkl')
    # load first level clustering
    clustering = load('../../models/clustering/kmeans_100.bin')
    data = data.iloc[:, 0:-1]
    data['Cluster'] = clustering.labels_
    grouped = data.groupby(data.Cluster)
    cluster_centers = clustering.cluster_centers_

    count = 0
    df = None
    for key, group in grouped.__iter__():
        if len(group) < 1000:
            count += 1
            if df is None:
                df = group.iloc[:, 0:-1]
            else:
                df.append(group.iloc[:, 0:-1], ignore_index=True)
    print(count)
    print(df)
    lst_ocsvm = []
    for key, group in grouped.__iter__():
        if len(group) >= 1000:
            X_train, test = train_test_split(group.iloc[:, 0:-1], test_size=0.1, random_state=10, shuffle=True, stratify=None)
            X_test = pd.concat([test, df.sample(n=150)])
            y_true = np.concatenate([np.repeat(1, len(test)), np.repeat(-1, 150)])
            model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale', verbose=True) 
            model.fit(X_train.values)
            y_pred = model.predict(X_test.values)
            # y_pred = np.where(y_pred == 1, 0, 1)
            tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
            tnr, fpr, fnr, tpr = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
            roc_auc = roc_auc_score(y_true, y_pred)
            print('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(tn, fp, fn, tp))
            print('TNR: {:.5f}, TPR: {:.5f}'.format(tnr, tpr))
            print('ROC AUC: {:.5f}\n'.format(roc_auc))
            lst_ocsvm.append(model)
            dump(lst_ocsvm, '../../models/prova.bin')
        else:
            count += 1
    print(count)