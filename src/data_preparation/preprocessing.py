import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def create_min_max_scaler(model_data):
    '''
    Create a min-max scaler

    Parameters:
        model_data (dataframe): data which model is fitted
    '''
    scaler = MinMaxScaler().fit(model_data.values)
    dump(scaler, '../../models/preprocessing/minmaxscaler.bin', compress=True)

def min_max_normalization(data):
    '''
    Normalize the data

    Parameters:
        data (dataframe): data to normalize
    
    Returns:
        (dataframe): normalized data
    '''
    scaler = load('../../models/preprocessing/minmaxscaler.bin')
    scaled_data = scaler.transform(data.values)
    return pd.DataFrame(scaled_data)

def plot_tsne(training_set, test_set):
    '''
    Plot a 2D representation of 5000 genuine transactions and all frauds

    Parameters:
        training_set (dataframe): training set
        test_set (dataframe): test set
    '''
    # Perform preprocessing on frauds
    test_set = test_set[test_set['Class'] == 1]
    scaled_data = min_max_normalization(test_set.iloc[:, 0:-1])
    target_class = test_set.iloc[:, -1].values
    test_set = scaled_data
    test_set['Class'] = target_class

    # select 5000 transaction from training set
    training_set = training_set.sample(n=5000)
    data = pd.concat([training_set, test_set])
    
    # perform t-SNE
    time_start = time.time()
    tsne = TSNE(n_components=2, perplexity=30.0, verbose=1)
    tsne_result = tsne.fit_transform(data.iloc[:, 0:-1].values)
    time_elapsed = time.time() - time_start
    print('t-SNE done! Time elapsed: {0} seconds ({1} minutes)'.format(time_elapsed, divmod(time_elapsed, 60)))

    # save t-SNE output in a data frame
    df_tsne = pd.DataFrame(tsne_result, columns=['tsne-2d-one','tsne-2d-two'])
    df_tsne['Class'] = data.iloc[:, -1].values
    # divide genuine transactions from frauds
    f_df_tsne = df_tsne[df_tsne['Class'] == 1] # frauds
    g_df_tsne = df_tsne[df_tsne['Class'] == 0] # genuine

    # plot t-SNE output
    blue_points = plt.scatter(x=g_df_tsne.iloc[:, 0], y=g_df_tsne.iloc[:, 1], color='blue', s=10, label='Genuine')
    red_points = plt.scatter(x=f_df_tsne.iloc[:, 0], y=f_df_tsne.iloc[:, 1], color='red', s=10, label='Frauds')
    plt.legend(handles=[blue_points, red_points])
    plt.xlabel('tsne-2d-one')
    plt.ylabel('tsne-2d-two')
    plt.show()

if __name__ == '__main__':
    # load sets
    training_set = pd.read_pickle('../../pickle/raw_trainingset.pkl')
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')

    # perform min-max normalization and dimensionality reduction using PCA on training set
    target_class_training_set = training_set.iloc[:, -1].values
    create_min_max_scaler(training_set.iloc[:, 0:-1]) # create the model based on the training set
    training_set = min_max_normalization(training_set.iloc[:, 0:-1])
    
    # add target class to tranining set
    training_set['Class'] = target_class_training_set

    # store training set
    print(training_set)
    training_set.to_pickle('../../pickle/trainingset.pkl')

    plot_tsne(training_set, test_set)
