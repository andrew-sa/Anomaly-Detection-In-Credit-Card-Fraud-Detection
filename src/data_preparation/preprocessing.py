import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def create_min_max_scaler(model_data):
    scaler = MinMaxScaler().fit(model_data.values)
    dump(scaler, '../../models/minmaxscaler.bin', compress=True)

def min_max_normalization(data):
    scaler = load('../../models/minmaxscaler.bin')
    scaled_data = scaler.transform(data.values)
    return pd.DataFrame(scaled_data)

def plot_cumulative_explained_variance(data):
    # plot cumulative explained variance
    pca_plot = PCA().fit(data.values)
    plt.plot(np.cumsum(pca_plot.explained_variance_ratio_))
    threshold = plt.hlines(y=0.95, xmin=0, xmax=data.shape[1], linestyles='dashed',colors='red', label='Threshold=0.95')
    plt.legend(handles=[threshold])
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

def create_pca_model(model_data):
    pca = PCA(n_components=0.95, svd_solver='full').fit(model_data.values)
    dump(pca, '../../models/pca.bin', compress=True)

def pca_dimensionality_reduction(data):
    # perform PCA
    pca = load('../../models/pca.bin')
    pca_result = pca.transform(data.values)
    df_pca = pd.DataFrame(pca_result)
    # print(df_pca)
    return df_pca

def plot_tsne(training_set, test_set):
    # Perform preprocessing on frauds
    test_set = test_set[test_set['Class'] == 1]
    scaled_data = min_max_normalization(test_set.iloc[:, 0:-1])
    pca_result = pca_dimensionality_reduction(scaled_data)
    target_class = test_set.iloc[:, -1].values
    test_set = pd.DataFrame(pca_result)
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
    create_pca_model(training_set) # create the model based on he training set
    plot_cumulative_explained_variance(training_set) # plot cumulative explained variance for PCA
    training_set = pca_dimensionality_reduction(training_set)
    
    # add target class to tranining set
    training_set['Class'] = target_class_training_set

    # store training set
    print(training_set)
    training_set.to_pickle('../../pickle/trainingset.pkl')

    plot_tsne(training_set, test_set)
