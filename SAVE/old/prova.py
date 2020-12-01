import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals.joblib import dump, load


def dimensionality_reduction(data):
    # perform PCA
    pca_plot = PCA().fit(data.values)
    plt.plot(np.cumsum(pca_plot.explained_variance_ratio_))
    plt.hlines(y=0.95, xmin=0, xmax=data.shape[1], linestyles='dashed',colors='red', label='')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    pca = PCA(n_components=0.95, svd_solver='full')
    pca_results = pca.fit_transform(data.values)
    df_pca = pd.DataFrame(pca_results)
    print(df_pca)
    return df_pca

def plot_tsne(genuine, frauds):
    # select 100000 transaction from genuines ones
    genuine = genuine.sample(n=1000)
    data = pd.concat([genuine, frauds])
    
    # perform t-SNE
    time_start = time.time()
    tsne = TSNE(n_components=2, perplexity=30.0, verbose=1)
    tsne_results = tsne.fit_transform(data.values)
    time_elapsed = time.time() - time_start
    print('t-SNE done! Time elapsed: {0} seconds ({1} minutes)'.format(time_elapsed, divmod(time_elapsed, 60)))

    # save t-SNE output in a data frame
    df_tsne = pd.DataFrame(tsne_results, columns=['tsne-2d-one','tsne-2d-two'])
    df_tsne['Class'] = np.concatenate((genuine.iloc[:, -1].values, frauds.iloc[:, -1].values), axis=None)
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

'''
def plot_pca(data):
    # select only independent variables
    independent_cols = data.iloc[:, 0:-1]
    
    # perform PCA
    time_start = time.time()
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(independent_cols.values)
    print('PCA done! Time elapsed: {0} seconds'.format(time.time() - time_start))

    # save PCA output in a data frame
    df_pca = pd.DataFrame(pca_results, columns=['pca-2d-one','pca-2d-two'])
    df_pca['Class'] = data.iloc[:, -1]
    # divide genuine transactions from frauds
    f_df_pca = df_pca[df_pca['Class'] == 1] # frauds
    g_df_pca = df_pca[df_pca['Class'] == 0] # genuine

    # plot PCA output
    blue_points = plt.scatter(x=g_df_pca.iloc[:, 0], y=g_df_pca.iloc[:, 1], color='blue', s=10, label='Genuine')
    red_points = plt.scatter(x=f_df_pca.iloc[:, 0], y=f_df_pca.iloc[:, 1], color='red', s=10, label='Frauds')
    plt.legend(handles=[blue_points, red_points])
    plt.xlabel('pca-2d-one')
    plt.ylabel('pca-2d-two')
    plt.show()
'''

def split_dataset(data):
    test_ratio = 0.2
    total_size = data.shape[0]
    shuffled_indices = np.random.permutation(total_size)
    test_set_size = int(total_size * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    training_indices = shuffled_indices[test_set_size:]
    return data.iloc[training_indices], data.iloc[test_indices]


if __name__ == '__main__':
    # load dataset
    data = pd.read_csv('../dataset/creditcard.csv', sep=',')
    
    # min-max normalization 
    independent_cols = data.iloc[:, 0:-1]
    scaled_data = MinMaxScaler().fit_transform(independent_cols)
    transactions = pd.DataFrame(scaled_data)
    # transactions = (independent_cols - independent_cols.min()) / (independent_cols.max() - independent_cols.min())
    # transactions = (independent_cols - independent_cols.mean()) / independent_cols.std()
    
    # add target class to normalized data
    transactions['Class'] = data.iloc[:, -1]
    print(transactions)
    
    # separe genuine transcations from frauds
    frauds = transactions[transactions['Class'] == 1]
    genuine = transactions[transactions['Class'] == 0]

    # split gunuine transaction in training and test sets
    genuine_training, genuine_test = split_dataset(genuine)

    # add frauds to test set
    test_set = pd.concat([genuine_test, frauds])

    print('Total: {0}, Training: {1}, Test: {2}'.format(transactions.shape[0], genuine_training.shape[0], test_set.shape[0]))
    '''
    # store traning set and test set
    genuine_training.to_pickle('../pickle/trainingset.pkl')
    test_set.to_pickle('../pickle/testset.pkl')
    '''
    # plot dataset after dimensionality reduction
    plot_tsne(genuine, frauds)
    # plot_pca(transactions)
