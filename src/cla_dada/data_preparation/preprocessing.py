import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.manifold import TSNE
from models_config import splitting_config
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def split_training_set(training_set):
    '''
    Split training set in training and validation sets

    Parameters:
        training_set (DataFrame): training set to split

    Returns:
        training_set (DataFrame): training set (only genuine transaction)
        validation_set (DataFrame): validation set
    '''
    # separe genuine transcations from frauds
    frauds = training_set[training_set['Class'] == 1]
    genuine = training_set[training_set['Class'] == 0]

    # split genuine transaction in training and validation sets
    genuine_training, genuine_validation = train_test_split(genuine, test_size=splitting_config['genuine_validate_ratio'], random_state=10, shuffle=True, stratify=None)

    # create validation set
    validation_set = pd.concat([genuine_validation, frauds])

    return genuine_training, validation_set

def create_min_max_scaler(model_data):
    '''
    Create a min-max scaler

    Parameters:
        model_data (dataframe): data which model is fitted
    '''
    scaler = MinMaxScaler().fit(model_data.values)
    dump(scaler, '../../../models/preprocessing/minmaxscaler.bin', compress=True)

def min_max_normalization(data):
    '''
    Normalize the data

    Parameters:
        data (dataframe): data to normalize
    
    Returns:
        (dataframe): normalized data
    '''
    scaler = load('../../../models/preprocessing/minmaxscaler.bin')
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
    training_set = training_set.sample(n=5000, random_state=10, axis=0)
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
    training_set = pd.read_pickle('../../../pickle/raw_trainingset.pkl')
    test_set = pd.read_pickle('../../../pickle/raw_testset.pkl')
    before_training_size = training_set.shape[0]

    training_set, validation_set = split_training_set(training_set)

    # perform min-max normalization
    target_class_training_set = training_set.iloc[:, -1].values
    create_min_max_scaler(training_set.iloc[:, 0:-1]) # create the model based on the training set
    training_set = min_max_normalization(training_set.iloc[:, 0:-1])
    # add target class to tranining set
    training_set['Class'] = target_class_training_set

    # perform min-max normalization on validation set
    target_class_validation_set = validation_set.iloc[:, -1].values
    validation_set = min_max_normalization(validation_set.iloc[:, 0:-1])
    # add target class to validation set
    validation_set['Class'] = target_class_validation_set

    # store training set
    print(training_set)
    training_set.to_pickle('../../../pickle/processed/trainingset.pkl')
    # store validation set
    print(validation_set)
    validation_set.to_pickle('../../../pickle/processed/validationset.pkl')

    plot_tsne(training_set, test_set)

    print('Before: {0}, Training: {1}, Validation: {2}'.format(before_training_size, training_set.shape[0], validation_set.shape[0]))
    