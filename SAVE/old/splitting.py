import numpy as np
import pandas as pd

from models_config import splitting_config

def split_dataset(data):
    '''
    Split dataset into training set (80%) and test set (20%)
    
    Parameters:
        data (dataframe): the dataset to split
    
    Returns:
        (dataframe): training set
        (dataframe): test set
    '''

    test_ratio = splitting_config['test_ratio']
    total_size = data.shape[0]
    shuffled_indices = np.random.permutation(total_size)
    test_set_size = int(total_size * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    training_indices = shuffled_indices[test_set_size:]
    return data.iloc[training_indices], data.iloc[test_indices]

if __name__ == '__main__':
    # load dataset
    data = pd.read_csv('../../dataset/creditcard.csv', sep=',')

    # separe genuine transcations from frauds
    frauds = data[data['Class'] == 1]
    genuine = data[data['Class'] == 0]

    # split gunuine transaction in training and test sets
    genuine_training, genuine_test = split_dataset(genuine)

    # add frauds to test set
    test_set = pd.concat([genuine_test, frauds])

    print('Total: {0}, Training: {1}, Test: {2}'.format(data.shape[0], genuine_training.shape[0], test_set.shape[0]))

    # store raw training set and raw test set (Feature 'Time' is dropped)
    genuine_training.drop(columns=['Time']).to_pickle('../../pickle/raw_trainingset.pkl')
    test_set.drop(columns=['Time']).to_pickle('../../pickle/raw_testset.pkl')