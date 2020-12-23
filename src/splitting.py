import numpy as np
import pandas as pd

from models_config import splitting_config
from sklearn.model_selection import train_test_split

def split_dataset(data):
    '''
    Split dataset into training set and test set
    
    Parameters:
        data (DataFrame): the dataset to split
    
    Returns:
        (DataFrame): training set
        (DataFrame): test set
    '''
    # separe genuine transcations from frauds
    frauds = data[data['Class'] == 1]
    genuine = data[data['Class'] == 0]

    # split genuine transaction in training and test sets
    genuine_training, genuine_test = train_test_split(genuine, test_size=splitting_config['genuine_test_ratio'], random_state=10, shuffle=True, stratify=None)

    # split frauds in training and test sets
    frauds_training, frauds_test = train_test_split(frauds, test_size=splitting_config['frauds_test_ratio'], random_state=10, shuffle=True, stratify=None)

    # create training set
    training_set = pd.concat([genuine_training, frauds_training])

    # create test set
    test_set = pd.concat([genuine_test, frauds_test])

    return training_set, test_set   

if __name__ == '__main__':
    # load dataset
    data = pd.read_csv('../../dataset/creditcard.csv', sep=',')

    # split dataset (Feature 'Time' is dropped)
    training_set, test_set = split_dataset(data.drop(columns=['Time']))

    print('Total: {0}, Training: {1}, Test: {2}'.format(data.shape[0], training_set.shape[0], test_set.shape[0]))

    # store raw training set and raw test set
    training_set.to_pickle('../../pickle/raw_trainingset.pkl')
    test_set.to_pickle('../../pickle/raw_testset.pkl')