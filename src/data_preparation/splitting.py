import numpy as np
import pandas as pd

from models_config import splitting_config
from sklearn.model_selection import train_test_split

def split_dataset(data):
    '''
    Split dataset into training set and test set
    
    Parameters:
        data (dataframe): the dataset to split
    
    Returns:
        (dataframe): training set
        (dataframe): test set
    '''
    
    test_ratio = splitting_config['test_ratio']

    # separe genuine transcations from frauds
    frauds = data[data['Class'] == 1]
    genuine = data[data['Class'] == 0]

    # split gunuine transaction in training and test sets
    genuine_training, genuine_test = train_test_split(genuine, test_size=test_ratio, random_state=10, shuffle=True, stratify=None)

    # add frauds to test set
    test_set = pd.concat([genuine_test, frauds])

    return genuine_training, test_set   

if __name__ == '__main__':
    # load dataset
    data = pd.read_csv('../../dataset/creditcard.csv', sep=',')

    # split dataset
    training_set, test_set = split_dataset(data)

    print('Total: {0}, Training: {1}, Test: {2}'.format(data.shape[0], training_set.shape[0], test_set.shape[0]))

    # store raw training set and raw test set (Feature 'Time' is dropped)
    training_set.drop(columns=['Time']).to_pickle('../../pickle/raw_trainingset.pkl')
    test_set.drop(columns=['Time']).to_pickle('../../pickle/raw_testset.pkl')