import logging
import numpy as np
import pandas as pd

from joblib import dump, load
from tensorflow import keras, random
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger('binary-classification')
hdlr = logging.FileHandler('../../logs/binary_classification.log')
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def set_reproducibility():
    '''
    Set Keras reproducibility
    '''
    seed_value = 0
    random.set_seed(seed_value)

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
    input_dim = X_train.shape[1]
    LEARNING_RATE = 0.001
    BATCH_SIZE = 4096
    EPOCHS = 300

    # construct neural network
    deep_neural_network = keras.models.Sequential()
    deep_neural_network.add(keras.layers.Dense(50, activation='tanh', input_shape=(input_dim, )))
    deep_neural_network.add(keras.layers.Dense(30, activation='tanh'))
    deep_neural_network.add(keras.layers.Dense(30, activation='tanh'))
    deep_neural_network.add(keras.layers.Dense(50, activation='tanh'))
    deep_neural_network.add(keras.layers.Dense(1, activation='sigmoid'))
    deep_neural_network.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])               
    deep_neural_network.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00005, patience=3, verbose=1, mode='min', restore_best_weights=False)
    callbacks = [early_stop]

    # train the model
    deep_neural_network.fit(
        X_train, y_train,
        epochs=EPOCHS,
        callbacks=callbacks,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(X_validate, y_validate)
    )

    # predict target class of test set
    y_prob = deep_neural_network.predict(X_test)
    y_prob = y_prob.flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # print result
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
    rates_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
    logger.info('DEEP NEURAL NETWORK')
    logger.info('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(conf_matrix[0], conf_matrix[1], conf_matrix[2], conf_matrix[3]))
    logger.info('TNR: {0:.5f}, TPR: {1:.5f}\n'.format(rates_matrix[0], rates_matrix[3]))
    # save the model
    deep_neural_network.save('../../models/binary_classifiers/deep_nn.h5')

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
    set_reproducibility()
    # load data
    training_set = pd.read_pickle('../../pickle/raw_trainingset.pkl')
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    # pre-processing
    X_train, y_train, X_validate, y_validate, X_test, y_test = preprocessing(training_set, test_set)
    # result
    get_result(X_train, y_train, X_validate, y_validate, X_test, y_test)