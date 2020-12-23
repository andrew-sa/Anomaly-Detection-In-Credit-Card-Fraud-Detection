import logging
import numpy as np
import pandas as pd

from joblib import dump, load
from tensorflow import keras, random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

logger = logging.getLogger('baselines')
hdlr = logging.FileHandler('../../logs/baselines.log')
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

def _calculate_outliers_threshold(values):
    '''
    This function calculate the threshold by which a data point is evaluated as not outlier

    Parameters:
        values (array): values

    Returns:
        outliers_threshold (float): outliers threshold
    '''
    sorted(values)
    # q1 = first quartile, q3 = third quartile
    q1, q3 = np.percentile(values, [25, 75])
    # iqr = interquartile range
    iqr = q3 - q1
    outliers_threshold = q3 + (1.5 * iqr)
    return outliers_threshold

def _one_hidden_layers(X_train, X_validate, y_true):
    '''
    Find best model with one hidden layer

    Parameters:
        X_train (ndarray of shape (n_samples_train, n_features)): training set
        X_validation (ndarray of shape (n_samples_validate, n_features)): validation set
        y_true (ndarray of shape (n_samples_validate,)): correct target values of validation set

    Returns:
        best_tpr: true postive rate of the best model
        best_tnr: true negative rate of the best model
        best_autoencoder: best model
    '''
    input_dim = X_train.shape[1]
    hidden_dims = [
        [int(input_dim/2), int(input_dim/4), int(input_dim/2)],
        [int(input_dim/2), int(input_dim/8), int(input_dim/2)],
        [int(input_dim/4), int(input_dim/8), int(input_dim/4)]
    ]
    activation_sequences = [
        ['elu', 'elu', 'elu', 'elu', 'elu'], 
        ['tanh', 'tanh', 'tanh', 'tanh', 'tanh'], 
        ['tanh', 'elu', 'tanh', 'elu', 'tanh'], 
        ['elu', 'tanh', 'elu', 'tanh', 'elu']
    ]
    BATCH_SIZE = 256
    EPOCHS = 300

    best_tpr = 0
    best_tnr = 0
    best_autoencoder = None

    for hidden_dim in hidden_dims:
        for activation_sequence in activation_sequences:
            # construct autoencoder
            autoencoder = keras.models.Sequential()
            autoencoder.add(keras.layers.Dense(input_dim, activation=activation_sequence[0], input_shape=(input_dim, )))
            autoencoder.add(keras.layers.Dense(hidden_dim[0], activation=activation_sequence[1]))
            autoencoder.add(keras.layers.Dense(hidden_dim[1], activation=activation_sequence[2]))
            autoencoder.add(keras.layers.Dense(hidden_dim[2], activation=activation_sequence[3]))
            autoencoder.add(keras.layers.Dense(input_dim, activation=activation_sequence[4]))

            autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            
            autoencoder.summary()

            early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00005, patience=3, verbose=1, mode='min',restore_best_weights=False)
            callbacks = [early_stop]

            autoencoder.fit(
                X_train, X_train,
                epochs=EPOCHS,
                callbacks=callbacks,
                batch_size=BATCH_SIZE,
                shuffle=True
                # validation_data=(X_validate, X_validate)
            )

            # pass the test set through the autoencoder to get the reconstructed result
            reconstructions = autoencoder.predict(X_validate)

            # calculating the mean squared error reconstruction loss per row
            mse = np.mean(np.power(X_validate - reconstructions, 2), axis=1)

            # calcualting the outliers threshold
            outliers_threshold = _calculate_outliers_threshold(mse)
            y_pred = (mse > outliers_threshold).astype(int)

            rates_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
            tpr = rates_matrix[3]
            tnr = rates_matrix[0]
            if tpr > best_tpr:
                best_tpr = tpr
                best_tnr = tnr
                best_autoencoder = autoencoder
            elif tpr == best_tpr and tnr > best_tnr:
                best_tpr = tpr
                best_tnr = tnr
                best_autoencoder = autoencoder
            print(rates_matrix)

    return best_tpr, best_tnr, best_autoencoder

def _zero_hidden_layers(X_train, X_validate, y_true):
    '''
    Find best model without hidden layers

    Parameters:
        X_train (ndarray of shape (n_samples_train, n_features)): training set
        X_validation (ndarray of shape (n_samples_validate, n_features)): validation set
        y_true (ndarray of shape (n_samples_validate,)): correct target values of validation set

    Returns:
        best_tpr: true postive rate of the best model
        best_tnr: true negative rate of the best model
        best_autoencoder: best model
    '''
    input_dim = X_train.shape[1]
    code_dims = [int(input_dim/2), int(input_dim/4), int(input_dim/8)]
    activation_sequences = [['elu', 'elu', 'elu'], ['tanh', 'tanh', 'tanh'], ['elu', 'tanh', 'elu'], ['tanh', 'elu', 'tanh']]
    BATCH_SIZE = 256
    EPOCHS = 300

    best_tpr = 0
    best_tnr = 0
    best_autoencoder = None

    for code_dim in code_dims:
        for activation_sequence in activation_sequences:
            # construct autoencoder
            autoencoder = keras.models.Sequential()
            autoencoder.add(keras.layers.Dense(input_dim, activation=activation_sequence[0], input_shape=(input_dim, )))
            autoencoder.add(keras.layers.Dense(code_dim, activation=activation_sequence[1]))
            autoencoder.add(keras.layers.Dense(input_dim, activation=activation_sequence[2]))

            autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
                            
            autoencoder.summary()

            early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00005, patience=3, verbose=1, mode='min',restore_best_weights=False)
            callbacks = [early_stop]

            autoencoder.fit(
                X_train, X_train,
                epochs=EPOCHS,
                callbacks=callbacks,
                batch_size=BATCH_SIZE,
                shuffle=True
                # validation_data=(X_validate, X_validate)
            )

            # pass the test set through the autoencoder to get the reconstructed result
            reconstructions = autoencoder.predict(X_validate)

            # calculating the mean squared error reconstruction loss per row
            mse = np.mean(np.power(X_validate - reconstructions, 2), axis=1)

            # calcualting the outliers threshold
            outliers_threshold = _calculate_outliers_threshold(mse)
            y_pred = (mse > outliers_threshold).astype(int)

            rates_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
            tpr = rates_matrix[3]
            tnr = rates_matrix[0]
            if tpr > best_tpr:
                best_tpr = tpr
                best_tnr = tnr
                best_autoencoder = autoencoder
            elif tpr == best_tpr and tnr > best_tnr:
                best_tpr = tpr
                best_tnr = tnr
                best_autoencoder = autoencoder

    return best_tpr, best_tnr, best_autoencoder

def _validation(X_train, X_validate, y_validate):
    '''
    Select best model between zero hidden layer and one hidden layer

    Parameters:
        X_train (ndarray of shape (n_samples_train, n_features)): training set
        X_validation (ndarray of shape (n_samples_validate, n_features)): validation set
        y_validate (ndarray of shape (n_samples_validate,)): correct target values of validation set
    '''
    best_zero_hidden = _zero_hidden_layers(X_train, X_validate, y_validate)
    best_one_hidden = _one_hidden_layers(X_train, X_validate, y_validate)
    if best_zero_hidden[0] > best_one_hidden[0]:
        return best_zero_hidden[2]
    elif best_zero_hidden[0] == best_one_hidden[0] and best_zero_hidden[1] > best_one_hidden[1]:
        return best_zero_hidden[2]
    else:
        return best_one_hidden[2]

def get_result(X_train, X_validate, y_validate, X_test, y_true):
    '''
    Find best configuration and calculate its results

    Parameters:
        X_train (ndarray of shape (n_samples_train, n_features)): independent columns of training set
        X_validate (ndarray of shape (n_samples_validate, n_features)): independent columns of validation set
        y_validate (ndarray of shape (n_samples_validate,)): labels of validation set
        X_test (ndarray of shape (n_samples_test, n_features)): independent columns of test set
        y_test (ndarray of shape (n_samples_validate,)): labels of test set

    Returns:
        best_score: score of the best model
        best_autoencoder: best model
        best_cm: confusion matrxi of the best model
        best_rates: rates of the best model
    '''
    autoencoder = _validation(X_train, X_validate, y_validate)

    # pass the test set through the autoencoder to get the reconstructed result
    reconstructions = autoencoder.predict(X_test)

    # calculating the mean squared error reconstruction loss per row
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

    # calcualting the outliers threshold
    outliers_threshold = _calculate_outliers_threshold(mse)
    y_pred = (mse > outliers_threshold).astype(int)

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
    rates_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()

    autoencoder.save('../../models/baselines/autoencoder.h5')
    print(autoencoder.summary())

    return conf_matrix, rates_matrix

if __name__ == '__main__':
    set_reproducibility()
    # load training set and validation set
    training_set = pd.read_pickle('../../pickle/processed/trainingset.pkl')
    validation_set = pd.read_pickle('../../pickle/processed/validationset.pkl')
    X_train = training_set.iloc[:, 0:-1].values
    X_validate = validation_set.iloc[:, 0:-1].values
    y_validate = validation_set.iloc[:, -1].values

    # load and transform raw test set
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    X_test = test_set.iloc[:, 0:-1].values
    y_true = test_set.iloc[:, -1].values
    scaler = load('../../models/preprocessing/minmaxscaler.bin')
    X_test = scaler.transform(X_test)

    conf_matrix, rates_matrix = get_result(X_train, X_validate, y_validate, X_test, y_true)
    logger.info('AUTOENCODER')
    logger.info('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(conf_matrix[0], conf_matrix[1], conf_matrix[2], conf_matrix[3]))
    logger.info('TNR: {:.5f}, TPR: {:.5f}\n'.format(rates_matrix[0], rates_matrix[3]))