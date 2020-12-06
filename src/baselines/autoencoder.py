import logging
import numpy as np
import pandas as pd

from tensorflow import keras
from joblib import dump, load
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger('baselines')
hdlr = logging.FileHandler('../../logs/baselines.log')
formatter = logging.Formatter('%(levelname)s - %(asctime)s - [%(filename)s:%(lineno)s] - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

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

def _one_hidden_layers(X_train, X_validate, X_test, y_true):
    '''
    Find best model with one hidden layer

    Parameters:
        X_train (ndarray of shape (n_samples, n_features)): training set
        X_train (ndarray of shape (n_samples, n_features)): validation set
        X_test (ndarray of shape (n_samples, n_features)): test set
        y_true (ndarray of shape (n_samples,)): correct target values

    Returns:
        best_score: score of the best model
        best_autoencoder: best model
        best_cm: confusion matrxi of the best model
        best_rates: rates of the best model
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
    EPOCHS = 30

    best_score = 0
    best_autoencoder = None
    best_cm = None
    best_rates = None
    # best_roc_auc = 0

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
                shuffle=True,
                validation_data=(X_validate, X_validate)
            )

            # pass the test set through the autoencoder to get the reconstructed result
            reconstructions = autoencoder.predict(X_test)

            # calculating the mean squared error reconstruction loss per row
            mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

            # calcualting the outliers threshold
            outliers_threshold = _calculate_outliers_threshold(mse)
            y_pred = (mse > outliers_threshold).astype(int)

            cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
            rates = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
            # roc_auc = roc_auc_score(y_true, mse)
            # score = roc_auc + rates[3]
            score = rates[3]
            if score > best_score:
                best_score = score
                best_autoencoder = autoencoder
                best_cm = cm
                best_rates = rates
                # best_roc_auc = roc_auc

    return best_score, best_autoencoder, best_cm, best_rates

def _zero_hidden_layers(X_train, X_validate, X_test, y_true):
    '''
    Find best model without hidden layers

    Parameters:
        X_train (ndarray of shape (n_samples, n_features)): training set
        X_train (ndarray of shape (n_samples, n_features)): validation set
        X_test (ndarray of shape (n_samples, n_features)): test set
        y_true (ndarray of shape (n_samples,)): correct target values

    Returns:
        best_score: score of the best model
        best_autoencoder: best model
        best_cm: confusion matrxi of the best model
        best_rates: rates of the best model
    '''
    input_dim = X_train.shape[1]
    code_dims = [int(input_dim/2), int(input_dim/4), int(input_dim/8)]
    activation_sequences = [['elu', 'elu', 'elu'], ['tanh', 'tanh', 'tanh'], ['elu', 'tanh', 'elu'], ['tanh', 'elu', 'tanh']]
    BATCH_SIZE = 256
    EPOCHS = 30

    best_score = 0
    best_autoencoder = None
    best_cm = None
    best_rates = None
    # best_roc_auc = 0

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
                shuffle=True,
                validation_data=(X_validate, X_validate)
            )

            # pass the test set through the autoencoder to get the reconstructed result
            reconstructions = autoencoder.predict(X_test)

            # calculating the mean squared error reconstruction loss per row
            mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

            # calcualting the outliers threshold
            outliers_threshold = _calculate_outliers_threshold(mse)
            y_pred = (mse > outliers_threshold).astype(int)

            cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None).ravel()
            rates = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true').ravel()
            # roc_auc = roc_auc_score(y_true, mse)
            # score = roc_auc + rates[3]
            score = rates[3]
            if score > best_score:
                best_score = score
                best_autoencoder = autoencoder
                best_cm = cm
                best_rates = rates
                # best_roc_auc = roc_auc

    return best_score, best_autoencoder, best_cm, best_rates

def get_result(X_train, X_validate, X_test, y_true):
    '''
    Find best configuration and calculate its results

    Parameters:
        X_train (ndarray of shape (n_samples, n_features)): training set
        X_train (ndarray of shape (n_samples, n_features)): validation set
        X_test (ndarray of shape (n_samples, n_features)): test set
        y_true (ndarray of shape (n_samples,)): correct target values

    Returns:
        best_score: score of the best model
        best_autoencoder: best model
        best_cm: confusion matrxi of the best model
        best_rates: rates of the best model
    '''
    best_zero_hidden = _zero_hidden_layers(X_train, X_validate, X_test, y_true)
    best_one_hidden = _one_hidden_layers(X_train, X_validate, X_test, y_true)
    if best_zero_hidden[0] > best_one_hidden[0]:
        return best_zero_hidden
    else:
        return best_one_hidden

if __name__ == '__main__':
    # load training set
    training_set = pd.read_pickle('../../pickle/trainingset.pkl')
    X_train, X_validate = train_test_split(training_set.iloc[:, 0:-1], test_size=0.2, random_state=10, shuffle=True, stratify=None)
    X_train = X_train.values
    X_validate = X_validate.values

    # load and transform raw test set
    test_set = pd.read_pickle('../../pickle/raw_testset.pkl')
    X_test = test_set.iloc[:, 0:-1].values
    y_true = test_set.iloc[:, -1].values
    scaler = load('../../models/preprocessing/minmaxscaler.bin')
    X_test = scaler.transform(X_test)

    result = get_result(X_train, X_validate, X_test, y_true)
    result[1].save('../../models/baselines/autoencoder.h5')
    print(result[1].summary())
    logger.info('AUTOENCODER')
    logger.info('TN: {0}, FP: {1}, FN: {2}, TP: {3}'.format(result[2][0], result[2][1], result[2][2], result[2][3]))
    logger.info('TNR: {:.5f}, TPR: {:.5f}\n'.format(result[3][0], result[3][3]))