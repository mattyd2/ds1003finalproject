from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd


def data_loader(model_spec_function):
    '''
    Purpose:
        Load the Training Data
    Args:
        model_spec_function - Default is test_model will load the train.csv
        data. Otherwise, you can pass a filepath.
    Returns:
        X_train, y_train dataframes.
    '''
    if model_spec_function == 'Identity':
        print 'Reading training data train.csv'
        train_data = pd.read_csv('./../data/final_data/train.csv')

        x_cols = np.setdiff1d(train_data.columns, ['appl_dec_G', 'appl_dec_D', 'appl_dec_F', 'appl_dec_L'])

        X_train = train_data[x_cols]
        y_train = train_data['appl_dec_G']

    # TODO - confirm this works
    else:
        print 'Reading training data train.csv and making design matrix using {}'.format(model_spec_function.__name__)
        X_train, y_train = model_spec_function('./../../data/final_data/train.csv')

    return X_train, y_train


def scaler(scale, X_train):
    '''
    Purpose:
        Scale training features
        Use for regularized linear models (SVC, log reg)
    Args:
        scale: true or false
    Returns:
        X_train scaled or un-changed
    '''
    if scale is True:
        print 'Scaling features'
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

    elif scale is False:
        X_train_scaled = X_train

    return X_train_scaled


def test_model(model, param_grid, scale=True, model_spec_function='Identity'):
    '''
    Purpose:
        Function for grid search optimization of hyperparameters.
    Args:
        model - sklearn model
        param_grid - dictionary of hyperparameters for grid search
        scale - boolean indicating if the X_train should be scaled
        model_spec_function - optional argument for loading data
    Returns:
        grid - the fit, scored GridSearchCV object
    '''
    X_train, y_train = data_loader(model_spec_function)
    X_train_scaled = scaler(scale, X_train)

    print 'Conducting grid search'
    np.random.seed(4590385)
    grid = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, pre_dispatch='2*n_jobs')
    grid.fit(X_train_scaled, y_train)

    return grid
