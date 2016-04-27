#!/usr/bin/python
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle

def test_model(model, param_grid, scale = True):
    '''
    Function for grid search optimization of hyperparameters.
    Args:
        model - sklearn model
        param_grid - dictionary of hyperparameters for grid search
        scale - Boolean indicating if the X_train should be scaled using standard scaler.
    Returns:
        grid - the fit, scored GridSearchCV object
    '''

    print 'Reading training data'

    train_data = pd.read_csv('./train.csv')

    x_cols = np.setdiff1d(train_data.columns, ['appl_dec_G', 'appl_dec_D', 'appl_dec_F', 'appl_dec_L'])

    X_train = train_data[x_cols]
    y_train = train_data['appl_dec_G']

    if scale == True:
        print 'Scaling features'

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

    elif scale == False:
        X_train_scaled = X_train

    print 'Conducting grid search'
    np.random.seed(4590385)
    grid = GridSearchCV(model(), param_grid=param_grid, scoring='roc_auc',cv=2)
    grid.fit(X_train_scaled, y_train)
    print 'Grid search done'
    return grid

if __name__ == '__main__':

    ### This is for my RandomForestClassifier_model
    model = RandomForestClassifier
    param_grid = {'n_estimators': [100], 'criterion': ['entropy'], 'oob_score': [True], 'n_jobs': [1], 'class_weight': [None]}

    grid = test_model(model, param_grid, scale = True)

    output = open('gridRF.pk1','wb')
    pickle.dump(grid, output)
    output.close()
