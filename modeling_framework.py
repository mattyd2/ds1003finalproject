# Authors Benjamin Ulrich Jakubowski, Matthew Dunn, Rafael, Rafael Garcia Cano Da Costa
# Abstracted utilities for testing models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV


def test_model(model, param_grid, scale=True):
    '''
    Function for grid search optimization of hyperparameters.

    Args:
        model - sklearn model
        param_grid - dictionary of hyperparameters for grid search
        scale - Boolean indicating if the X_train should be scaled using
        standard scaler. Use for regularized linear models (SVC, log reg)

    Returns:
        grid - the fit, scored GridSearchCV object
    '''

    print 'Reading training data'

    train_data = pd.read_csv('./../data/final_data/train.csv')

    x_cols = np.setdiff1d(train_data.columns, ['appl_dec_G', 'appl_dec_D',
                                               'appl_dec_F', 'appl_dec_L'])

    X_train = train_data[x_cols]
    y_train = train_data['appl_dec_G']

    if scale is True:
        print 'Scaling features'

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

    elif scale is False:
        X_train_scaled = X_train

    print 'Conducting grid search'
    np.random.seed(4590385)
    grid = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid.fit(X_train_scaled, y_train)

    return grid


def make_learning_curve_from_gridsearchcsv(model, hyperparm):
    '''
    Make learning curve from fit gridsearchcv sklearn object.

    Args:
        -model: gridsearchcv object from sklearn
        -hyperparm: string, hyperparameter to see learning curve for. Ex: from
        LinearSVC, 'C'
    '''
    means = [np.mean(x[2]) for x in model.grid_scores_]
    sds = [np.std(x[2]) for x in model.grid_scores_]
    plus_sd = [np.mean(x[2]) + np.std(x[2]) for x in model.grid_scores_]
    minus_sd = [np.mean(x[2]) - np.std(x[2]) for x in model.grid_scores_]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(model.param_grid[hyperparm], means, color='blue', lw=2)
    ax.errorbar(model.param_grid[hyperparm], means, yerr=sds, fmt='o')
    ax.fill_between(model.param_grid[hyperparm], plus_sd, minus_sd,
                    facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
    # ax.set_xscale('log')
    ax.set_xlabel(hyperparm)
    ax.set_ylabel('ROC AUC')
    ax.set_title('Optimizing {} in Linear SVC\nAsylum Court Grant Decisions'.format(hyperparm))
    plt.show()

    return
