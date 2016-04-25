from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def test_model(model, param_grid, scale = True):
    '''
    Function for grid search optimization of hyperparameters.

    Args:
        model - sklearn model
        param_grid - dictionary of hyperparameters for grid search
        scale - Boolean indicating if the X_train should be scaled using standard scaler. Use for regularized linear models (SVC, log reg)

    Returns:
        grid - the fit, scored GridSearchCV object
    '''

    print 'Reading training data'

    train_data = pd.read_csv('./../data/final_data/train.csv')

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
    grid = GridSearchCV(model(), param_grid=param_grid, scoring='roc_auc',cv=5)
    grid.fit(X_train_scaled, y_train)

    return grid

if __name__ == '__main__':

    ### This is for my svc_model
    model = LinearSVC
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = {'C': Cs}

    grid = test_model(model, param_grid, scale = True)

    output = open('grid.pk1','wb')
    pickle.dump(grid, output)
    output.close()
