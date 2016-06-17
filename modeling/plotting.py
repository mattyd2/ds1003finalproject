import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, svm
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.preprocessing import StandardScaler


def plot_auc(pickle_path, test_path, train_path, model_name):

    print 'Reading pickle file.'
    grid = read_pickle_file(pickle_path)

    print 'Reading train and test files.'
    test_data, train_data = read_dataset(test_path, train_path)

    print 'Preparing data.'
    X_test_scaled, y_test = prepare_dataset(test_data, train_data)

    print 'Plotting AUC.'
    plot_ROC(y_test, grid.predict_proba(X_test_scaled)[:, 1], model_name)
    plt.savefig('./RF_ROC.png')
    plt.close('all')
    print 'AUC saved locally.'


def read_pickle_file(file_path):
    grid = open(file_path, 'rb')
    grid_read = pickle.load(grid)
    grid.close()
    return grid_read


def read_dataset(file_path_test, file_path_train):
    test_data = pd.read_csv(file_path_test)
    train_data = pd.read_csv(file_path_train)
    return test_data, train_data


def prepare_dataset(test_data, train_data):
    x_cols = np.setdiff1d(test_data.columns, ['appl_dec_G', 'appl_dec_D', 'appl_dec_F', 'appl_dec_L'])

    X_test = test_data[x_cols]
    y_test = test_data['appl_dec_G']

    X_train = train_data[x_cols]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test


def plot_ROC(y_true, y_predicted, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
    roc_auc = auc(fpr, tpr)
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c, label=model_name+' (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Set ROC')
    plt.legend(loc="lower right")


def make_learning_curve_from_gridsearchcsv(model, hyperparm, filename):
    '''
    Make learning curve from fit gridsearchcv sklearn object.

    Args:
        -model: gridsearchcv object from sklearn
        -hyperparm: string, hyperparameter to see learning curve for. Ex: from LinearSVC, 'C'
        -file_name: file name to save plot
    '''
    means = [np.mean(x[2]) for x in model.grid_scores_]
    sds = [np.std(x[2]) for x in model.grid_scores_]
    plus_sd = [np.mean(x[2]) + np.std(x[2]) for x in model.grid_scores_]
    minus_sd = [np.mean(x[2]) - np.std(x[2]) for x in model.grid_scores_]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(model.param_grid[hyperparm], means, color='blue', lw=2)
    ax.errorbar(model.param_grid[hyperparm], means,yerr=sds, fmt='o')
    ax.fill_between(model.param_grid[hyperparm], plus_sd, minus_sd, facecolor='#F0F8FF', alpha=1.0, edgecolor='none')
    ax.set_xscale('log')
    ax.set_xlabel(hyperparm)
    ax.set_ylabel('ROC AUC')
    ax.set_title('Optimizing {} in Linear SVC\nAsylum Court Grant Decisions'.format(hyperparm))
    plt.savefig('{}.png'.format(file_name))
    return
