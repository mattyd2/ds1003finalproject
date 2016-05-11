import numpy as np
import matplotlib.pyplot as plt

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

