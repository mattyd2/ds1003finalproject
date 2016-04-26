import pickle
import pandas as pd
import numpy as np
import modeling_framework as mf
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def adaboostsetup(pickled_model_name):
    # This is for ada_boost
    tdclf = DecisionTreeClassifier(random_state=11, max_features="auto",
                                   class_weight="balanced", max_depth=None)
    model = AdaBoostClassifier(base_estimator=tdclf)
    n_estimators = [1000, 800, 600, 400, 300, 200, 100]
    param_grid = {'n_estimators': n_estimators, 'random_state': [4590385]}

    grid = grid_test_executor(model, param_grid, pickled_model_name)

    # to be used in the learning curve plotting
    return 'n_estimators', grid


def svcsetup(pickled_model_name):
    # This is for my svc_model
    model = LinearSVC
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = {'C': Cs}

    grid = grid_test_executor(model, param_grid, pickled_model_name)

    # to be used in the learning curve plotting
    return 'Cs', grid


def grid_test_executor(model, param_grid, pickled_model_name):

    grid = mf.test_model(model, param_grid, scale=True)

    output = open(pickled_model_name, 'wb')
    pickle.dump(grid, output)
    output.close()

    return grid
