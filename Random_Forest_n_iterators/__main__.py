from sklearn.ensemble import RandomForestClassifier
import sys
import pickle
from modeling.atty_interactions import *
from modeling.learning_curve import *
from modeling.general_modeling_function import *

if __name__ == '__main__':

    ### This is for my RandomForestClassifier_model
    model = RandomForestClassifier
    param_grid = {'n_estimators': [100, 200, 300], 'criterion': ['entropy'], 'oob_score': [True], 'class_weight': [None]}

    grid = test_model(model, param_grid, scale = True)

    make_learning_curve_from_gridsearchcsv(grid, 'n_estimators', 'learning_curve_RF1')

    output = open('RF.pk1','wb')
    pickle.dump(grid, output)
    output.close()
