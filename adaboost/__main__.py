from sklearn.svm import AdaBoostClassifier
import sys
import pickle
from atty_interactions import *
from learning_curve import *
from general_modeling_function import *

if __name__ == '__main__':

    # This is for ada_boost
    model = AdaBoostClassifier()
    n_estimators = [800, 400, 200, 100]
    param_grid = {'n_estimators': n_estimators, 'random_state': [4590385]}

    grid = test_model(model, param_grid, scale=True, model_spec_function=make_atty_interactions_df)
    make_learning_curve_from_gridsearchcsv(grid, 'C', 'learning_curve_Ada_atty_interactions')
    output = open('Ada_atty_interactions.pkl', 'wb')
    pickle.dump(grid, output)
    output.close()
