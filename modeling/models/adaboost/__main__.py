from sklearn.ensemble import AdaBoostClassifier
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

    sys.setrecursionlimit(5000)

    grid = test_model(model, param_grid, scale=True, model_spec_function=make_atty_interactions_df)
    output = open('Ada_atty_interactions.pkl', 'wb')
    make_learning_curve_from_gridsearchcsv(grid, 'n_estimators', 'learning_curve_Ada_atty_interactions')
    pickle.dump(grid, output)
    output.close()
