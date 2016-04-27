from sklearn.svm import LinearSVC
import sys
import pickle
from modeling.atty_interactions import *
from modeling.learning_curve import *
from modeling.general_modeling_function import *

if __name__ == '__main__':

    ### This is for my svc_model
    model = LinearSVC
    Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = {'C': Cs}

    grid = test_model(model, param_grid, scale = True)
    make_learning_curve_from_gridsearchcsv(grid, 'C', 'learning_curve_svc1')
    output = open('svc1.pkl','wb')
    pickle.dump(grid, output)
    output.close()

    ##SVC with interactions feature set

    ##First set recursion limit to allow for feature construction
    sys.setrecursionlimit(5000)

    grid = test_model(model, param_grid, scale = True, model_spec_function = make_atty_interactions_df)
    make_learning_curve_from_gridsearchcsv(grid, 'C', 'learning_curve_svc1_atty_interactions')
    output = open('svc1_atty_interactions.pkl','wb')
    pickle.dump(grid, output)
    output.close()

