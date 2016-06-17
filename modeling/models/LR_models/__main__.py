from sklearn.linear_model import LogisticRegression
import sys
import pickle
from modeling.atty_interactions import *
from modeling.learning_curve import *
from modeling.general_modeling_function import *
from modeling.plot_auc import *


if __name__ == '__main__':

    ### This is for my LogisticRegression_model
    model = LogisticRegression
    Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    param_grid = {'C': Cs, 'penalty':'l2'}

    ## L2 reg with base feature set
    grid = test_model(model, param_grid, scale = True)
    make_learning_curve_from_gridsearchcsv(grid, 'C', 'learning_curve_l2_log_reg')
    output = open('l2_log_reg.pk1','wb')
    pickle.dump(grid, output)
    output.close()

    ## L2 reg with interactions feature set

    ##First set recursion limit to allow for feature construction
    sys.setrecursionlimit(5000)

    grid = test_model(model, param_grid, scale = True, model_spec_function = make_atty_interactions_df)
    make_learning_curve_from_gridsearchcsv(grid, 'C', 'learning_curve_l2_log_reg_atty_interactions')
    output = open('l2_log_reg_atty_interactions.pk1','wb')
    pickle.dump(grid, output)
    output.close()

    ## L1 reg with base feature set
    param_grid = {'C': Cs, 'penalty':['l1']}

    grid = test_model(model, param_grid, scale = True)
    make_learning_curve_from_gridsearchcsv(grid, 'C', 'learning_curve_l1_log_reg')
    output = open('l1_log_reg.pk1','wb')
    pickle.dump(grid, output)
    output.close()

    ## L1 reg with interactions feature set
    grid = test_model(model, param_grid, scale = True, model_spec_function = make_atty_interactions_df)
    make_learning_curve_from_gridsearchcsv(grid, 'C', 'learning_curve_l1_log_reg_atty_interactions')
    output = open('l1_log_reg_atty_interactions.pk1','wb')
    pickle.dump(grid, output)
    output.close()
    
    plot_auc('./l2_log_reg.pk1', './test.csv', './train.csv', 'LogReg')
