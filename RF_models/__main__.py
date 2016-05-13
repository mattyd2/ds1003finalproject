from sklearn.ensemble import RandomForestClassifier
import sys
import pickle
from modeling.atty_interactions import *
from modeling.learning_curve import *
from modeling.general_modeling_function import *
from modeling.plot_auc import *

if __name__ == '__main__':

    ### This is for my RandomForestClassifier_model
    model = RandomForestClassifier
    param_grid = {'n_estimators': [100, 300, 600, 900], 'criterion': ['entropy']}
    grid = test_model(model, param_grid, scale = True)

    make_learning_curve_from_gridsearchcsv(grid, 'n_estimators', 'learning_curve_RF10')
    

    output = open('RF10.pk1','wb')
    pickle.dump(grid, output)
    output.close()
    
    plot_auc('./RF10.pk1', './test.csv', './train.csv', 'Random Forest')
