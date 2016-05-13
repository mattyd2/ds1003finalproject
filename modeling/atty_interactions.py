import pandas as pd
from patsy import dmatrix
from numpy.linalg import matrix_rank
import re
import sys

def clean_col_names(col_name):
    col_name = col_name.replace(':','_')
    col_name = col_name.replace('**','2stars')
    col_name = col_name.replace('??','2questions')
    col_name = col_name.replace('--','2minus')
    return col_name

def make_atty_interactions_df(train_file_path):
    '''
    Description:

        Takes as input the file path for the train.csv data, and returns a new pandas dataframe with all attorney_flag_1 interactions (plus the target vector).

    Usage: To fit models with attorney interactions, import this script and call

        X_train, y_train = make_atty_interactions_df(train_file_path)
        where train_file_path is the relative path to the train.csv file.

    Args:
        train_file_path: the file path to the training data

    Returns:
        atty_interactions_train: the design matrix with the attorney_flag_1 interactions
        appl_dec_G: the target vector

    '''

    ##Read in training data and drop the multinomial targets

    train_data = pd.read_csv(train_file_path)

    train_data.rename(columns=lambda x: x.replace(':','_'), inplace = True)

    train_data.drop(['attorney_flag_0', 'appl_dec_D', 'appl_dec_F', 'appl_dec_L'], axis=1, inplace=True)

    ##Note the original dataset had column names with unnessary features
    ##Clean column names

    train_data.rename(columns=lambda x: clean_col_names(x), inplace = True)

    ##Specify model with interactions

    X_columns = ' + '.join(train_data.columns.difference(['appl_dec_G']))

    my_formula = '({})*attorney_flag_1'.format(X_columns)

    ##Create design matrix

    X_design_matrix = dmatrix(my_formula, train_data, return_type="dataframe")

    ##Return design matrix and targets

    y = train_data['appl_dec_G']

    return X_design_matrix, y
