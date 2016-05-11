import pandas as pd
import re
from sklearn.preprocessing import PolynomialFeatures
from scipy import sparse

def clean_col_names(col_name):
    col_name = col_name.replace(':','_')
    col_name = col_name.replace('**','2stars')
    col_name = col_name.replace('??','2questions')
    col_name = col_name.replace('--','2minus')
    return col_name

def get_target_cols(train_file_path):

    train_data = pd.read_csv(train_file_path)

    train_data.rename(columns = lambda x: x.replace(':','_'), inplace = True)

    train_data.drop(['attorney_flag_0', 'appl_dec_D', 'appl_dec_F', 'appl_dec_L'], axis=1, inplace=True)

    ##Note the original dataset had column names with unnessary features
    ##Clean column names

    train_data.rename(columns = lambda x: clean_col_names(x), inplace = True)

    cols_for_prob_est = []
    for col in train_data.columns:
        if re.match('base_city_code', col) or re.match('attorney_flag',col) or re.match('nat_[^and]',col):
            cols_for_prob_est.append(col)

    X = train_data[cols_for_prob_est]
    sX = sparse.csr_matrix(X)

    return sX, train_data['appl_dec_G']
