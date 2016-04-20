# drops 10 rows where 'case_type' has values not in master_case_type.csv
# replace some NaN values as a new categorical value,
# drop features that are not part of the modeling,
# code 'attorney_flag' not equals '1' as '0'

import pandas as pd
import numpy as np
from numba import jit


@jit
def clean_data(features, decision_scheduling):
    data = Drop_Features(features, decision_scheduling)
    data = Drop_Row_Value(['01', '06', '07', '02', '04'], 'case_type', data)
    data = NaN_Into_Code(data)
    return data

@jit
def Drop_Row_Value(values, feature, data):
    for i in values:
        data = data[data[feature] != i]
    return data

@jit
def Drop_Features(features, decision_scheduling):
    features_keep = pd.read_csv(features, sep="\n",
                                header=None, low_memory=False)
    data = pd.read_csv(decision_scheduling, sep=",", low_memory=False)
    return data[features_keep[0].unique()]

@jit
def NaN_Into_Code(data):
     nan_features = ['attorney_flag', 'nat', 'c_asy_type', 'hearing_loc_code',
                     'langid']
     nan_codes = [0, 'ZZ', 'E_or_I', 'ZZZ', 000.]
     for i in range(len(nan_features)):
         data[nan_features[i]].fillna(nan_codes[i], inplace=True)
     return data
