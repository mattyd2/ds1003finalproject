# This code drop 10 rows where feature 'case_type' has values not in master_case_type.csv,
# replace some NaN values as a new categorical value, drop features that are not part of the modeling,
# code 'attorney_flag' not equals '1' as '0'

import pandas as pd
import numpy as np
from numba import jit

data = pd.DataFrame()

@jit
def Clean_Data(feature_keepcsv, decision_scheduling_merge_final_convertedcsv, data):
    data = Drop_Features(feature_keepcsv, decision_scheduling_merge_final_convertedcsv)
    NaN_Into_Code(0, 'attorney_flag', data)
    NaN_Into_Code('ZZ', 'nat', data)
    data = Drop_Row_Value(['01', '06', '07', '02', '04'], 'case_type', data)
    NaN_Into_Code('E_or_I', 'c_asy_type', data)
    NaN_Into_Code('ZZZ', 'hearing_loc_code', data)
    NaN_Into_Code(000., 'langid', data)
    return data

@jit
def Drop_Row_Value(values, feature, data):
    for i in values:
        data = data[data[feature] != i]
    return data

@jit
def NaN_Into_Code(code, feature, data):
    return data[feature].fillna(code, inplace=True)

@jit
def Drop_Features(feature_keepcsv, decision_scheduling_merge_final_convertedcsv):
    features_keep = Read_Csv_Pandas(feature_keepcsv)
    data = Read_Dataset(decision_scheduling_merge_final_convertedcsv)
    return data[features_keep[0].unique()]
    
@jit
def Read_Csv_Pandas(feature_drop):
    features_keep = pd.read_csv(feature_drop, sep="\n", header = None, low_memory=False)
    return features_keep

@jit
def Read_Dataset(decision_scheduling_merge_final_converted):
    data = pd.read_csv(decision_scheduling_merge_final_converted, sep=",", low_memory=False)
    return data
