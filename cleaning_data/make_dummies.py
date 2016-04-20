import pandas as pd
import numpy as np
from datetime import timedelta
import datetime

def make_dummies(courts_data, categoricals):
    '''
    Makes all dummies for categorical features- note user needs to drop one of each of the dummy features for regression.

    Args:
        courts_data: the cleaned courts dataset
        categoricals: a list of cateogorical features to code as dummies

    Returns:
        Courts_data with dummies appended.
    '''
    for cat in categoricals:
        print "Making dummies for: ", cat
        if courts_data[cat].dtype != float:
            courts_data = pd.concat([courts_data, pd.get_dummies(courts_data[cat].astype(str), prefix=cat, dummy_na=True)], axis=1)

        elif courts_data[cat].dtype == float:
            courts_data = pd.concat([courts_data, pd.get_dummies(courts_data[cat].fillna(0.0).astype(int).astype(str), prefix=cat, dummy_na=True)], axis=1)
        courts_data.drop(cat, axis=1, inplace=True)
    return courts_data
