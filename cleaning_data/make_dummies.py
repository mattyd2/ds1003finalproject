import pandas as pd
import numpy as np
from datetime import timedelta
import datetime

categoricals = ['nat','case_type','appl_code','c_asy_type', 'base_city_code','hearing_loc_code','attorney_flag', 'schedule_type', 'langid']

features_to_keep = pd.read_csv('features_to_keep.txt', header=None)
features_to_keep = np.ravel(features_to_keep.values)

courts_data = pd.read_csv('./../data/decision_scheduling_merge_final_converted.csv', nrows=10000, usecols=features_to_keep)


def make_dummies(courts_data, categoricals):
    for cat in categoricals:
        print cat
        if courts_data[cat].dtype != float:
            courts_data = pd.concat([courts_data, pd.get_dummies(courts_data[cat].astype(str), prefix=cat, dummy_na=True)], axis=1)

        elif courts_data[cat].dtype == float:
            courts_data = pd.concat([courts_data, pd.get_dummies(courts_data[cat].fillna(0.0).astype(int).astype(str), prefix=cat, dummy_na=True)], axis=1)
        courts_data.drop(cat, axis=1, inplace=True)
    return courts_data


courts_data = make_dummies(courts_data, categoricals)
print cou
