import numpy as np
import pandas as pd
import os
from data_cleaning import clean_data
from cleaning_data import make_dummies, make_history_features, make_time_features

dict_of_groupbys = {('nat', 'base_city_code'): [5,1], ('nat', 'c_asy_type', 'base_city_code'): [5,1], ('nat','langid', 'c_asy_type', 'base_city_code'): [5,1]c


categoricals = ['nat', 'case_type', 'appl_code', 'c_asy_type', 'base_city_code', 'hearing_loc_code','attorney_flag', 'schedule_type', 'langid']


def cleaner():
    # put the full path name to these files
    features_to_keep = './cleaning_data/features_to_keep.txt'

#    decision_scheduling_merge = './../data/dsmfc_short.csv'
    decision_scheduling_merge = './../data/decision_scheduling_merge_final_converted.csv'

    # clean the data
    print 'Cleaning data with courts_data'
    courts_data = clean_data(features_to_keep, decision_scheduling_merge)

    # make features
    print 'Making history features add_history_features_to_courts_data'
    make_history_features.add_history_features_to_courts_data(courts_data, dict_of_groupbys)
    courts_data.fillna(0, inplace=True)
    print courts_data.head(6)

    print 'Making dummy features make_dummies'
    courts_data = make_dummies.make_dummies(courts_data, categoricals)

    print 'Making time features make_hearing_edate_features, make_hearing_half_hour'
    make_time_features.make_hearing_edate_features(courts_data)
    make_time_features.make_hearing_half_hour(courts_data)

    courts_data.to_csv('./../data/cleaned_with_features.csv', index=False)


def main():
    cleaner()

if __name__ == "__main__":
    main()
