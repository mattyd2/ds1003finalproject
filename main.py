# Authors Benjamin Ulrich Jakubowski, Matthew Dunn, Rafael, Rafael Garcia Cano Da Costa
# Module used for program execution

import pickle
import numpy as np
import pandas as pd
import os
# from dataloader import loader
# from data_cleaning import clean_data
# from cleaning_data import make_dummies, make_history_features, make_time_features
import model_setup as md
import modeling_framework as mf


dict_of_groupbys = {('nat', 'base_city_code'): [5, 1],
                    ('nat', 'c_asy_type', 'base_city_code'): [5, 1],
                    ('nat', 'langid', 'c_asy_type', 'base_city_code'): [5, 1]}


categoricals = ['nat', 'case_type', 'appl_code', 'c_asy_type',
                'base_city_code', 'hearing_loc_code', 'attorney_flag',
                'schedule_type', 'langid']


def cleaner():
    # put the full path name to these files
    features_to_keep = './cleaning_data/features_to_keep.txt'

    # decision_scheduling_merge = './../data/dsmfc_short.csv'
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


def modelfitting(pickled_model_name):
    hyperparm, grid = md.adaboostsetup(pickled_model_name)


def modelplotting(pickled_model_name, hyperparm, model_type):
    pkl_file = open(pickled_model_name, 'rb')
    model = pickle.load(pkl_file)
    mf.make_learning_curve_from_gridsearchcsv(model, hyperparm, model_type)


def main():
    pickled_model_name = 'grid_ada2.pk1'
    hyperparm = 'n_estimators'
    model_type = 'AdaBoost Decision Stump'
    modelfitting(pickled_model_name)
    modelplotting(pickled_model_name, hyperparm, model_type)

if __name__ == "__main__":
    main()
