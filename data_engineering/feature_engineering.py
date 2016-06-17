# module to hold all of the feature engineering functions in single file

import pandas as pd
import numpy as np
from datetime import timedelta
import datetime
from sklearn.cross_validation import train_test_split


# makes the dummies for each categorical variable included in the data
def make_dummies(courts_data, categoricals):
    '''
    Purpose:
        Makes all dummies for categorical features- note user needs to
        drop one of each of the dummy features for regression.
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


# Get recent history features
def add_history_features_to_courts_data(courts_data, dict_of_groupbys):
    '''
    Purpose:
        Appends historical distributions to the courts_data. Note this is done
        in place- mutates the input courts_data data frame.
    Args:
        courts_data: dataframe of courts data
        dict_of_groupbys: A dictionary mapping tuples of groupby features to
        a list of number of years to include for that set.
            Ex: {('natid','base_city_code'):[3,1], ('base_city_code'):[5,1]}
    Returns:
        None (note this mutates the input courts_data)
    '''
    for feature_set, n_years in dict_of_groupbys.items():
        grouped = courts_data.groupby(list(feature_set))
        for index, row in courts_data.iterrows():
            row_key = tuple(row[feature] for feature in feature_set)
            if len(row_key) == 1:
                row_key = row_key[0]

            similar_cases = grouped.get_group(row_key)
            for year in n_years:
                similar_cases = similar_cases[similar_cases.comp_date < row['osc_date']]
                similar_cases = similar_cases[similar_cases.comp_date >= (row['osc_date'] - 365*year)]

                dist_of_decisions = similar_cases.appl_dec.value_counts()

                string_features = "_and_".join(feature_set)

                dist_of_decisions.rename(lambda decision: '%s_%s_count_%i_years' % (string_features, decision, year), inplace=True)

                for i in dist_of_decisions.index:
                    courts_data.loc[index, i] = dist_of_decisions[i]

            if index % 1000 == 0:
                print 'Adding history features to record = %i' % (index)

    return courts_data


def stata_edate_to_pd_datetime(edate):
    '''
    Converts elapsed days (STATA date format: number of days since
    01-01-1960) to datatime
    '''
    edate_datetime = pd.to_datetime('1960-01-01') + timedelta(days=edate)
    return edate_datetime


def make_hearing_edate_features(courts_data):
    '''
    Purpose:
        Mutates the courts_data, adding a hearing_day_of_week, hearing month,
        feature to the data frame:
    Args:
        courts_data: The courts data as a pandas dataframe
    Returns:
        None- mutate in place
    '''
    edate_datetime = courts_data['adj_date_stamp'].apply(stata_edate_to_pd_datetime)

    courts_data['hearing_day_of_week'] = edate_datetime.apply(lambda x: x.dayofweek)
    courts_data['hearing_month'] = edate_datetime.apply(lambda x: x.month)
    return


def round_to_half_hour(tm):
    tm += datetime.timedelta(minutes=15)
    tm -= datetime.timedelta(minutes=tm.minute % 30, seconds=tm.second)
    return tm


def make_hearing_half_hour(courts_data):
    courts_data['start_time_half_hour'] = pd.to_datetime(courts_data['adj_time_start'].astype(int), format='%H%M')
    courts_data['start_time_half_hour'] = courts_data['start_time_half_hour'].apply(round_to_half_hour).apply(lambda x: x.time())
    return
