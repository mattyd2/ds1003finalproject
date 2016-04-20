import pandas as pd
import numpy as np


### Get recent history features

def make_history_features(courts_data, groupby_features, row, n_years):
    '''
    Takes a list of features to groupby, a current osc_date, and n_years,
    and generates a distribution of decisions for those group_by features
    during the period from osc_date - n_years to osc_date.

    Args:
        -courts_data: The courts data in a pandas courts_data
        -groupby_features: A set of features to groupby
        -row: The current record used for feature generation
        -n_years: Time window for distribution generation- generate
        distribution from osc_date - n_years to osc_date.

    Returns:
        -dist_of_decisions: A pandas series with the distribution of
        decision types.
    '''

    target_data = courts_data[courts_data[groupby_features[0]] == row[groupby_features[0]]]

    if len(groupby_features) > 1:
        for feature in groupby_features[1:]:
            target_data = target_data[target_data[feature] == row[feature]]

    target_data = target_data[target_data.comp_date < row['osc_date']]
    target_data = target_data[target_data.comp_date >= (row['osc_date'] - 365*n_years)]


    dist_of_decisions = target_data.appl_dec.value_counts()

    string_features = "_and_".join(groupby_features)

    dist_of_decisions.rename(lambda decision: '%s_%s_count_%i_years' \
                             % (string_features, decision, n_years), inplace=True)
    return dist_of_decisions



def add_history_features_to_courts_data(courts_data, dict_of_groupbys):
    '''
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

    for index, row in courts_data.iterrows():
        for feature_set, list_of_n_years in dict_of_groupbys.items():
            for n_years in list_of_n_years:
                history = make_history_features(courts_data, feature_set, row, n_years)
                for feature in history.index:
                    courts_data.loc[index, feature] = history[feature]
        if index % 500 == 0:
            print 'Adding history features to record = %i' % (index)
    return
