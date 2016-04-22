
import pandas as pd
import numpy as np


### Get recent history features

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

    for feature_set, n_years in dict_of_groupbys.items():
        grouped = courts_data.groupby(list(feature_set))
        for index, row in courts_data.iterrows():
            row_key = tuple(row[feature] for feature in feature_set)
            if len(row_key) ==1:
                row_key = row_key[0]

            similar_cases = grouped.get_group(row_key)
            for year in n_years:
                similar_cases = similar_cases[similar_cases.comp_date < row['osc_date']]
                similar_cases = similar_cases[similar_cases.comp_date >= (row['osc_date'] - 365*year)]

                dist_of_decisions = similar_cases.appl_dec.value_counts()

                string_features = "_and_".join(feature_set)

                dist_of_decisions.rename(lambda decision: '%s_%s_count_%i_years' % (string_features, decision, year), inplace=True)

                for i in dist_of_decisions.index:
                    courts_data.loc[index,i] = dist_of_decisions[i]

            if index % 1000 == 0:
                print 'Adding history features to record = %i' % (index)

    return courts_data
