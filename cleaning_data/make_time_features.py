import pandas as pd
import numpy as np
from datetime import timedelta
import datetime

### Read in all data and keep only allowed features

features_to_keep = pd.read_csv('features_to_keep.txt', header=None)
features_to_keep = np.ravel(features_to_keep.values)

courts_data = pd.read_csv('./../data/decision_scheduling_merge_final_converted.csv', nrows=10000, usecols=features_to_keep)



def stata_edate_to_pd_datetime(edate):
    '''
    Converts elapsed days (STATA date format: number of days since
    01-01-1960) to datatime
    '''
    edate_datetime = pd.to_datetime('1960-01-01') + timedelta(days=edate)
    return edate_datetime

def make_hearing_edate_features(courts_data):
    '''
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
    tm -= datetime.timedelta(minutes=tm.minute % 30,
                         seconds=tm.second)
    return tm

def make_hearing_half_hour(courts_data):
    courts_data['start_time_half_hour'] = pd.to_datetime(courts_data['adj_time_start'].astype(int), format='%H%M')
    courts_data['start_time_half_hour'] = courts_data['start_time_half_hour'].apply(round_to_half_hour).apply(lambda x: x.time())
    return


make_hearing_edate_features(courts_data)
make_hearing_half_hour(courts_data)
