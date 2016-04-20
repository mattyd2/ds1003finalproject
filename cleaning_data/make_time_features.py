import pandas as pd
import numpy as np
from datetime import timedelta
import datetime


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
