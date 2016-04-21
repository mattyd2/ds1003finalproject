import pandas as pd
import numpy as np


def conflictMerger():
    '''Function merges historical conflict index sheet with final prepped decision
    data from dr. chen.  It uses the nationality lookup table to get the
    country full name, then merges the nat code into the conflict sheet.  Then
    creates unique year nat key by concatenating the nat and osc_year field
    which we then use to merge the conflict data with final sheet

    inputs: currently NONE, could pass in the decision merged file as df

    returns: could return dataframe or could write to csv'''

    # Read in the data
    # https://drive.google.com/open?id=0B-7pzj2oZ2TjUGl6dE1kdlF0MGs
    raw = pd.read_csv('../raw_data_dr_chen/tblLookupNationality.csv')
    # https://drive.google.com/open?id=0B-7pzj2oZ2Tja19PUTh6aTlBSjA
    raw_conflict = pd.read_csv('../conflict_data/gcri_data_v5.0.1.csv')
    raw_merged = pd.read_csv('../_decision_scheduling_merge_final_converted_1000.csv')
    print "+_+_+_+_+ The data has been loaded successfuly +_+_+_+_+_ \n"

    # Prepare the nationality merge key
    nat_mergekey = raw[['??', 'UNKNOWN COUNTRY']]
    nat_mergekey.columns = ['nat', 'COUNTRY']

    # Set all country names to be upper case
    raw_conflict.COUNTRY = raw_conflict.COUNTRY.str.upper()

    # Join the conflict index data and the nationality merge key
    conflict_prepped = raw_conflict.merge(nat_mergekey, on='COUNTRY')
    print "+_+_+_+_+ The conflict data and nationality merge key have been joined +_+_+_+_+_ \n"

    # Check if "nat" is blank to ensure quality merge
    blanks = np.where(pd.isnull(conflict_prepped.nat))
    if blanks[0].size == 0:
        print "+_+_+_+_+ The nat merge key merged succesfully with conflict data +_+_+_+_+_ \n"
    else:
        print "+_+_+_+_+ Blanks values detected in nat merge...please review +_+_+_+_+_ \n"

    # Create Merge Key for Nat and Year
    conflict_prepped['nat_app_year'] = conflict_prepped.nat+conflict_prepped.YEAR.astype(str)
    raw_merged['nat_app_year'] = raw_merged.nat+raw_merged.osc_year.astype(str)

    # Join raw_merged with conflict prepped on nat_app_year
    final_merged = pd.merge(raw_merged, conflict_prepped, on='nat_app_year', how='left')
    print "+_+_+_+_+ final data and conflict data have been merged +_+_+_+_+_ \n"
    return final_merged

if __name__ == '__main__':
    conflictMerger()
