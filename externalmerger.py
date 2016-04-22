import pandas as pd
import numpy as np
import re


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
    raw = pd.read_csv('../data/tblLookupNationality.csv')
    # https://drive.google.com/open?id=0B-7pzj2oZ2Tja19PUTh6aTlBSjA
    raw_conflict = pd.read_csv('../data/gcri_data_v5.0.1.csv')
    raw_merged = pd.read_csv('../data/cleaned_with_features.csv', nrows=10000)
    raw_merged = raw_merged[raw_merged.osc_year >= 2000]
    print "+_+_+_+_+ The data has been loaded successfuly +_+_+_+_+_ \n"

    #Add back in the NAT col to raw_merged

    nat_col = [c for c in raw_merged.columns if re.match(r'nat_(([A-Z]{2})|(\?{2}))',c)]


    raw_merged['nat'] = raw_merged[nat_col].idxmax(axis=1).apply(lambda x: x.strip('nat_').upper())

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

    to_drop = [x for x in final_merged.columns if x.endswith('_y')]
    final_merged.drop(to_drop, axis=1, inplace=True)

    final_merged.rename(columns={'nat_x':'nat'}, inplace=True)

    print "+_+_+_+_+ final data and conflict data have been merged +_+_+_+_+_ \n"

    # Merge the gdp with the final_merged

    print "+_+_+_+_+ Merging GDP data +_+_+_+_+_ \n"


    gdp = pd.read_csv('../data/gdp.csv')
    years = gdp.columns[2:]
    countries = gdp['nat']
    final_merged.insert(final_merged.shape[0], 'gdp', np.nan)
    for place in countries:
        for year in years:
            if len(gdp.loc[gdp['nat'] == place,year].values) > 0:
                final_merged.loc[np.logical_and((final_merged['nat']==place), (final_merged['osc_year']==int(year))), 'gdp'] = gdp.loc[gdp['nat'] == place,year].values[0]


    final_merged.drop('nat', axis=1, inplace=True)


    print "+_+_+_+_+ Merging bios  data +_+_+_+_+_ \n"

    bios = pd.read_csv('../data/bios_clean2.csv', index_col=0)
    final_merged = pd.merge(final_merged, bios, on='ij_code', how='left')

    print len(final_merged)
    print final_merged[['CORRUPT', 'gdp']].describe()

    #final_merged.to_csv('./../data/merged_externals.csv', index=False)

if __name__ == '__main__':
    conflictMerger()
