import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

def merge_judge_bios():

    ##Read in judge_bios and clean data

    judge_bios = pd.read_csv('./../data/bios_clean2.csv', usecols = ['Male_judge','Year_Appointed_SLR','Year_College_SLR', 'Year_Law_school_SLR', 'President_SLR','Government_Years_SLR', 'Govt_nonINS_SLR', 'INS_Years_SLR', 'Military_Years_SLR', 'NGO_Years_SLR', 'Privateprac_Years_SLR', 'Academia_Years_SLR','ij_code'])

    judge_bios['President_SLR'] = judge_bios['President_SLR'].apply(lambda x: str(x).replace(' ', '_'))
    judge_bios = pd.concat([judge_bios, pd.get_dummies(judge_bios['President_SLR'], prefix='president')], axis=1)

    judge_bios.drop('President_SLR', axis=1, inplace=True)

    print 'Origial shape judge bios: ', judge_bios.shape

    judge_bios.dropna(axis=0, how='any', inplace=True)

    print 'Cleaned judge bios shape: ', judge_bios.shape

    ##Read in cleaned_features, subset data by date.

    cleaned_features = pd.read_csv('./../data/cleaned_with_features.csv', dtype={'start_time_half_hour':str})

    print 'Origial shape cleaned_features: ', cleaned_features.shape

    cleaned_features = cleaned_features[cleaned_features['osc_year']>=2000]

    print '2000+ shape cleaned features pre merge: ', cleaned_features.shape

    #Merge bios and cleaned features

    merged = pd.merge(cleaned_features, judge_bios, on = 'ij_code', how='inner')

    ##Covert year features to time elapsed for linear modeling

    merged['Years_since_appointed'] = merged['osc_year'] - merged['Year_Appointed_SLR']
    merged['Years_since_college'] = merged['osc_year'] - merged['Year_College_SLR']
    merged['Years_since_law_school'] = merged['osc_year'] - merged['Year_Law_school_SLR']

    merged.drop(['comp_year', 'comp_month', 'comp_day', 'comp_date', 'Year_Appointed_SLR','Year_College_SLR','Year_Law_school_SLR', 'ij_code','tracid', 'adj_date_stamp', 'adj_time_start', 'osc_date','osc_day','osc_month','osc_year'], axis=1, inplace = True)

    print 'Post-merge shape 2000+ cleaned features: ', merged.shape

    ##Drop zero variance features (wasted space)

    for col in np.setdiff1d(merged.columns, ['appl_dec', 'start_time_half_hour']):
        if merged[col].var() == 0:
            merged.drop(col, axis=1, inplace=True)

    print 'Shape after dropping zero variance features: ', merged.shape

    ##Make final set of dummies:

    more_dummies = ['hearing_day_of_week', 'hearing_month', 'start_time_half_hour', 'appl_dec']

    for dummy in more_dummies:
        merged = pd.concat([merged, pd.get_dummies(merged[dummy], prefix=dummy)], axis=1)
        merged.drop(dummy, axis=1, inplace=True)

    print 'Shape after making dummies for day of week, month, half hour, and appl_dec: ', merged.shape

    ## Add features for G/D/F/L counts as percentages

    hist_feature_sets = [['nat_and_base_city_code_D_count_5_years', 'nat_and_base_city_code_G_count_5_years',     'nat_and_base_city_code_L_count_5_years'], ['nat_and_base_city_code_D_count_1_years','nat_and_base_city_code_G_count_1_years',    'nat_and_base_city_code_L_count_1_years'], [    'nat_and_c_asy_type_and_base_city_code_D_count_5_years',    'nat_and_c_asy_type_and_base_city_code_G_count_5_years',    'nat_and_c_asy_type_and_base_city_code_L_count_5_years',], [    'nat_and_c_asy_type_and_base_city_code_D_count_1_years',    'nat_and_c_asy_type_and_base_city_code_G_count_1_years', 'nat_and_c_asy_type_and_base_city_code_L_count_1_years'], [    'nat_and_langid_and_c_asy_type_and_base_city_code_D_count_5_years',    'nat_and_langid_and_c_asy_type_and_base_city_code_G_count_5_years',    'nat_and_langid_and_c_asy_type_and_base_city_code_L_count_5_years'],[    'nat_and_langid_and_c_asy_type_and_base_city_code_D_count_1_years',    'nat_and_langid_and_c_asy_type_and_base_city_code_G_count_1_years',    'nat_and_langid_and_c_asy_type_and_base_city_code_L_count_1_years']]

    for feature_set in hist_feature_sets:
        feature_totals = np.sum(merged[feature_set], axis=1)
        feature_totals = feature_totals.apply(lambda x: 1 if x == 0 else x)
        for feature in feature_set:
            name = feature.replace('_count_','_percent_')
            merged[name] = merged[feature] / feature_totals

    print 'Shape after adding historic percentage features: ', merged.shape

    print 'Final check for NaN values: ', np.sum(merged.isnull().values)

    ##split into training/test and save.
    ##NOTE FOR ANALYSIS WE WILL NEED TO DROP appl_dec_D, appl_dec_F, and appl_dec_L (for binary analysis). This features are retained for multiclass classification problem

    print 'Saving training/test split'

    train, test = train_test_split(merged, test_size=0.2)


    ##Save final training/test datasets-
    train.to_csv('./../data/final_data/train.csv', index=False)
    test.to_csv('./../data/final_data/test.csv', index=False)

    print 'Final training/test split shapes: ', train.shape, test.shape

if __name__ == '__main__':
    merge_judge_bios()
