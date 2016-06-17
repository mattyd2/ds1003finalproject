# ds1003finalproject
### Authors:
Matthew Dunn, Rafael Garcia Cano Da Costa, and Benjamin Jakubowski
### Project Description:
This repository is for a final course project for NYU DS1003: Machine Learning and Computational Statistics. The objective of the project was to develop a predictive model to predict the outcome of US Immigration Court decisions on asylum applications. This repository contains code for
- Data cleaning and merging
- Model training, evaluation, and visualization.

## Data cleaning and merging

### Usage:
#### Step 1 -- Confirm valid file paths for the following files:

1. `bios_clean2.csv`
2. `features_to_keep.text`
3. `decision_scheduling_merge_final_converted.csv`
4. `cleaned_with_features.csv` - created by main.py
5. `train.csv` - created by main.py
6. `test.csv` - created by main.py

#### Step 2 -- Execute the following scripts

1. `python main.py`
2. `python merge_judge.py`

## Model training, evaluation, and visualization.

### Usage Notes:
1. The modeling module encapsulates generic functionality for model training, evaluation, and visualization.
2. The model families are encapsulated in separate modules, which call functions from the generic modeling module. 
3. Structure allows each author to independently optimize hyperparameters using the same grid search framework.

## Scripts to Review:
###Data cleaning and merging:
1. `feature_engineering.py` - this module contains functions that handle the bulk of the feature engineering, i.g., creating lag variables, dummy variables, and cleaning up time formatting issues.  
2. `merge_judge.py` - see TODO
3. `data_cleaning.py` - handles cleaning raw data.
4. `general_modeling_function.py` - useful abstraction so team members can each work on differnt models with same random seed.
5. `plotting.py` - used for plotting results.
6. `__main__.py` - in any of the example models directories to understand usage.

## TODO:
 `merge_judge.py` is currently bloated- it was intended to simply merge in the clean judge bios, but following this merge a number of issues were identified with the resulting data set (which was indended to be consumed by sklearn model.fit methods).

As written, this script currently:

1. Drops asylum application records with notice to appear before the year 2000.
2. Merges the post-1999 asylum application data with the judge bios.
3. Converts judge bio time features to years passed.
4. Drops a number of duplicate features/features retained to ensure data integrity.
5. Creates additional dumy variables missed in the `main.py` first pass cleaning script.
6. Creates additional lag features for percent of each decision type, in addition to the count features created by the `main.py` first pass cleaning script.
7. Splits training and test sets, and saves each set as a .csv file.

