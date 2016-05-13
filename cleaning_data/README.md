# cleaning_data subdirectory in ds1003finalproject
### Authors:
Matthew Dunn, Rafael Garcia Cano Da Costa, and Benjamin Jakubowski
### Description:
This subdirectory contains scripts for data cleaning and merging. Note this code base is not fully modular, and would benefit from refactoring (TODO). As such, instead of simply calling the ```main.py``` script, or executing as a module, the data was cleaned through sequential execution of the individual scripts in this directory.

As such, the usage section in this README describes:

1. The purpose/functionality of each cleaning script.
2. The order of execution of the cleaning scripts.

## Usage

### Scripts with helper functions

The following scripts contain the specified helper functions for data cleaning and feature engineering:

|Script         | Functionality|
| --------------| -------------|
|`make_history_features.py` | Generate counts of each decision type over 1- and 5- year lag windows, at specified court for specific population|
|`make_time_features.py` | Generates day-of-week and scheduled hearing time features|
|`make_dummies.py` | Generates dummy variables from categorical features and appends to dataframe|
|`data_cleaning.py` | Drop all but specified features, drop rows with mis-entered values for case type (values not in lookup table), and re-code NaN values based on specific semantics of feature|

### First pass cleaning and feature generation script

Helper functions in these scripts are required by `main.py`, which is the **first** cleaning and feature generation script executed. This script reads in the decision_scheduling_merge_final_converted.csv file saves a new .csv file (cleaned_with_features.csv) following this inital cleaning and feature generation.

### Second pass cleaning and merging

After executing `main.py`, the **second** cleaning and feature generation script is `merge_judge.py`. This script is currently bloated- it was intended to simply merge in the clean judge bios, but following this merge a number of issues were identified with the resulting data set (which was indended to be consumed by sklearn model.fit methods). Thus, this script would benefit from refactoring (TODO).

As written, this script currently:

1. Drops asylum application records with notice to appear before the year 2000.
2. Merges the post-1999 asylum application data with the judge bios.
3. Converts judge bio time features to years passed.
4. Drops a number of duplicate features/features retained to ensure data integrity.
5. Creates additional dumy variables missed in the `main.py` first pass cleaning script.
6. Creates additional lag features for percent of each decision type, in addition to the count features created by the `main.py` first pass cleaning script.
7. Splits training and test sets, and saves each set as a .csv file.

Thus (assuming valid file paths for `bios_clean2.csv`, `features_to_keep.text`, `decision_scheduling_merge_final_converted.csv`, and saved files `cleaned_with_features.csv`, `train.csv`, and `test.csv`), execting the following scripts will save the final cleaned train/test data:

```
python main.py
python merge_judge.py
```

### Archive
Finally, in addition to the cleaning and merging scripts, cleaning_data contains a subdirectory archive, which contains unused code for
- loading data
- analyzing the raw judge bios (flat text files)
- merging external datasets related to country-level conflict and GDP.



