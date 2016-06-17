# cleaning_data subdirectory in ds1003finalproject
### Authors:
Matthew Dunn, Rafael Garcia Cano Da Costa, and Benjamin Jakubowski
### Description:
This subdirectory contains scripts for data cleaning and merging. Note this code base is not fully modular, and would benefit from refactoring (TODO). As such, instead of simply calling the ```main.py``` script, or executing as a module, the data was cleaned through sequential execution of the individual scripts in this directory.

As such, the usage section in this README describes:

1. The purpose/functionality of each cleaning script.
2. The order of execution of the cleaning scripts.

## Usage

### Step 1

#### Confirm valid file paths for the following files:

1. `bios_clean2.csv`
2. `features_to_keep.text`
3. `decision_scheduling_merge_final_converted.csv`
4. `cleaned_with_features.csv`
5. `train.csv`
6. `test.csv`

### Step 2
#### Execute the following scripts

1. `python main.py`
2. `python merge_judge.py`

## Notes:
#### `merge_judge.py` is currently bloated- it was intended to simply merge in the clean judge bios, but following this merge a number of issues were identified with the resulting data set (which was indended to be consumed by sklearn model.fit methods). Thus, this script would benefit from refactoring (TODO).

As written, this script currently:

1. Drops asylum application records with notice to appear before the year 2000.
2. Merges the post-1999 asylum application data with the judge bios.
3. Converts judge bio time features to years passed.
4. Drops a number of duplicate features/features retained to ensure data integrity.
5. Creates additional dumy variables missed in the `main.py` first pass cleaning script.
6. Creates additional lag features for percent of each decision type, in addition to the count features created by the `main.py` first pass cleaning script.
7. Splits training and test sets, and saves each set as a .csv file.