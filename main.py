import numpy as np
import pandas as pd
import os
from dataloader import loader
from judgebioextractor import mergeWriteJudgeToClean
from data_cleaning import clean_data
from bioanalyzer import analyzetext


def main():
    # put the full path name to these files
    features_to_keep = ''
    decision_scheduling_merge = '/Users/matthewdunn/Dropbox/NYU/Spring2016/MachineLearning/project/_decision_scheduling_merge_final_converted.csv'

    # clean the data
    clean_data(features_to_keep, decision_scheduling_merge, data)

    # df = loader()
    # bioAndgrantrate = judgeAnalyzer(df)
    # csvwriter(bioAndgrantrate)
    analyzetext(df)

if __name__ == "__main__":
    main()
