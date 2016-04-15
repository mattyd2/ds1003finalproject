import numpy as np
import pandas as pd
import os
from dataloader import loader
from judgebioextractor import mergeWriteJudgeToClean
from data_cleaning import Clean_Data
from bioanalyzer import analyzetext


def main():
    Clean_Data(feature_keepcsv, decision_scheduling_merge_final_convertedcsv, data)
    df = loader()
    # bioAndgrantrate = judgeAnalyzer(df)
    # csvwriter(bioAndgrantrate)
    analyzetext(df)

if __name__ == "__main__":
    main()
