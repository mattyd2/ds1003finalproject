# Read the train data
import pandas as pd
import numpy as np
from numba import jit
import os
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def loader():
    fileName = filetoload()
    currntpath = os.path.dirname(os.path.realpath('__file__'))
    fdir = os.path.abspath(os.path.join(currntpath, os.pardir))
    raw_data = pd.read_csv(fdir+'/'+fileName)
    return raw_data


def filetoload():
    # fileName = raw_input('What is the name of the file to be loaded?')
    fileName = 'bioAndGrantRate.csv'
    return fileName


# def prepdffordummy(df, labels):
#     tmpdf = df.drop(labels, axis=1, inplace=False)
#     tmpdf.replace(r'\s+', np.nan, regex=True)
#     return tmpdf


# def dummyconverter(df, dummycolumns):
#     prefixes = {el:el for el in dummycolumns}
#     tmpdf = pd.get_dummies(df, prefix=prefixes, prefix_sep='_', dummy_na=True, columns=dummycolumns)
#     print type(tmpdf)
#     return tmpdf


def csvwriter(dftowrite):
    filename = raw_input('What do you want to call this csv? Please enter "filename.csv" \n')
    # need to validate input
    currntpath = os.path.dirname(os.path.realpath('__file__'))
    fdir = os.path.abspath(os.path.join(currntpath, os.pardir))
    dftowrite.to_csv(fdir+"/"+filename)
    print filename+" has been saved "+fdir+"/"+filename
