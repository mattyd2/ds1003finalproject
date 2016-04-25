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
    fileName = raw_input('What is the name of the file to be loaded?')
    # fileName = 'bioAndGrantRate.csv'
    return fileName


def csvwriter(dftowrite):
    '''generic csv writing utility to be called when data needs to be written
    to a .csv file'''

    filename = raw_input('Name of file? Please enter "filename.csv" \n')
    currntpath = os.path.dirname(os.path.realpath('__file__'))
    fdir = os.path.abspath(os.path.join(currntpath, os.pardir))
    dftowrite.to_csv(fdir+"/"+filename)
    print filename+" has been saved "+fdir+"/"+filename
