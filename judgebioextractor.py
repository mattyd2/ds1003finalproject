import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def judgeAnalyzer(preppred_df):
    '''calls necessary functions to merge, group, and tokenize
    judge bio data.'''
    judgegrantrate = groupbybytracid(preppred_df)
    bioAndgrantrate = joinJudgeAndKey(judgegrantrate)
    return b


def groupbybytracid(preppred_df):
    ''' Function used to calculate the grant rate of an individual judge
    and return an dataframe with tracid to be merged in subsequent stepes'''
    df = preppred_df[['tracid', 'appl_dec']].copy()
    df['appl_decion_value'] = np.nan
    df.loc[df.appl_dec == 'D', 'appl_decion_value'] = 0.
    df.loc[df.appl_dec == 'F', 'appl_decion_value'] = 1.
    df.loc[df.appl_dec == 'G', 'appl_decion_value'] = 1.
    df.loc[df.appl_dec == 'L', 'appl_decion_value'] = 1.
    judgegrantrate = df.groupby('tracid', as_index=False).mean()
    return judgegrantrate


def joinJudgeAndKey(judgegrantrate):
    judgeDescripts = getJudgeFiles()
    judgeDescripts = judgeDescripts.merge(judgegrantrate, on='tracid')
    return judgeDescripts


def getJudgeFiles():
    currntpath = os.path.dirname(os.path.realpath('__file__'))
    fdir = os.path.abspath(os.path.join(currntpath, os.pardir))
    directoryName = fdir+'/raw_data_dr_chen/bio/'
    judgetext = []
    filenames = []
    for i in os.listdir(directoryName):
        filenames.append(i.split(".")[0])
        tempdirectoryName = directoryName+i
        f = open(tempdirectoryName, 'r')
        judgestring = f.read()
        judgestring = judgestring.replace('\n', ' ').replace('\r', '')
        judgetext.append(judgestring)
    df = pd.DataFrame(judgetext)
    df['tracid'] = filenames
    df['tracid'] = pd.to_numeric(df['tracid'], errors='coerce')
    df.columns = ['judgedescript', 'tracid']
    return df
