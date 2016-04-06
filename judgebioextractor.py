import numpy as np
import pandas as pd
import os


def mergeWriteJudgeToClean():
    judgeAndKeys = joinJudgeAndKey()
    asylumProcedings = pd.read_csv('/Users/matthewdunn/Dropbox/NYU/Spring2016/MachineLearning/project/asylum_clean.csv')
    asylumProcedings = asylumProcedings.merge(judgeAndKeys, on="ij_code")
    asylumProcedings.to_csv('asylum_clean_judge_text.csv')


def joinJudgeAndKey():
    judgeJoinKeys = pd.read_csv('/Users/matthewdunn/Dropbox/raw/ijcodemap.csv')
    judgeDescripts = getJudgeFiles()
    judgeJoinKeys.columns = ['ij_code', 'judgefullname', 'tracid']
    judgeDescripts = judgeDescripts.merge(judgeJoinKeys, on='tracid')
    return judgeDescripts


def getJudgeFiles():
    directoryName = '/Users/matthewdunn/Dropbox/raw/bio/'
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
