# This code is exploring and cleaning file asylum_clean.csv
# This code is implementing a basic Log Reg as a baseline model using simple input such as the features ’hearing_loc_code’ and ‘natid’. The outcome is ‘grantraw’ 

#Read the train data.
import pandas as pd
from numba import jit

data = pd.read_csv('asylum_clean.csv')
data # 501053 rows

# Check if there is duplicated 'idncase' 
data_duplicated = data.duplicated('idncase', keep =False)
data_duplicated[data_duplicated==True]

# Visualize data sorted by 'idncase' and 'idnproceeding'
data_duplicated_sorted = data[data_duplicated].sort_values(by=['idncase', 'idnproceeding'])

# keep last proceeding on the same idncase
data_duplicated_sorted_keep_last = data_duplicated_sorted.duplicated('idncase', keep ='first')
duplicated_to_keep = data_duplicated_sorted[data_duplicated_sorted_keep_last==True] # 22783 rows
duplicated_to_keep # 22783 rows

# Drop all duplicated idncase from data
data_duplicated_idncase_removed = data.drop_duplicates('idncase', keep =False) 
data_duplicated_idncase_removed.shape # 456810 rows

# Count duplicated: all rows
data_duplicated[data_duplicated==True].shape # 44243 rows

# Merge data_duplicated_idncase_removed and duplicated_to_keep
data_no_duplicates = data_duplicated_idncase_removed.append(duplicated_to_keep, ignore_index=True)
data_no_duplicates.shape # 479593 rows matching the expected

# Check for missing values by column 'grant'
data_no_duplicates['grant'].isnull().values.any() # there are missing values!

# Check for missing values by column 'grantraw'
data_no_duplicates['grantraw'].isnull().values.any() # no NaN value
data_no_duplicates[data_no_duplicates['grantraw']==1] # 167912 rows with 'grantraw'==1
data_no_duplicates[data_no_duplicates['grantraw']==0] # 311681 rows with 'grantraw'==0 No missing values!

# Check for missing values by column 'hearing_loc_code'
data_no_duplicates['hearing_loc_code'].isnull().values.any() # No missing values!
data_no_duplicates['hearing_loc_code'].unique()
data_no_duplicates['hearing_loc_code'].unique().shape # 247 unique locations

# Check for missing values by column 'natid'
data_no_duplicates['natid'].isnull().values.any() # There are missing values!
data_no_duplicates['natid'].unique()
data_no_duplicates['natid'].unique().shape # 227 unique natid including 'NaN'
data_no_duplicates['natid'].isnull().sum() # 337 rows with missing values on column 'natid'

# Prepare data for basic Log Reg using features 'hearing_loc_code', 'natid', and 'grantraw' as X
data_X = data_no_duplicates[['hearing_loc_code', 'natid', 'grantraw']]

# Remove 337 rows where 'natid' has missing values
data_X = data_X.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# Codify data on hearing_loc_code as numbers
@jit
def map_to_dict():
    dicti = {}
    ad = -1
    for i in data_no_duplicates['hearing_loc_code'].unique():
        ad += 1
        dicti[i] = ad
    return dicti
mapp = map_to_dict() # dictionary where key = hearing_loc_code and value = number

@jit
def codify_in_place(mapp, data_X):
    data_X.hearing_loc_code.replace(to_replace=mapp, inplace=True)
    return data_X
codify_in_place(mapp, data_X)
data_X
# Random split into train and test
import numpy as np
split = np.random.rand(len(data_X)) < .8
train = data_X[split]
test = data_X[~split]

train.shape # 383673 rows
test.shape # 95583 rows

# Implement Log Reg
from sklearn import linear_model

logreg = linear_model.LogisticRegression(C = 1e30)
logreg.fit(train.drop('grantraw', 1), train['grantraw'])

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
%matplotlib inline

@jit
def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c, label= lab+' (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")
    
plotAUC(test['grantraw'], logreg.predict_proba(test.drop('grantraw', 1))[:,1], 'LR')  

plt.show()
