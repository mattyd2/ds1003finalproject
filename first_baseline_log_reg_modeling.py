# This code is exploring and cleaning file asylum_clean.csv
# This code is implementing a basic Log Reg as an optional baseline model for using simple input such as the features ’lawyer’ and ‘defensive’. The outcome is ‘grantraw’ 

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

# Check for missing values by column 'grant'
data['grant'].isnull().values.any() # there are missing values!

# Check for missing values by column 'grantraw'
data['grantraw'].isnull().values.any() # no NaN value
data[data['grantraw']==1] # 167912 rows with 'grantraw'==1
data[data['grantraw']==0] # 311681 rows with 'grantraw'==0 No missing values!

# Check for missing values by column 'lawyer'
data['lawyer'].isnull().values.any() # no NaN value
data[data['lawyer']==1] # 423609 rows with 'lawyer'==1
data[data['lawyer']==0] # 55984 rows with 'lawyer'==0 No missing values!

# Check for missing values by column 'defensive'
data['defensive'].isnull().values.any() # there are missing values!
data['defensive'].isnull().sum() # 12304 rows with missing values on column 'defensive'

# Prepare data for basic Log Reg using features 'lawyer', 'defensive', and 'grantraw' as X
data_X = data[['lawyer', 'defensive', 'grantraw']]

# Remove 12304 rows where 'defensive' has missing values
data_X = data_X.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# Random split into train and test
import numpy as np
import random

np.random.seed(seed=0)
new_index = range(data_X.shape[0])
np.random.shuffle(new_index)

train = data_X.iloc[new_index[0:int(data_X.shape[0]*.8)]]
test = data_X.iloc[new_index[int(data_X.shape[0]*.8)]:]

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
