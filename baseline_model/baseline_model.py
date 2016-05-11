import pandas as pd
import numpy as np
from numba import jit
from sklearn import linear_model
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# attorney_flag_0 # attorney_flag_1
# c_asy_type_E # c_asy_type_I # c_asy_type_E_or_I # (E = defensive I = affirmative)
# appl_dec_G # target

train = pd.DataFrame()
test = pd.DataFrame()


def plot_roc():
    train = read_train()
    test = read_test()
    logreg = linear_model.LogisticRegression(C = 1e30)
    logreg.fit(train[['attorney_flag_0','attorney_flag_1','c_asy_type_E','c_asy_type_I','c_asy_type_E_or_I']], train['appl_dec_G'])
    plotAUC(test['appl_dec_G'], logreg.predict_proba(test[['attorney_flag_0','attorney_flag_1','c_asy_type_E','c_asy_type_I','c_asy_type_E_or_I']])[:,1], 'LR')  
    plt.show()

   
def read_train():
    train = pd.read_csv('./train.csv')
    return train

   
def read_test():
    test = pd.read_csv('./test.csv')
    return test
   
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
    
if __name__ == '__main__':
    plot_roc()