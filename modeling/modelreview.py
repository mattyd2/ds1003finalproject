from sklearn.ensemble import AdaBoostClassifier
import sys
import pickle
from atty_interactions import *
from learning_curve import *
from general_modeling_function import *

test = pd.read_csv("../../data/final_data/test.csv")
train = pd.read_csv("../../data/final_data/train.csv")
x_cols = np.setdiff1d(test.columns, ['appl_dec_G', 'appl_dec_D', 'appl_dec_F', 'appl_dec_L'])

X_test = test[x_cols]
y_test = test['appl_dec_G']

X_train = train[x_cols]
y_train = train['appl_dec_G']

pkl_file = open('Ada_atty_interactions.pkl', 'rb')
model = pickle.load(pkl_file)

# print 'model.best_estimator_\n', model.best_estimator_
# print 'model.best_score_\n', model.best_score_
# print 'model.best_params_\n', model.best_params_
# print 'model.scorer_\n', model.scorer_
# print 'model.grid_scores_\n', 
# for i in model.grid_scores_:
#     print i

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

mf.make_learning_curve_from_gridsearchcsv(model, hyperparm, model_type)