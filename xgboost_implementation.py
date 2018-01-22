#%%
# import dependencies(libraries)
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

# visualization (EDA)
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

# load data
a_train = pd.read_csv('training_data/A_hhold_train.csv', index_col='id')
b_train = pd.read_csv('training_data/B_hhold_train.csv', index_col='id')
c_train = pd.read_csv('training_data/C_hhold_train.csv', index_col='id')

a_test = pd.read_csv('test_data/A_hhold_test.csv', index_col='id')
b_test = pd.read_csv('test_data/B_hhold_test.csv', index_col='id')
c_test = pd.read_csv('test_data/C_hhold_test.csv', index_col='id')

a_train.head()

# function to create XGBoost models and perform cross-validation.
def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round)