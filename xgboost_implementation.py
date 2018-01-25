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

# preprocess the data

# standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df

def per_process_data(df, enforce_cols=None):
    print("Input shape: \t {}".format(df.shape))

    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals: \t {}".format(df.shape))

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    df.fillna(0, inplace=True)

    return df

# convert the data.
aX_train = per_process_data(a_train.drop('poor', axis=1))
ay_train = np.ravel(a_train.poor)

bX_train = per_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

cX_train = per_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)

# split the data for both training and cross validation to evaluate our model
from sklearn.model_selection import train_test_split

test_size = 0.32

x_train, x_test, y_train, y_test = train_test_split(aX_train, ay_train, test_size=test_size)

# function to create XGBoost models and perform cross-validation.
def modelfit(alg, xtrain, ytrain, dtest, ytest, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(xtrain, label=ytrain)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # fit the algorithm on the data
    alg.fit(xtrain, ytrain, eval_metric='auc')

    # predict training set:
    dtrain_predictions = alg.predict(xtrain)
    dtrain_predprob = alg.predict_proba(xtrain)[:, 1]

    # print model report
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(ytrain, dtrain_predictions))
    print('AUC Score (Train) : %f' % metrics.roc_auc_score(ytrain, dtrain_predprob))

    # predict on testing data
    results = alg.predict_proba(dtest)[:, 1]
    print("AUC Score (Test): %f" % metrics.roc_auc_score(ytest, results))

xgb1 = XGBClassifier(
    learning_rate = 0.1,
    n_estimators = 1000,
    max_depth = 5,
    min_child_weight = 1,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = 'binary:logistic',
    nthread = 4,
    scale_pos_weight = 1,
    seed = 27
)
modelfit(xgb1, x_train, y_train, x_test, y_test)

# let's fine tune our model
# - we are tuning max_depth and min_child_weight

param_test1 = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight':[1, 3, 5]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, n_estimators=140, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
param_grid = param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch1.fit(x_train, y_train)
gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_


# # lets make see how we generalize with real data (test data)
# ax_test = per_process_data(a_test, enforce_cols=aX_train.columns)
# bx_test = per_process_data(b_test, enforce_cols=bX_train.columns)
# cx_test = per_process_data(c_test, enforce_cols=cX_train.columns)

# a_preds = model_a.predict_prob(ax_test)

# # save submission
# def make_country_sub(preds,)