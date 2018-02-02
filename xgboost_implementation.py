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
xb_train, xb_test, yb_train, yb_test = train_test_split(bX_train, by_train, test_size=test_size)
xc_train, xc_test, yc_train, yc_test = train_test_split(cX_train, cy_train, test_size=test_size)

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
    learning_rate = 0.05,
    n_estimators = 1000,
    max_depth = 7,
    min_child_weight = 3,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.8,
    reg_alpha = 0.002,
    objective = 'binary:logistic',
    nthread = 4,
    scale_pos_weight = 1,
    seed = 27
)

a_model = modelfit(xgb1, x_train, y_train, x_test, y_test)
b_model = modelfit(xgb1, xb_train, yb_train, xb_test, yb_test)
c_model = modelfit(xgb1, xc_train, yc_train, xc_test, yc_test)

# # lets make see how we generalize with real data (test data)
ax_test = per_process_data(a_test, enforce_cols=aX_train.columns)
bx_test = per_process_data(b_test, enforce_cols=bX_train.columns)
cx_test = per_process_data(c_test, enforce_cols=cX_train.columns)

# make predictions
a_preds = a_model.predict_proba(ax_test)
b_preds = b_model.predict_proba(bx_test)
c_preds = c_model.predict_proba(cx_test)

# save submission
def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1], # proba p=1
            columns=['poor'],
            index=test_feat.index)
    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

# convert preds to data frames
a_sub = make_country_sub(a_preds, ax_test, 'A')
b_sub = make_country_sub(b_preds, bx_test, 'B')
c_sub = make_country_sub(c_preds, cx_test, 'C')

# combine our predictions and save the submission file
submission = pd.concat([a_sub, b_sub, c_sub])