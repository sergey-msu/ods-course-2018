import os
import glob
import math
import datetime
import numpy as np
import pandas as pd
import pickle
import utils
from collections import Counter
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

def header(): return 'KAGGLE ALICE: https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2'

def run():

  logit_baseline()

  return

def write_to_submission_file(pred_labels, out_file, target='target', index_label='session_id'):
  pred_df = pd.DataFrame(pred_labels,
                         index=np.arange(1, pred_labels.shape[0] + 1),
                         columns=[target])
  submission_fname = utils.PATH.STORE_FOR('kaggle_alice\submissions', out_file)
  pred_df.to_csv(submission_fname, index_label=index_label)
  return

def get_auc_lr_valid(X, y, seed=17, ratio=0.9):
  idx = int(round(X.shape[0]*ratio))

  params_lr = { 'C': np.logspace(-1, 1, 5) }
  grid_lr = GridSearchCV(LogisticRegression(random_state=seed, n_jobs=-1),
                         params_lr,
                         cv=4,
                         n_jobs=-1,
                         scoring='roc_auc')
  grid_lr.fit(X[:idx, :], y[:idx])
  print(grid_lr.best_params_, grid_lr.best_score_)

  #grid_lr = LogisticRegressionCV(
  #      Cs=list(np.logspace(-2, 2, 5))
  #      ,penalty='l2'
  #      ,scoring='roc_auc'
  #      ,cv=7
  #      ,random_state=seed
  #      ,fit_intercept=True
  #      ,tol=10
  #  )
  #grid_lr.fit(X[:idx, :], y[:idx])
  #print ('Scores:', grid_lr.scores_)
  #print ('C_:', grid_lr.C_)
  #print ('Max auc_roc:', grid_lr.scores_[1].max())

  #X_test = X[idx:, :]
  #y_pred = grid_lr.predict_proba(X_test)[:, 1]
  #
  #score = roc_auc_score(y[idx:], y_pred)
  score = -1

  return score, grid_lr

def get_sites():
  with open(utils.PATH.COURSE_FILE('site_dic.pkl', 'kaggle_alice'), 'rb') as f:
    site_dict = pickle.load(f)
  site_dict_df = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])

  return site_dict

###################################################################
########################## LOGIT BASELINE #########################
###################################################################

def logit_baseline():

  X_train, X_test, y_train = logit_prepare_data()

  # training the first model - 90% of train data, test with the rest 10%
  score, predictor = get_auc_lr_valid(X_train, y_train)
  print(score)

  # training the second model - 100% of train data, test on test data
  lr = LogisticRegression(C=1.0, random_state=17, n_jobs=-1)
  lr.fit(X_train, y_train)

  y_test = lr.predict_proba(X_test)[:, 1]

  write_to_submission_file(y_test, 'baseline_1.csv')

  return

def logit_prepare_data():
  # prepare data

  train_df = pd.read_csv(utils.PATH.COURSE_FILE('train_sessions.csv', 'kaggle_alice'), index_col='session_id')
  test_df  = pd.read_csv(utils.PATH.COURSE_FILE('test_sessions.csv',  'kaggle_alice'), index_col='session_id')

  times = [ 'time{}'.format(i) for i in range(1, 11) ]
  train_df[times] = train_df[times].apply(pd.to_datetime)
  test_df[times]  = test_df[times].apply(pd.to_datetime)

  train_df = train_df.sort_values(by='time1')
  print(train_df.shape)
  print(test_df.shape)
  #print(train_df.head())

  sites = [ 'site{}'.format(i) for i in range(1, 11) ]
  train_df[sites] = train_df[sites].fillna(0).astype('int')
  test_df[sites]  = test_df[sites].fillna(0).astype('int')

  y_train = train_df['target']

  full_df = pd.concat([train_df.drop('target', axis=1), test_df])
  train_idxs = train_df.shape[0]

  # take only sites, not dates for very simple model

  full_sites = full_df[sites]
  print(full_sites.head())

  # data -> to site frequencies

  sites_flatten = full_sites.values.flatten()
  full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                  sites_flatten,
                                  range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]

  X_train = full_sites_sparse[:train_idxs, :]
  X_test  = full_sites_sparse[train_idxs:, :]

  return X_train, X_test, y_train
