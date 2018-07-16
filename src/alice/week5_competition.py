import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from alice import data_utils

def header(): return 'Week 5: Kaggle Competition http://localhost:8888/notebooks/Alice/week5_sgd_kaggle.ipynb'

def run():

  print('The same as in l6_homework !!!')

  X_train_sparse, X_test_sparse, y = prepare_data()
  print(X_train_sparse.shape)
  print(X_test_sparse.shape)

  train_share = int(0.7 * X_train_sparse.shape[0])
  X_train, y_train = X_train_sparse[:train_share], y[:train_share]
  X_valid, y_valid = X_train_sparse[train_share:], y[train_share:]

  sgd = SGDClassifier(loss='log', random_state=17, n_jobs=-1)
  sgd.fit(X_train, y_train)

  y_pred = sgd.predict_proba(X_valid)[:, 1]
  auc = roc_auc_score(y_valid, y_pred)
  print(auc)

  sgd.fit(X_train_sparse, y)
  y_pred = sgd.predict_proba(X_test_sparse)[:, 1]

  write_to_submission_file(y_pred, 'baseline_0.csv')

  return

def prepare_data():
  # prepare data

  train_df = pd.read_csv(utils.PATH.COURSE_FILE('train_sessions.csv', 'kaggle_alice'), index_col='session_id')
  test_df  = pd.read_csv(utils.PATH.COURSE_FILE('test_sessions.csv',  'kaggle_alice'), index_col='session_id')

  print(train_df['target'].value_counts())
  print(train_df.shape)
  print(test_df.shape)

  y_train = train_df['target']

  full_df = pd.concat([train_df.drop('target', axis=1), test_df])
  train_idxs = train_df.shape[0]

  # take only sites, not dates for very simple model

  sites = [ 'site{}'.format(i) for i in range(1, 11) ]
  full_sites = full_df[sites].fillna(0).astype('int')
  print(full_sites.head(10))

  # data -> to site frequencies

  sites_flatten = full_sites.values.flatten()
  full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                  sites_flatten,
                                  range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]

  X_train_sparse = full_sites_sparse[:train_idxs, :]
  X_test_sparse  = full_sites_sparse[train_idxs:, :]

  return X_train_sparse, X_test_sparse, y_train

def write_to_submission_file(pred_labels, out_file, target='target', index_label='session_id'):
  pred_df = pd.DataFrame(pred_labels,
                         index=np.arange(1, pred_labels.shape[0] + 1),
                         columns=[target])
  submission_fname = utils.PATH.STORE_FOR('kaggle_alice\submissions', out_file)
  pred_df.to_csv(submission_fname, index_label=index_label)
  return
