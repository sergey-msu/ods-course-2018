import os
import numpy as np
import pandas as pd
import pickle
import utils
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from alice import data_utils

def header(): return 'Week 5: Vowpal Wabbit http://localhost:8888/notebooks/Alice/week6_vowpal_wabbit.ipynb'

def run():

  train_df, test_df, X_train_sparse, X_test_sparse, y, y_for_vw, encoder = prepare_data()

  sites = [ 'site{}'.format(i) for i in range(1, 11) ]
  train_share = int(0.7 * train_df.shape[0])
  train_df_part = train_df[sites].iloc[:train_share, :].values.astype(int)
  valid_df      = train_df[sites].iloc[train_share:, :].values.astype(int)
  test_df       = test_df[sites].values.astype(int)
  train_df      = train_df[sites].values.astype(int)
  X_train_part_sparse = X_train_sparse[:train_share, :]
  X_valid_sparse      = X_train_sparse[train_share:, :]

  y_train_part = y[:train_share]
  y_valid      = y[train_share:]
  y_train_part_for_vw = y_for_vw[:train_share]
  y_valid_for_vw      = y_for_vw[train_share:]

  train_vw = arrays_to_vw(train_df, y_for_vw, train=True,
                          out_file=utils.PATH.COURSE_FILE('train.vw', 'kaggle_alice400'))
  train_part_vw = arrays_to_vw(train_df_part, y_train_part_for_vw, train=True,
                               out_file=utils.PATH.COURSE_FILE('train_part.vw', 'kaggle_alice400'))
  valid_vw = arrays_to_vw(valid_df, y_valid_for_vw, train=True,
                          out_file=utils.PATH.COURSE_FILE('valid.vw', 'kaggle_alice400'))
  test_vw = arrays_to_vw(test_df, None, train=False,
                         out_file=utils.PATH.COURSE_FILE('test.vw', 'kaggle_alice400'))

  # VALID

  #print('C:\Program Files\VowpalWabbit>vw --oaa 400 --passes 3 -c --bit_precision 26 --random_seed 17 --loss_function hinge -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\train_part.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\train_part_model.vw')
  #print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\train_part_model.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\vw_valid_pred.csv --quiet')
  #
  #with open(utils.PATH.COURSE_FILE('vw_valid_pred.csv', 'kaggle_alice400')) as pred_file:
  #  valid_prediction = [int(label) for label in pred_file.readlines()]
  #  print("VOWPAL validation accuracy: {0}".format(round(accuracy_score(y_valid_for_vw, valid_prediction), 6)))
  #
  #sgd = SGDClassifier(loss='log', n_iter=3, random_state=17, n_jobs=-1)
  #sgd.fit(X_train_part_sparse, y_train_part)
  #valid_prediction = sgd.predict(X_valid_sparse)
  #print("SGD validation accuracy: {0}".format(round(accuracy_score(y_valid, valid_prediction), 6)))
  #
  #lr = LogisticRegression(random_state=17, n_jobs=-1)
  #lr.fit(X_train_part_sparse, y_train_part)
  #valid_prediction = lr.predict(X_valid_sparse)
  #print("LOGIT validation accuracy: {0}".format(round(accuracy_score(y_valid, valid_prediction), 6)))

  # TEST

  print('C:\Program Files\VowpalWabbit>vw --oaa 400 --passes 3 -c --bit_precision 26 --random_seed 17 --loss_function hinge -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\train_model.vw')
  print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\train_model.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\kaggle_alice400\vw_pred.csv --quiet')

  with open(utils.PATH.COURSE_FILE('vw_pred.csv', 'kaggle_alice400')) as pred_file:
    test_prediction = np.array([int(label) for label in pred_file.readlines()]) - 1
    test_prediction = encoder.inverse_transform(test_prediction)
    write_to_submission_file(test_prediction, utils.PATH.COURSE_FILE('vw_400_users.csv', 'kaggle_alice400'))

  sgd = SGDClassifier(loss='log', n_iter=3, random_state=17, n_jobs=-1)
  sgd.fit(X_train_sparse, y)
  test_prediction = sgd.predict(X_test_sparse)
  write_to_submission_file(test_prediction, utils.PATH.COURSE_FILE('sgd_400_users.csv', 'kaggle_alice400'))

  lr = LogisticRegression(random_state=17, n_jobs=-1)
  lr.fit(X_train_sparse, y)
  test_prediction = lr.predict(X_test_sparse)
  write_to_submission_file(test_prediction, utils.PATH.COURSE_FILE('logit_400_users.csv', 'kaggle_alice400'))

  return

def arrays_to_vw(X, y=None, train=False, out_file=None):
  with open(out_file, 'w') as file:
    for i, row in enumerate(X):
      label = y[i] if train else 1
      file.write(str(label) + ' | ' + ' '.join(row.astype(str)) + '\n')

def write_to_submission_file(predicted_labels, out_file,
                             target='user_id', index_label="session_id"):
  # turn predictions into data frame and save as csv file
  predicted_df = pd.DataFrame(predicted_labels,
                              index = np.arange(1, predicted_labels.shape[0] + 1),
                              columns=[target])
  predicted_df.to_csv(out_file, index_label=index_label)

def prepare_data():
  # prepare data

  train_df = pd.read_csv(utils.PATH.COURSE_FILE('train_sessions_400users.csv', 'kaggle_alice400'), index_col='session_id')
  test_df  = pd.read_csv(utils.PATH.COURSE_FILE('test_sessions_400users.csv',  'kaggle_alice400'), index_col='session_id')

  print(train_df['user_id'].value_counts())
  print(train_df.shape)
  print(test_df.shape)

  y = train_df['user_id']
  class_encoder = LabelEncoder()
  y_for_vw = class_encoder.fit_transform(y) + 1

  full_df = pd.concat([train_df.drop('user_id', axis=1), test_df])
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

  return train_df, test_df, X_train_sparse, X_test_sparse, y, y_for_vw, class_encoder

