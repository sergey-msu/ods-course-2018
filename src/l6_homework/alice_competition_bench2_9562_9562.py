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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier

def header(): return 'KAGGLE ALICE: https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2'

def run():

  #baseline_9562()

  ensure_extra_data()

  features = [
               #'hour7',
               ##'hour8',
               ##'hour9',
               ##'hour10',
               ##'hour11',
               #'hour12',
               #'hour13',
               ##'hour14',
               ##'hour15',
               #'hour16',
               #'hour17',
               #'hour18',
               #'hour19',
               #'hour20',
               #'hour21',
               #'hour22',
               #'hour23',

               'day0',
               'day1',
               #'day2',
               'day3',
               'day4',
               'day5',
               #'day6',

               'week1',
               'week2',
               'week3',
               'week4',

               #'month1',
               #'month2',
               #'month3',
               #'month4',
               #'month5',
               #'month6',
               #'month7',
               #'month8',
               #'month9',
               #'month10',
               #'month11',
               #'month12',

               #'year2013',
               #'year2014',

               #'duration',
               #'is_work',
               #'is_work_time',
               #'is_free',
               #'hour_c',
               #'hour_s',

               #'morning_time',
               #'day_time',
               #'evening_time',
               #'night_time',

               #'top0',
               #'top1',
               #'top2',
               #'top3',
               #'top4',
               #'top5',
               #'top6',

               #'dtop0',
               #'dtop1',
               #'dtop2',
               #'dtop3',
               #'dtop4',
               #'dtop5',
               #'dtop6',

               #'top_cnt',
             ]

  __fs = ['hour10', 'hour11', 'hour8', 'hour14', 'hour9', 'hour15', 'day2', 'day6' ]
  score = assignment6_baseline(__fs)
  print(score)

  #scores = []
  #for j in range(1, len(features)):
  #  for i in range(j):
  #    feats = __fs + [features[i], features[j]]
  #    score = assignment6_baseline(feats)
  #    scores.append(feats+[score])
  #    print(scores)

  #scores = []
  #for j in range(1, len(features)):
  #  feats = __fs + [features[j]]
  #  score = assignment6_baseline(feats)
  #  scores.append(feats+[score])
  #  print(scores)

  return

SESSION_LENGTH = 10
WINDOW_SIZE = 10

###################################################################
###################### ASSIGNMENT 6 BASELINE ######################
###################################################################

def baseline_9562():
  lr_pred  = pd.read_csv(utils.PATH.STORE_FOR('kaggle_alice\submissions', 'assignment6_alice_submission_9569_9536.csv'))['target'].values
  gbm_pred = pd.read_csv(utils.PATH.STORE_FOR('kaggle_alice\submissions', 'assignment6_alice_submission_gbm_1.csv'))['target'].values

  alpha = 0.6
  pred = np.clip(alpha*lr_pred + (1 - alpha)*gbm_pred, 0, 1)
  write_to_submission_file(pred, 'lin_comb.csv')
  return


def assignment6_baseline(features):

  train_data, test_data = ensure_data(features)
  X_train = train_data[:, :-1]
  y_train = train_data[:, -1].toarray().reshape([train_data.shape[0]])
  X_test  = test_data

  print('------------- training -------------')
  #predictor1, score = get_auc_lr_valid(X_train, y_train)
  predictor2, score = f(X_train, y_train)
  #predictor3, score = gbm(X_train, y_train)
  #predictor, score = get_auc_compose_valid(predictor1, predictor3, X_train, y_train)

  print('------------- saving results -------------')
  #y_test1 = predictor1.predict_proba(X_test)[:, 1]
  y_test2 = predictor2.predict(X_test)
  #y_test3 = predictor3.predict_proba(X_test)[:, 1]
  write_to_submission_file(y_test2, 'assignment6_alice_submission_svm.csv')
  #write_to_submission_file((y_test1+y_test3)/2, 'assignment6_alice_submission_sgd2.csv')
  #write_to_submission_file(np.maximum(y_test1, y_test2), 'assignment6_alice_submission_3.csv')
  #write_to_submission_file(np.minimum(y_test1, y_test2), 'assignment6_alice_submission_4.csv')

  return score

def ensure_data(features):
  train_tfidf_fname = utils.PATH.STORE_FOR('kaggle_alice', 'train_tfidf.pkl')
  train_extra_fname = utils.PATH.STORE_FOR('kaggle_alice', 'train_extra.csv')
  test_tfidf_fname  = utils.PATH.STORE_FOR('kaggle_alice', 'test_tfidf.pkl')
  test_extra_fname  = utils.PATH.STORE_FOR('kaggle_alice', 'test_extra.csv')

  # check if data files exist
  if os.path.exists(train_tfidf_fname) and \
     os.path.exists(train_extra_fname) and \
     os.path.exists(test_tfidf_fname) and \
     os.path.exists(test_extra_fname):
    try:
      with open(train_tfidf_fname, 'rb') as f:
        train_tfidf_data = pickle.load(f)
      with open(test_tfidf_fname, 'rb') as f:
        test_tfidf_data = pickle.load(f)
      train_extra_data = pd.read_csv(train_extra_fname)
      test_extra_data  = pd.read_csv(test_extra_fname)
    except:
      os.remove(train_tfidf_fname)
      os.remove(train_extra_fname)
      os.remove(test_tfidf_fname)
      os.remove(test_extra_fname)
  else:
    # if not - prepare data and save data files
    train_tfidf_data, train_extra_data, test_tfidf_data, test_extra_data = prepare_data()

    with open(train_tfidf_fname, 'wb') as f:
      pickle.dump(train_tfidf_data, f, protocol=2)
    with open(test_tfidf_fname, 'wb') as f:
      pickle.dump(test_tfidf_data, f, protocol=2)
    train_extra_data.to_csv(train_extra_fname)
    test_extra_data.to_csv(test_extra_fname)

  # filter only features under consideration
  train_data, test_data = filter_data(features, train_tfidf_data, train_extra_data, test_tfidf_data, test_extra_data)

  return train_data, test_data

def prepare_data():
  # read data from files
  print('------------- read data -------------')

  train_df = pd.read_csv(utils.PATH.COURSE_FILE('train_sessions.csv', 'kaggle_alice'), index_col='session_id')
  print('train data:', len(train_df))
  extra_df = pd.read_csv(utils.PATH.COURSE_FILE('extra_sessions_s{0}_w{1}.csv'.format(SESSION_LENGTH, WINDOW_SIZE), 'kaggle_alice'), index_col='session_id')
  extra_alice_df = extra_df[extra_df['target']==1]
  train_df = pd.concat([train_df, extra_alice_df])
  train_df = train_df.sort_values(by='time1')
  print('alice extra data:', len(extra_alice_df))
  print('total data:', len(train_df))

  test_df = pd.read_csv(utils.PATH.COURSE_FILE('test_sessions.csv', 'kaggle_alice'), index_col='session_id')
  print('test data:', len(test_df))

  y_train = train_df['target']
  train_len = len(y_train)

  full_df = pd.concat([train_df.drop('target', axis=1), test_df])

  # site freq features
  print('------------- prepare site freq features -------------')

  sites = [ 'site{}'.format(i) for i in range(1, 11) ]
  times = [ 'time{}'.format(i) for i in range(1, 11) ]
  train_df[sites] = train_df[sites].fillna(0).astype('int')
  test_df[sites]  = test_df[sites].fillna(0).astype('int')
  full_df[times]  = full_df[times].apply(pd.to_datetime)

  top_cnt = 10
  all_freqs   = get_site_freqs('all_freqs', train_df[sites], top_cnt=top_cnt)
  alice_freqs = get_site_freqs('alice_freqs', train_df[train_df['target']==1][sites], top_cnt=top_cnt)
  other_freqs = get_site_freqs('other_freqs', train_df[train_df['target']==0][sites], top_cnt=top_cnt)
  print('top freqs:', all_freqs)
  print('alice freqs:', alice_freqs)
  print('other freqs:', other_freqs)
  all_top_sites   = set(all_freqs.keys())
  alice_top_sites = set(alice_freqs.keys()).difference(all_top_sites)
  print('alice unique freqs:', alice_top_sites)

  tspans = []
  for i in range(1, SESSION_LENGTH):
    col_name='tspan{0}'.format(i)
    tspans.append(col_name)
    full_df[col_name] = full_df['time{0}'.format(i+1)] - full_df['time{0}'.format(i)]
    full_df[col_name] = full_df[col_name].apply(lambda x: x.total_seconds())
  full_df[tspans] = full_df[tspans].fillna(0)

  def apply_top_duration(row, id):
    sids = row[:SESSION_LENGTH]
    ds   = row[SESSION_LENGTH:]
    dur  = 0
    for i in range(SESSION_LENGTH-1):
      sid = sids[i]
      if sid==id:
        dur += ds[i]
    return dur

  tops = []
  top_spans = []
  for i, id in enumerate(alice_top_sites):
    col_name = 'top{0}'.format(i)
    span_col_name = 'dtop{0}'.format(i)
    tops.append(col_name)
    top_spans.append(span_col_name)
    full_df[col_name] = full_df[sites].apply(lambda r: 1 if id in r.values else 0, axis=1)
    full_df[span_col_name] = full_df[sites+tspans].apply(lambda r: apply_top_duration(r.values, id), axis=1)
  full_df['top_cnt'] = full_df[tops].apply(lambda r: sum(r.values), axis=1)
  full_df = full_df.drop(tspans, axis=1)

  # time features
  print('------------- prepare time features -------------')

  session_dur = full_df[times].max(axis=1) - full_df[times].min(axis=1)
  full_df['duration'] = session_dur.apply(lambda x: x.seconds)

  start = full_df['time1']

  weekday = start.apply(lambda x: x.weekday())
  full_df = pd.concat([full_df, pd.get_dummies(weekday, prefix='day', prefix_sep='')], axis=1)

  full_df['week1'] = start.apply(lambda x: 1 if x.day<=7 else 0)
  full_df['week2'] = start.apply(lambda x: 1 if x.day>7  and x.day<=14 else 0)
  full_df['week3'] = start.apply(lambda x: 1 if x.day>14 and x.day<=21 else 0)
  full_df['week4'] = start.apply(lambda x: 1 if x.day>21 else 0)

  month = start.apply(lambda x: x.month)
  full_df = pd.concat([full_df, pd.get_dummies(month, prefix='month', prefix_sep='')], axis=1)

  year = start.apply(lambda x: x.year)
  full_df = pd.concat([full_df, pd.get_dummies(year, prefix='year', prefix_sep='')], axis=1)

  full_df['is_work_time'] = start.apply(lambda x: 1 if x.weekday()<=5 and (x.hour>=9 and x.hour<=18 and x.hour != 13) else 0)
  full_df['is_work'] = start.apply(lambda x: 1 if x.weekday()<=5 else 0)
  full_df['is_free'] = start.apply(lambda x: 1 if x.weekday()>5  else 0)

  hour = start.apply(lambda x: x.hour)
  full_df = pd.concat([full_df, pd.get_dummies(hour, prefix='hour', prefix_sep='')], axis=1)

  full_df['hour_c']       = hour.apply(lambda x: np.cos(x*np.pi/12))
  full_df['hour_s']       = hour.apply(lambda x: np.sin(x*np.pi/12))
  full_df['morning_time'] = hour.apply(lambda x: 1 if (x>=5)  and (x<10) else 0)
  full_df['day_time']     = hour.apply(lambda x: 1 if (x>=10) and (x<18) else 0)
  full_df['evening_time'] = hour.apply(lambda x: 1 if (x>=18) and (x<22) else 0)
  full_df['night_time']   = hour.apply(lambda x: 1 if (x>=22) or  (x<5)  else 0)

  print('------------- TFIDF vectorization -------------')
  site_paths = full_df[sites] \
                 .astype(str) \
                 .apply(lambda x: ' '.join(x), axis=1) \
                 .apply(lambda x: str.rstrip(x, '0 '))
  vect = TfidfVectorizer(ngram_range=(1, 3), max_features=100000)
  full_tfidf_data  = vect.fit_transform(site_paths.values)
  print('TFIDF vocabulary:', len(vect.vocabulary_))

  train_tfidf_data = full_tfidf_data[:train_len, :]
  test_tfidf_data  = full_tfidf_data[train_len:, :]

  print('------------- Extra features -------------')

  full_df = full_df.drop(sites+times, axis=1)
  scalables = ['duration', 'top_cnt'] + top_spans
  full_df[scalables] = MinMaxScaler().fit_transform(full_df[scalables])
  train_extra_data = pd.concat([full_df.iloc[:train_len, :], y_train], axis=1)
  test_extra_data  = full_df.iloc[train_len:, :]

  return train_tfidf_data, train_extra_data, test_tfidf_data, test_extra_data

def filter_data(features, train_tfidf_data, train_extra_data, test_tfidf_data, test_extra_data):
  print('------------- stack final matrices -------------')

  y_train = train_extra_data['target'].values.reshape([-1, 1])
  train_data = hstack((train_tfidf_data, csr_matrix(train_extra_data[features]), y_train)).tocsr()
  test_data  = hstack((test_tfidf_data,  csr_matrix(test_extra_data[features]))).tocsr()

  print('final train shape:', train_data.shape)
  print('final test shape:',  test_data.shape)

  return train_data, test_data



def ensure_extra_data():
  extra_fname = utils.PATH.COURSE_FILE('extra_sessions_s{0}_w{1}.csv'.format(SESSION_LENGTH, WINDOW_SIZE), 'kaggle_alice')
  if os.path.exists(extra_fname):
    return

  print('------------- load extra data -------------')
  extra_data = prepare_extra_data(SESSION_LENGTH, WINDOW_SIZE)
  extra_data.to_csv(extra_fname)
  return

def prepare_extra_data():
  site_dict = get_site_dict()

  cols = ['']*(2*SESSION_LENGTH)
  cols[::2]  = [ 'site{0}'.format(j) for j in range(1, SESSION_LENGTH+1) ]
  cols[1::2] = [ 'time{0}'.format(j) for j in range(1, SESSION_LENGTH+1) ]
  cols.append('target')

  extra_df = pd.DataFrame(columns=cols)

  # alice extra data
  extra_alice_df = pd.read_csv(utils.PATH.COURSE_FILE('Alice_log.csv', 'kaggle_alice/train'))
  data = process_extra_data(extra_alice_df, 1.0, SESSION_LENGTH, WINDOW_SIZE, site_dict)
  extra_df = pd.concat([extra_df, pd.DataFrame(data=data, columns=cols)])
  print('alice extra len:', len(extra_df))

  # other extra data
  extra_other_dpath = utils.PATH.COURSE_PATH('kaggle_alice/train/other_user_logs')
  pattern = extra_other_dpath+'/user*.csv'
  processed = 0
  batch_size = 100
  batch_df = pd.DataFrame(columns=cols)
  for f in glob.glob(pattern):
    extra_other_df = pd.read_csv(f)
    data = process_extra_data(extra_other_df, 0.0, SESSION_LENGTH, WINDOW_SIZE, site_dict)
    batch_df = pd.concat([batch_df, pd.DataFrame(data=data, columns=cols)])
    processed += 1
    if processed%batch_size == 0:
      extra_df = pd.concat([extra_df, batch_df])
      batch_df = pd.DataFrame(columns=cols)
      print('processed:', processed)

  print('processed:', processed)
  print('total extra len:', len(extra_df))

  extra_df.index.names= ['session_id']

  return extra_df

def process_extra_data(extra_df, target, site_dict):
  site_ids = extra_df['site'].apply(lambda s: site_dict[s])
  timestamps = extra_df['timestamp']
  n = len(site_ids)
  r_num = math.ceil(n/WINDOW_SIZE)
  data = []

  for i in range(r_num):
    i_begin = i*WINDOW_SIZE
    i_end   = min(i_begin + SESSION_LENGTH, n)
    row = [None]*(SESSION_LENGTH*2 + 1)
    row[0: 2*(i_end-i_begin): 2] = site_ids.iloc[i_begin:i_end]
    row[1: 2*(i_end-i_begin): 2] = timestamps.iloc[i_begin:i_end]
    row[-1] = target
    data.append(row)

  return data



def write_to_submission_file(pred_labels, out_file, target='target', index_label='session_id'):
  pred_df = pd.DataFrame(pred_labels,
                         index=np.arange(1, pred_labels.shape[0] + 1),
                         columns=[target])
  submission_fname = utils.PATH.STORE_FOR('kaggle_alice\submissions', out_file)
  pred_df.to_csv(submission_fname, index_label=index_label)
  return

class ComposeAlg(BaseEstimator):
  def __init__(self, a1, a2, alpha = 0.5):
    self._a1 = a1
    self._a2 = a2
    self.alpha = alpha
    return

  def fit(self, X, y, sample_weight=None):
    return

  def predict_proba(self, X):
    p1 = self._a1.predict_proba(X)
    p2 = self._a2.predict_proba(X)
    return self.alpha*p1 + (1 - self.alpha)*p2

def get_auc_compose_valid(a1, a2, X, y, seed=17):
  params_lr = { 'alpha': np.linspace(0, 1, 10) }
  grid_lr = GridSearchCV(ComposeAlg(a1, a2),
                         params_lr,
                         cv=4,
                         n_jobs=-1,
                         scoring='roc_auc')
  grid_lr.fit(X, y)
  print(grid_lr.best_params_, grid_lr.best_score_)

  return grid_lr, grid_lr.best_score_

def get_auc_lr_valid(X, y, seed=17):
  #skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
  params_lr = { 'C': np.logspace(0, 1.5, 5) }
  grid_lr = GridSearchCV(LogisticRegression(random_state=seed, n_jobs=-1),
                         params_lr,
                         cv=4,
                         n_jobs=-1,
                         scoring='roc_auc')
  grid_lr.fit(X, y)
  print(grid_lr.best_params_, grid_lr.best_score_)

  return grid_lr, grid_lr.best_score_

def get_site_dict():
  with open(utils.PATH.COURSE_FILE('site_dic.pkl', 'kaggle_alice'), 'rb') as f:
    site_dict = pickle.load(f)
  site_dict_df = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])

  return site_dict

def get_site_freqs(fname, df, top_cnt=30):
  fpath = utils.PATH.COURSE_FILE((fname+'_c{0}.pkl').format(top_cnt), 'kaggle_alice')

  if os.path.exists(fpath):
    try:
      with open(fpath, 'rb') as f:
        freqs_dict = pickle.load(f)
      return freqs_dict
    except:
      os.remove(fpath)

  print('------------- load freqs: {0} -------------'.format(fname))
  frqs = Counter()
  for i, row in df.iterrows():
    frqs.update(row)
  del frqs[0]
  freqs_dict = dict(frqs.most_common(top_cnt))

  with open(fpath, 'wb') as f:
    pickle.dump(freqs_dict, f, protocol=2)

  return freqs_dict


def f(X, y, seed=17):
  skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
  svm = LinearSVR(random_state=seed)
  svm_params = { 'C': np.logspace(-2, 2, 5) }
  svm_grid = GridSearchCV(svm, svm_params, cv=skf, scoring='roc_auc', n_jobs=-1)
  svm_grid.fit(X, y)

  print(svm_grid.best_params_, svm_grid.best_score_)

  return svm_grid, svm_grid.best_score_

def g(X, y, seed=17):
  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
  sgd = SGDClassifier(loss='log', random_state=seed, n_jobs=-1)
  sgd_params = { 'l1_ratio': np.linspace(0.1, 0.3, 30) }
  sgd_grid = GridSearchCV(sgd, sgd_params, cv=skf, scoring='roc_auc', n_jobs=-1)
  sgd_grid.fit(X, y)

  print(sgd_grid.best_params_, sgd_grid.best_score_)

  return sgd_grid, sgd_grid.best_score_

def gbm(X, y, seed=17):
  print('GBM')
  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
  gbm = XGBClassifier(learning_rate=0.15, max_depth=7, n_jobs=-1, random_state=17)
  params_gbm = { 'n_estimators': [300] }
  grid_gbm = GridSearchCV(gbm,
                          params_gbm,
                          cv=3,
                          n_jobs=-1,
                          scoring='roc_auc')
  grid_gbm.fit(X, y)
  print(grid_gbm.best_params_, grid_gbm.best_score_)

  return grid_gbm, grid_gbm.best_score_



















