import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def header(): return 'LECTURE 10: Gradient Boosting Machine https://habrahabr.ru/company/ods/blog/327250/';

def run():

  homework()

  return

def homework():


  train = pd.read_csv(utils.PATH.COURSE_FILE('flight_delays_train.csv'))
  test = pd.read_csv(utils.PATH.COURSE_FILE('flight_delays_test.csv'))
  #print(train.head())
  print(train.shape)
  print(test.shape)

  #benchmark1(train, test)
  #benchmark2(train, test)
  #benchmark2_71865(train, test)
  benchmark2(train, test)

  #lin_comb(train);

  return

def benchmark1(train, test):
  X_train = train[['Distance', 'DepTime']].values
  y_train = train['dep_delayed_15min'].map({ 'Y': 1, 'N': 0 }).values
  X_test  = test[['Distance', 'DepTime']].values

  X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)

  scaler = StandardScaler()
  X_train_part = scaler.fit_transform(X_train_part)
  X_valid = scaler.transform(X_valid)

  lr = LogisticRegression()
  lr.fit(X_train_part, y_train_part)
  y_pred = lr.predict_proba(X_valid)[:, 1]

  score = roc_auc_score(y_valid, y_pred)
  print(score)

  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  lr.fit(X_train_scaled, y_train)
  y_pred = lr.predict_proba(X_test_scaled)[:, 1]

  write_to_submission_file(y_pred, 'logit_2feat.csv')

  return

def benchmark2_71865(train, test):
  y_train = train['dep_delayed_15min'].map({ 'Y': 1, 'N': 0 }).values
  train_cnt = y_train.shape[0]

  full = pd.concat([train.drop('dep_delayed_15min', axis=1), test])
  full['Route'] = full['Origin'] + '-' + full['Dest']

  res = full[['Distance', 'DepTime']]

  res = pd.concat([res, pd.get_dummies(full['Month'],      prefix='month')], axis=1)
  res = pd.concat([res, pd.get_dummies(full['DayofMonth'], prefix='mday')],  axis=1)
  res = pd.concat([res, pd.get_dummies(full['DayOfWeek'],  prefix='wday')],  axis=1)
  res = pd.concat([res, pd.get_dummies(full['UniqueCarrier'])], axis=1)
  res = pd.concat([res, pd.get_dummies(full['Route'])], axis=1)

  scaler = StandardScaler()
  res[['Distance', 'DepTime']] = scaler.fit_transform(res[['Distance', 'DepTime']])

  X_train, X_test = res.iloc[:train_cnt], res.iloc[train_cnt:]
  X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)

  X_train_part = scaler.fit_transform(X_train_part)
  X_valid = scaler.transform(X_valid)

  # LogisticRegression
  print('LOGIT')

  lr = LogisticRegressionCV(Cs=[0.1], cv=None, scoring='roc_auc', n_jobs=-1, random_state=17)
  print(X_train.shape, y_train.shape)
  lr.fit(X_train, y_train)
  print ('Scores:', lr.scores_)
  print ('C_:', lr.C_)

  y_pred = lr.predict_proba(X_test)[:, 1]
  write_to_submission_file(y_pred, 'lr.csv')
  write_to_submission_file(lr.predict_proba(X_train)[:, 1], 'lr_train.csv')

  # GradientBoosting
  print('GBM')

  gbm = XGBClassifier(n_estimators=500, learning_rate=0.3, max_depth=7, n_jobs=-1, random_state=17)
  gbm.fit(X_train, y_train)
  y_pred = gbm.predict_proba(X_test)[:, 1]
  write_to_submission_file(y_pred, 'gbm.csv')
  write_to_submission_file(gbm.predict_proba(X_train)[:, 1], 'gbm_train.csv')
  return


  #gbm = XGBClassifier(n_estimators=100, learning_rate=0.3, n_jobs=-1, random_state=17)
  #params_gbm = { 'max_depth': [4, 5, 6] }
  #grid_gbm = GridSearchCV(gbm,
  #                        params_gbm,
  #                        cv=3,
  #                        n_jobs=-1,
  #                        scoring='roc_auc')
  #grid_gbm.fit(X_train, y_train)
  #print(grid_gbm.best_params_, grid_gbm.best_score_)
  #
  #y_pred = grid_gbm.predict_proba(X_test)[:, 1]
  #
  #write_to_submission_file(y_pred, 'gbm.csv')

  return

def benchmark2_72138(train, test):
  y_train = train['dep_delayed_15min'].map({ 'Y': 1, 'N': 0 }).values
  train_cnt = y_train.shape[0]

  full = pd.concat([train.drop('dep_delayed_15min', axis=1), test])
  full['Route'] = full['Origin'] + '-' + full['Dest']

  res = full[['Distance', 'DepTime']]

  res = pd.concat([res, pd.get_dummies(full['Month'],      prefix='month')], axis=1)
  res = pd.concat([res, pd.get_dummies(full['DayofMonth'], prefix='mday')],  axis=1)
  res = pd.concat([res, pd.get_dummies(full['DayOfWeek'],  prefix='wday')],  axis=1)
  res = pd.concat([res, pd.get_dummies(full['UniqueCarrier'])], axis=1)
  res = pd.concat([res, pd.get_dummies(full['Route'])], axis=1)

  scaler = StandardScaler()
  res[['Distance', 'DepTime']] = scaler.fit_transform(res[['Distance', 'DepTime']])

  X_train, X_test = res.iloc[:train_cnt], res.iloc[train_cnt:]
  X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)

  X_train_part = scaler.fit_transform(X_train_part)
  X_valid = scaler.transform(X_valid)

  # LogisticRegression
  print('LOGIT')

  lr = LogisticRegressionCV(Cs=[0.1], cv=None, scoring='roc_auc', n_jobs=-1, random_state=17)
  print(X_train.shape, y_train.shape)
  lr.fit(X_train, y_train)
  print ('Scores:', lr.scores_)
  print ('C_:', lr.C_)

  y_pred = lr.predict_proba(X_test)[:, 1]
  write_to_submission_file(y_pred, 'lr.csv')

  # GradientBoosting
  print('GBM')

  #gbm = XGBClassifier(n_estimators=500, max_depth=7, n_jobs=-1, random_state=17)
  #params_gbm = { 'learning_rate': [0.275, 0.3, 0.325] }
  #grid_gbm = GridSearchCV(gbm,
  #                       params_gbm,
  #                       cv=3,
  #                       n_jobs=-1,
  #                       scoring='roc_auc')
  #grid_gbm.fit(X_train, y_train)
  #print(grid_gbm.best_params_, grid_gbm.best_score_)
  #
  #y_pred = grid_gbm.predict_proba(X_test)[:, 1]
  #
  #write_to_submission_file(y_pred, 'gbm.csv')

  gbm = XGBClassifier(n_estimators=500, learning_rate=0.3, max_depth=7, n_jobs=-1, random_state=17)
  gbm.fit(X_train, y_train)
  y_pred = gbm.predict_proba(X_test)[:, 1]
  write_to_submission_file(y_pred, 'gbm.csv')

  print('LINEAR COMBINATION')

  lr_pred  = pd.read_csv(utils.PATH.STORE_FOR('xgboost\submissions', 'lr.csv'))['dep_delayed_15min'].values
  gbm_pred = pd.read_csv(utils.PATH.STORE_FOR('xgboost\submissions', 'gbm.csv'))['dep_delayed_15min'].values

  alpha = 0.2
  pred = np.clip(alpha*lr_pred + (1 - alpha)*gbm_pred, 0, 1)
  write_to_submission_file(pred, 'lin_comb.csv')

  return

def benchmark2(train, test):
  y_train = train['dep_delayed_15min'].map({ 'Y': 1, 'N': 0 }).values
  train_cnt = y_train.shape[0]

  full = pd.concat([train.drop('dep_delayed_15min', axis=1), test])
  full['Route'] = full['Origin'] + '-' + full['Dest']

  res = full[['Distance', 'DepTime']]

  res = pd.concat([res, pd.get_dummies(full['Month'],      prefix='month')], axis=1)
  res = pd.concat([res, pd.get_dummies(full['DayofMonth'], prefix='mday')],  axis=1)
  res = pd.concat([res, pd.get_dummies(full['DayOfWeek'],  prefix='wday')],  axis=1)
  res = pd.concat([res, pd.get_dummies(full['UniqueCarrier'])], axis=1)
  res = pd.concat([res, pd.get_dummies(full['Route'])], axis=1)

  scaler = StandardScaler()
  res[['Distance', 'DepTime']] = scaler.fit_transform(res[['Distance', 'DepTime']])

  X_train, X_test = res.iloc[:train_cnt], res.iloc[train_cnt:]
  X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=17)

  X_train_part = scaler.fit_transform(X_train_part)
  X_valid = scaler.transform(X_valid)

  # LogisticRegression
  print('LOGIT')

  preds = []
  skf = StratifiedKFold(n_splits=3)
  for train_index, test_index in skf.split(X_train, y_train):
    print("LOGIT with some FOLD")
    X_tr, X_tt = X_train.values[train_index], X_train.values[test_index]
    y_tr, y_tt = y_train[train_index], y_train[test_index]

    lr = LogisticRegressionCV(Cs=[0.05, 0.1, 0.15], cv=3, scoring='roc_auc', n_jobs=-1, random_state=17)
    print(X_tr.shape, y_tr.shape)
    lr.fit(X_tr, y_tr)
    print ('FOLD Scores:', lr.scores_)
    print ('FOLD C_:', lr.C_)

    y_pred = lr.predict_proba(X_test)[:, 1]
    preds.append(y_pred)



  lr = LogisticRegressionCV(Cs=[0.1], cv=None, scoring='roc_auc', n_jobs=-1, random_state=17)
  print(X_train.shape, y_train.shape)
  lr.fit(X_train, y_train)
  print ('Scores:', lr.scores_)
  print ('C_:', lr.C_)

  y_pred = lr.predict_proba(X_test)[:, 1]
  write_to_submission_file(y_pred, 'lr_test.csv')

  return

def write_to_submission_file(y_pred, out_file, target='dep_delayed_15min', index_label='id'):
  pred_df = pd.DataFrame(y_pred,
                         index=np.arange(0, y_pred.shape[0]),
                         columns=[target])
  submission_fname = utils.PATH.STORE_FOR('xgboost\submissions', out_file)
  pred_df.to_csv(submission_fname, index_label=index_label)
  return
