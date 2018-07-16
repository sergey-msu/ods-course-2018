import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from alice import data_utils

def header(): return 'Week 4: Modeling'

def run():
  ensure_files()

  #model_comparison()
  #parameter_selection_10users()
  #parameter_selection_150users()
  #single_user_idendification()
  learning_curves()

  return

def ensure_files():
  data_utils.ensure_train_set_with_fe(data_utils.DIR_3USERS, 10, 10)
  data_utils.ensure_train_set_with_fe(data_utils.DIR_10USERS, 10, 10)
  data_utils.ensure_train_set_with_fe(data_utils.DIR_150USERS, 10, 10)
  return

def model_comparison():
  with open(utils.PATH.STORE_FOR('alice', 'sparse_data_10users_s10_w10.pkl'), 'rb') as f:
    X_sparse_10users = pickle.load(f)
  y_10users = pd.read_csv(utils.PATH.STORE_FOR('alice', 'extra_data_10users_s10_w10.csv'))['target'].values

  print(X_sparse_10users.shape)
  print(y_10users.shape)

  X_train, X_valid, y_train, y_valid = train_test_split(X_sparse_10users, y_10users,
                                                        test_size=0.3, random_state=17, stratify=y_10users)

  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)

  # KNN classifier
  def knn():
    print('\nKNN')

    knn = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
    cv_score = np.mean(cross_val_score(knn, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1))
    print('CV score on test:', cv_score)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_valid)
    score = accuracy_score(y_valid, y_pred)
    print('Accuracy score on valid:', score)
    return
  #knn()

  # RandomForest classifier
  def rf():
    print('\nRF')

    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=17, n_jobs=-1)
    rf.fit(X_train, y_train)
    print('OOB score:', rf.oob_score_)

    y_pred = rf.predict(X_valid)
    score = accuracy_score(y_valid, y_pred)
    print('Accuracy score on valid:', score)
    return
  #rf()

  # LogisticRegression classifier
  def lr():
    print('\nLR')

    lr = LogisticRegression(random_state=17)
    cv_score = np.mean(cross_val_score(lr, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1))
    print('CV score on test:', cv_score)

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_valid)
    score = accuracy_score(y_valid, y_pred)
    print('Accuracy score on valid:', score)
    return
  #lr()

  # LogisticRegressionCV classifier
  def lrcv():
    print('\nLR-CV')

    cs = np.logspace(-4, 2, 10)
    lrcv1 = LogisticRegressionCV(Cs=cs, multi_class='multinomial', cv=skf, random_state=17, n_jobs=-1)
    lrcv1.fit(X_train, y_train)

    scores = np.array(list(lrcv1.scores_.values())[0])

    print(cs)
    print(scores.shape)
    av_scores = np.mean(scores, axis=0)
    print('average scores:', av_scores)
    max_score = np.max(av_scores)
    print('max score:', max_score)
    print('best C:', lrcv1.C_)

    plt.semilogx(cs, av_scores)
    plt.show()

    cs = np.linspace(0.1, 7, 20)
    lrcv2 = LogisticRegressionCV(Cs=cs, multi_class='multinomial', cv=skf, random_state=17, n_jobs=-1)
    lrcv2.fit(X_train, y_train)

    scores = np.array(list(lrcv2.scores_.values())[0])

    av_scores = np.mean(scores, axis=0)
    print('average scores:', av_scores)
    max_score = np.max(av_scores)
    print('max score:', max_score)
    print('best C:', lrcv2.C_)

    plt.plot(cs, av_scores)
    plt.show()

    y_pred = lrcv2.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print('Accuracy score on valid:', accuracy)
    return
  lrcv()

  # SVM classifier
  def svm():
    print('SVM')

    svm = LinearSVC(C=1, random_state=17)
    score = np.mean(cross_val_score(svm, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1))
    print(score)

    svm = LinearSVC(random_state=17)
    svm_params = { 'C': np.linspace(1e-4, 1e4, 10) }
    svm_grid = GridSearchCV(svm, svm_params, scoring='accuracy', cv=skf, n_jobs=-1)
    svm_grid.fit(X_train, y_train)

    print(svm_grid.best_params_)
    print(svm_grid.best_score_)
    plot_validation_curves(svm_params['C'], svm_grid.cv_results_)

    svm = LinearSVC(random_state=17)
    svm_params = { 'C': np.linspace(1e-3, 1, 30) }
    svm_grid = GridSearchCV(svm, svm_params, cv=skf, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train, y_train)

    print(svm_grid.best_params_)
    print(svm_grid.best_score_)
    plot_validation_curves(svm_params['C'], svm_grid.cv_results_)

    y_pred = svm_grid.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print('Accuracy score on valid:', accuracy)

    return
  #svm()

  return

def parameter_selection_10users():
  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
  estimator = LinearSVC(C=0.1, random_state=17)

  for window_size, session_length in itertools.product([10, 7, 5], [15, 10, 7, 5]):
    if (window_size <= session_length):
      data_utils.ensure_train_set_with_fe(data_utils.DIR_10USERS, session_length, window_size)
      path_to_X_pickle = utils.PATH.STORE_FOR('alice', 'sparse_data_10users_s{0}_w{1}.pkl'.format(session_length, window_size))
      path_to_y_pickle = utils.PATH.STORE_FOR('alice', 'extra_data_10users_s{0}_w{1}.csv'.format(session_length, window_size))

      score, accuracy = model_assessment(estimator, path_to_X_pickle, path_to_y_pickle, cv=skf)
      print('S={0} W={1}: '.format(session_length, window_size), score, accuracy)

  return

def parameter_selection_150users():
  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
  estimator = LinearSVC(C=0.1, random_state=17)

  for window_size, session_length in [(5, 5), (7, 7), (10, 10)]:
    print('started for S={0} W={1}'.format(session_length, window_size))

    data_utils.ensure_train_set_with_fe(data_utils.DIR_150USERS, session_length, window_size)
    path_to_X_pickle = utils.PATH.STORE_FOR('alice', 'sparse_data_150users_s{0}_w{1}.pkl'.format(session_length, window_size))
    path_to_y_pickle = utils.PATH.STORE_FOR('alice', 'extra_data_150users_s{0}_w{1}.csv'.format(session_length, window_size))

    score, accuracy = model_assessment(estimator, path_to_X_pickle, path_to_y_pickle, cv=skf)
    print('S={0} W={1}: '.format(session_length, window_size), score, accuracy)

  return

def plot_validation_curves(param_values, grid_cv_results_):
  train_mu, train_std = grid_cv_results_['mean_train_score'], grid_cv_results_['std_train_score']
  valid_mu, valid_std = grid_cv_results_['mean_test_score'], grid_cv_results_['std_test_score']
  train_line = plt.plot(param_values, train_mu, '-', label='train', color='green')
  valid_line = plt.plot(param_values, valid_mu, '-', label='test', color='red')
  plt.fill_between(param_values, train_mu - train_std, train_mu + train_std, edgecolor='none',
                   facecolor=train_line[0].get_color(), alpha=0.2)
  plt.fill_between(param_values, valid_mu - valid_std, valid_mu + valid_std, edgecolor='none',
                   facecolor=valid_line[0].get_color(), alpha=0.2)
  plt.legend()
  plt.show()
  return

def model_assessment(estimator, path_to_X_pickle, path_to_y_pickle, cv, random_state=17, test_size=0.3):
  '''
  Estimates CV-accuracy for (1 - test_size) share of (X_sparse, y)
  loaded from path_to_X_pickle and path_to_y_pickle and holdout accuracy for (test_size) share of (X_sparse, y).
  The split is made with stratified train_test_split with params random_state and test_size.

  :param estimator – Scikit-learn estimator (classifier or regressor)
  :param path_to_X_pickle – path to pickled sparse X (instances and their features)
  :param path_to_y_pickle – path to pickled y (responses)
  :param cv – cross-validation as in cross_val_score (use StratifiedKFold here)
  :param random_state –  for train_test_split
  :param test_size –  for train_test_split

  :returns mean CV-accuracy for (X_train, y_train) and accuracy for (X_valid, y_valid) where (X_train, y_train)
  and (X_valid, y_valid) are (1 - test_size) and (testsize) shares of (X_sparse, y).
  '''

  with open(path_to_X_pickle, 'rb') as f:
    X_sparse = pickle.load(f)
  y = pd.read_csv(path_to_y_pickle)['target'].values

  X_train, X_valid, y_train, y_valid = train_test_split(X_sparse, y, test_size=test_size,
                                                      random_state=random_state, stratify=y)

  score = np.mean(cross_val_score(estimator, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1))

  estimator.fit(X_train, y_train)

  y_pred = estimator.predict(X_valid)
  accuracy = accuracy_score(y_valid, y_pred)

  return score, accuracy

def read_150users_data():
  data_utils.ensure_train_set_with_fe(data_utils.DIR_150USERS, 10, 10)
  path_to_X_pickle = utils.PATH.STORE_FOR('alice', 'sparse_data_150users_s10_w10.pkl')
  path_to_y_pickle = utils.PATH.STORE_FOR('alice', 'extra_data_150users_s10_w10.csv')
  with open(path_to_X_pickle, 'rb') as f:
    X_sparse = pickle.load(f)
  y = pd.read_csv(path_to_y_pickle)['target'].values

  return X_sparse, y

def single_user_idendification():
  X, y = read_150users_data()
  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)

  X_train, X_valid, y_train, y_valid = train_test_split(X_sparse, y, test_size=0.3, random_state=17, stratify=y)

  lrcv = LogisticRegressionCV(Cs=[0.46315789], multi_class='ovr', cv=skf, random_state=17, n_jobs=-1)
  lrcv.fit(X_train, y_train)
  cv_scores_by_user = {}
  class_distr = np.bincount(y.astype(int))
  acc_diff_vs_constant = {}

  for user_id in lrcv.scores_:
    cv_score = np.mean(lrcv.scores_[user_id])
    other_ids_cnt = (len(y) - class_distr[user_id])/len(y)
    acc_diff_vs_constant[user_id] = cv_score-other_ids_cnt
    print('User {0}, CV score etc.: {1}\t{2}\t{3}'.format(user_id, cv_score, other_ids_cnt, cv_score-other_ids_cnt))

  num_better_than_default = (np.array(list(acc_diff_vs_constant.values())) > 0).sum()
  print(num_better_than_default/len(acc_diff_vs_constant))

  return

def learning_curves():
  X, y = read_150users_data()
  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
  y_binary_128 = (y == 128).astype(int)

  def plot_learning_curve(val_train, val_test, train_sizes, xlabel='Training Set Size', ylabel='score'):
    def plot_with_err(x, data, **kwargs):
        mu, std = data.mean(1), data.std(1)
        lines = plt.plot(x, mu, '-', **kwargs)
        plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                         facecolor=lines[0].get_color(), alpha=0.2)
    plot_with_err(train_sizes, val_train, label='train')
    plot_with_err(train_sizes, val_test, label='valid')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(loc='lower right');
    plt.show()

  train_sizes = np.linspace(0.25, 1, 20)
  lrcv = LogisticRegressionCV(Cs=[0.46315789], multi_class='ovr', cv=skf, random_state=17, n_jobs=-1)
  n_train, val_train, val_test = learning_curve(lrcv, X, y_binary_128,
                                                train_sizes=train_sizes,
                                                cv=skf, scoring='accuracy',
                                                shuffle=True,
                                                random_state=17, n_jobs=-1)
  plot_learning_curve(val_train, val_test, n_train, xlabel='train_size', ylabel='accuracy')

  return

