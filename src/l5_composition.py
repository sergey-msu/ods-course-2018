from __future__ import division,print_function
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_circles
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

def header(): return 'LECTURE 5: Compositions: bagging, random forest https://habrahabr.ru/company/ods/blog/324402/'

def run():

  #df = pd.read_csv(utils.PATH.COURSE_FILE('telecom_churn.csv'))
  #bagging(df)
  #tree_bagging_forest_regression()
  #tree_bagging_forest_classification()
  #forest_example(df)

  homework()

  return

def bagging(df):
  plt.style.use('ggplot')
  plt.rcParams['figure.figsize'] = 10, 6


  fig = sns.kdeplot(df[df['Churn']==False]['Customer service calls'], label='Loyal')
  fig = sns.kdeplot(df[df['Churn']==True]['Customer service calls'], label='Churn')
  fig.set(xlabel='Number of Calls', ylabel='Density')

  loyal_calls = df[df['Churn']==False]['Customer service calls'].values
  churn_calls = df[df['Churn']==True]['Customer service calls'].values

  np.random.seed(0)

  loyal_mean_scores = [ np.mean(sample) for sample in get_bootstrap_samples(loyal_calls, 1000) ]
  churn_mean_scores = [ np.mean(sample) for sample in get_bootstrap_samples(churn_calls, 1000) ]

  print('Service calls fromloyals: mean interval', stat_intervals(loyal_mean_scores, 0.05))
  print('Service calls fromloyals: mean interval', stat_intervals(churn_mean_scores, 0.05))

  #plt.show()
  return

def get_bootstrap_samples(data, n_samples):
  indices = np.random.randint(0, len(data), (n_samples, len(data)))
  samples = data[indices]
  return samples

def stat_intervals(stat, alpha):
  boundaries = np.percentile(stat, [100*alpha/2.0, 100*(1 - alpha/2.0)])
  return boundaries

def tree_bagging_forest_regression():
  np.random.seed(42)
  plt.rcParams['figure.figsize'] = 8, 6

  n_train = 150
  n_test  = 1000
  noise   = 0.1

  def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

  def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)
    y = f(X) + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))

    return X, y

  X_train, y_train = generate(n_samples=n_train, noise=noise)
  X_test, y_test   = generate(n_samples=n_test, noise=noise)

  # One decision tree regressor
  dtree = DecisionTreeRegressor().fit(X_train, y_train)
  d_predict = dtree.predict(X_test)
  plot_regression(X_train, y_train, X_test, y_test, d_predict, f, 'Decision Trees')


  # Bagging decision tree regressor
  bdt = BaggingRegressor(DecisionTreeRegressor()).fit(X_train, y_train)
  bdt_predict = bdt.predict(X_test)
  plot_regression(X_train, y_train, X_test, y_test, bdt_predict, f, 'Bagging: Decision Trees')

  # Random Forest
  rf = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
  rf_predict = rf.predict(X_test)
  plot_regression(X_train, y_train, X_test, y_test, rf_predict, f, 'Random Forest')

  plt.show()

def tree_bagging_forest_classification():
  plt.style.use('ggplot')
  plt.rcParams['figure.figsize'] = 10, 6

  np.random.seed(42)

  X, y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=42)
  X_train_circles, X_test_circles, y_train_circles, y_test_circles = train_test_split(X, y, test_size=0.2)

  dtree = DecisionTreeClassifier(random_state=42)
  dtree.fit(X_train_circles, y_train_circles)
  plot_classification(X, y, dtree, 'Decision Tree')

  b_dtree = BaggingClassifier(DecisionTreeClassifier(), n_estimators=300, random_state=42)
  b_dtree.fit(X_train_circles, y_train_circles)
  plot_classification(X, y, b_dtree, 'Bagging: Decision Tree')

  rf = RandomForestClassifier(n_estimators=300, random_state=42)
  rf.fit(X_train_circles, y_train_circles)
  plot_classification(X, y, rf, 'Random Forest')

  return

def plot_regression(X_train, y_train, X_test, y_test, predict, f, title):
  plt.figure(figsize=(10, 6))
  plt.plot(X_test, f(X_test), "b")
  plt.scatter(X_train, y_train, c="b", s=20)
  plt.plot(X_test, predict, "r", lw=2)
  plt.xlim([-5, 5])
  plt.title(title+", MSE = %.2f" % np.sum((y_test - predict) ** 2));

def plot_classification(X, y, alg, title):
  x_range = np.linspace(X.min(), X.max(), 100)
  xx1, xx2 = np.meshgrid(x_range, x_range)
  y_hat = alg.predict(np.c_[xx1.ravel(), xx2.ravel()])
  y_hat = y_hat.reshape(xx1.shape)
  plt.contourf(xx1, xx2, y_hat, alpha=0.2)
  plt.scatter(X[:,0], X[:,1], c=y, cmap='autumn')
  plt.title(title)
  plt.show()

def forest_example(df):
  cols = []
  for i in df.columns:
    if (df[i].dtype=='float64') or (df[i].dtype=='int64'):
      cols.append(i)

  X = df[cols].copy()
  y = np.asarray(df['Churn'], dtype='int8')

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  rf = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
  results = cross_val_score(rf, X, y, cv=skf)

  print('CV accuracy score: {:.2f}%'.format(results.mean()*100))

  # validation curves

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

  train_acc = []
  test_acc  = []
  temp_train_acc = []
  temp_test_acc  = []
  trees_grid = [5, 10, 15, 20, 30, 75, 100]

  for ntrees in trees_grid:
    rf = RandomForestClassifier(n_estimators=ntrees, random_state=42, n_jobs=-1, oob_score=True)
    temp_train_acc = []
    temp_test_acc  = []
    for train_index, test_index in skf.split(X, y):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y[train_index], y[test_index]
      rf.fit(X_train, y_train)
      temp_train_acc.append(rf.score(X_train, y_train))
      temp_test_acc.append(rf.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

  train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)

  print('Best accuracy on CV is {:.2f}% with {} trees'.format(max(test_acc.mean(axis=1))*100,
                                                              trees_grid[np.argmax(test_acc.mean(axis=1))]))

  # plot

  plt.style.use('ggplot')

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
  ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
  ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
  ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
  ax.legend(loc='best')
  ax.set_ylim([0.88,1.02])
  ax.set_ylabel("Accuracy")
  ax.set_xlabel("N_estimators")

  #  # Regularization: max_depth
  #
  #  train_acc = []
  #  test_acc = []
  #  temp_train_acc = []
  #  temp_test_acc = []
  #  max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]
  #
  #  for max_depth in max_depth_grid:
  #    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, max_depth=max_depth)
  #    temp_train_acc = []
  #    temp_test_acc = []
  #    for train_index, test_index in skf.split(X, y):
  #      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  #      y_train, y_test = y[train_index], y[test_index]
  #      rfc.fit(X_train, y_train)
  #      temp_train_acc.append(rfc.score(X_train, y_train))
  #      temp_test_acc.append(rfc.score(X_test, y_test))
  #    train_acc.append(temp_train_acc)
  #    test_acc.append(temp_test_acc)
  #
  #  train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
  #  print("Best accuracy on CV is {:.2f}% with {} max_depth".format(max(test_acc.mean(axis=1))*100,
  #                                                                  max_depth_grid[np.argmax(test_acc.mean(axis=1))]))
  #
  #  fig, ax = plt.subplots(figsize=(8, 4))
  #  ax.plot(max_depth_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
  #  ax.plot(max_depth_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
  #  ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
  #  ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
  #  ax.legend(loc='best')
  #  ax.set_ylim([0.88,1.02])
  #  ax.set_ylabel("Accuracy")
  #  ax.set_xlabel("Max_depth")
  #
  #  # Regularization: min_samples_leaf
  #
  #  train_acc = []
  #  test_acc = []
  #  temp_train_acc = []
  #  temp_test_acc = []
  #  min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]
  #
  #  for min_samples_leaf in min_samples_leaf_grid:
  #    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
  #                                 oob_score=True, min_samples_leaf=min_samples_leaf)
  #    temp_train_acc = []
  #    temp_test_acc = []
  #    for train_index, test_index in skf.split(X, y):
  #      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  #      y_train, y_test = y[train_index], y[test_index]
  #      rfc.fit(X_train, y_train)
  #      temp_train_acc.append(rfc.score(X_train, y_train))
  #      temp_test_acc.append(rfc.score(X_test, y_test))
  #    train_acc.append(temp_train_acc)
  #    test_acc.append(temp_test_acc)
  #
  #  train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
  #  print("Best accuracy on CV is {:.2f}% with {} min_samples_leaf".format(max(test_acc.mean(axis=1))*100,
  #                                                                         min_samples_leaf_grid[np.argmax(test_acc.mean(axis=1))]))
  #
  #  fig, ax = plt.subplots(figsize=(8, 4))
  #  ax.plot(min_samples_leaf_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
  #  ax.plot(min_samples_leaf_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
  #  ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
  #  ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
  #  ax.legend(loc='best')
  #  ax.set_ylim([0.88,1.02])
  #  ax.set_ylabel("Accuracy")
  #  ax.set_xlabel("Min_samples_leaf")
  #
  #  # Regularization: max_features
  #
  #  train_acc = []
  #  test_acc = []
  #  temp_train_acc = []
  #  temp_test_acc = []
  #  max_features_grid = [2, 4, 6, 8, 10, 12, 14, 16]
  #
  #  for max_features in max_features_grid:
  #    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, max_features=max_features)
  #    temp_train_acc = []
  #    temp_test_acc = []
  #    for train_index, test_index in skf.split(X, y):
  #      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  #      y_train, y_test = y[train_index], y[test_index]
  #      rfc.fit(X_train, y_train)
  #      temp_train_acc.append(rfc.score(X_train, y_train))
  #      temp_test_acc.append(rfc.score(X_test, y_test))
  #    train_acc.append(temp_train_acc)
  #    test_acc.append(temp_test_acc)
  #
  #  train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
  #  print("Best accuracy on CV is {:.2f}% with {} max_features".format(max(test_acc.mean(axis=1))*100,
  #                                                                     max_features_grid[np.argmax(test_acc.mean(axis=1))]))
  #
  #  fig, ax = plt.subplots(figsize=(8, 4))
  #  ax.plot(max_features_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
  #  ax.plot(max_features_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
  #  ax.fill_between(max_features_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
  #  ax.fill_between(max_features_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
  #  ax.legend(loc='best')
  #  ax.set_ylim([0.88,1.02])
  #  ax.set_ylabel("Accuracy")
  #  ax.set_xlabel("Max_features")

  # Parameters optimization

  parameters = { 'max_features': [4, 7, 10, 13],
                 'min_samples_leaf': [1, 3, 5, 7],
                 'max_depth': [5, 10, 15, 20] }
  rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
  gcv = GridSearchCV(rf, parameters, n_jobs=-1, cv=skf, verbose=1)
  gcv.fit(X, y)


  plt.show()

def homework():
  # ---------- 1
  p = 0.8
  q = 1 - p
  n = 7
  #prob = C(7,4)p^4q^3 + C(7,5)p^5q^2 + C(7,6)p^6q + C(7,7)p^7 = 0.9666

  # ---------- prepare data

  def impute_nan_with_median(table):
    for col in table.columns:
      table[col] = table[col].fillna(table[col].median())
    return table

  data = pd.read_csv(utils.PATH.COURSE_FILE('credit_scoring_sample.csv'), sep=';')
  print(data.head())
  print(data.dtypes)

  ax = data['SeriousDlqin2yrs'].hist(orientation='horizontal', color='red')
  ax.set_xlabel('number_of_observations')
  ax.set_ylabel('unique_value')
  ax.set_title('Target distribution')

  print('Distripution of target:')
  print(data['SeriousDlqin2yrs'].value_counts() / data.shape[0])

  independent_column_names = [x for x in data if x != 'SeriousDlqin2yrs' ]
  print(independent_column_names)

  table = impute_nan_with_median(data)
  X = table[independent_column_names]
  y = table['SeriousDlqin2yrs']

  # ---------- 2

  faulted_month_income = table[table['SeriousDlqin2yrs']==1]['MonthlyIncome'].values
  good_month_income    = table[table['SeriousDlqin2yrs']==0]['MonthlyIncome'].values

  np.random.seed(17)

  faulted_mean_scores = [ np.mean(sample) for sample in get_bootstrap_samples(faulted_month_income, 1000) ]
  good_mean_scores    = [ np.mean(sample) for sample in get_bootstrap_samples(good_month_income, 1000) ]

  bad_income_lower, bad_income_upper   = stat_intervals(faulted_mean_scores, 0.10)
  good_income_lower, good_income_upper = stat_intervals(good_mean_scores, 0.10)

  print('Faulted mean interval', (bad_income_lower, bad_income_upper))
  print('Good mean interval', (good_income_lower, good_income_upper))
  print(int(good_income_lower - bad_income_upper))

  # ---------- 3

  dt = DecisionTreeClassifier(random_state=17, class_weight='balanced')
  max_depth_values = [5, 6, 7, 8, 9]
  max_features_values = [4, 5, 6, 7]
  tree_params = { 'max_depth': max_depth_values, 'max_features': max_features_values }

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
  gcv = GridSearchCV(dt, tree_params, n_jobs=-1, cv=skf, scoring='roc_auc', verbose=1)
  gcv.fit(X, y)

  print(gcv.best_score_)
  print(gcv.best_params_)
  print(gcv.cv_results_['mean_train_score'].std())
  print(gcv.cv_results_['mean_test_score'].std())

  dt = DecisionTreeClassifier(random_state=17, class_weight='balanced', max_depth=7, max_features=6)
  score = cross_val_score(dt, X, y, cv=skf, scoring='roc_auc')
  print(score.std())

  # ---------- 4

  max_depth = 7 #gcv.best_params_['max_depth']
  max_features = 6 #gcv.best_params_['max_features']
  n_trees = 10
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

  rj = RandomJungleClassifier(n_trees, max_depth, max_features)
  results = cross_val_score(rj, X, y, cv=skf, scoring='roc_auc')
  score = np.mean(results)
  print(score)

  # ---------- 5

  rf = RandomForestClassifier(n_jobs=1, random_state=17, max_depth=max_depth, max_features=max_features)
  results = cross_val_score(rf, X, y, cv=skf, scoring='roc_auc')
  score = np.mean(results)
  print(score)

  # ---------- 6

  max_depth_values = range(5, 16)
  max_features_values = range(4, 8)
  forest_params = {'max_depth': max_depth_values,
                   'max_features': max_features_values}

  rf = RandomForestClassifier(n_jobs=-1, random_state=17)
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
  gcv = GridSearchCV(rf, forest_params, n_jobs=-1, cv=skf, scoring='roc_auc', verbose=1)
  gcv.fit(X, y)

  print(gcv.best_score_)
  print(gcv.best_params_)

  # ---------- 7

  pipe = Pipeline([('scaler', StandardScaler()),
                   ('logit', LogisticRegression(class_weight='balanced', random_state=17))])
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
  logit_params = { 'logit__C': np.logspace(-8, 8, 17) }
  gcv = GridSearchCV(pipe, logit_params, n_jobs=-1, cv=skf, scoring='roc_auc', verbose=1)
  gcv.fit(X, y)
  print(gcv.best_score_)
  print(gcv.best_params_)
  print(gcv.cv_results_['mean_train_score'].mean())
  print(gcv.cv_results_['mean_test_score'].mean())

  # ---------- 8

  df = pd.read_csv(utils.PATH.COURSE_FILE('movie_reviews_train.csv'), nrows=50000)
  X_text = df['text']
  y_text = df['label']
  print(df.label.value_counts())

  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
  pipe = Pipeline([('vectorizer', CountVectorizer(max_features=100000, ngram_range = (1, 3))),
                   ('clf', LogisticRegression(random_state=17))])
  params = { 'clf__C': [0.1, 1, 10, 100] }
  gcv = GridSearchCV(pipe, params, n_jobs=-1, cv=skf, scoring='roc_auc', verbose=1)
  gcv.fit(X_text, y_text)
  print(gcv.best_score_)
  print(gcv.best_params_)
  #for c in [0.1, 1, 10, 100]:
  #  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
  #  pipe = Pipeline([('vectorizer', CountVectorizer(max_features=100000, ngram_range = (1, 3))),
  #                   ('clf', LogisticRegression(random_state=17, C=c))])
  #  score = cross_val_score(pipe, X_text, y_text, cv=skf, scoring='roc_auc')
  #  print(score.mean(), score.max())

  # ---------- 9

  skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
  pipe = Pipeline([('vectorizer', CountVectorizer(max_features=100000, ngram_range = (1, 3))),
                   ('clf', RandomForestClassifier(random_state=17, n_jobs=-1))])
  min_samples_leaf = [1, 2, 3]
  max_features = [0.3, 0.5, 0.7]
  max_depth = [None]
  params = { 'clf__min_samples_leaf': min_samples_leaf,
             'clf__max_features': max_features,
             'clf__max_depth': max_depth}
  gcv = GridSearchCV(pipe, params, n_jobs=-1, cv=skf, scoring='roc_auc', verbose=1)
  gcv.fit(X_text, y_text)
  print(gcv.best_score_)
  print(gcv.best_params_)

  #plt.show()
  return

class RandomJungleClassifier(BaseEstimator):
  def __init__(self, n_estimators, max_depth, max_features, random_state=17):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.max_features = max_features
    self.random_state = random_state
    self.jungle = None
    self.feat_ids_by_tree = None

  def fit(self, X, y):
    X = X.values
    y = np.reshape(y.values, [y.shape[0], 1])
    self.jungle = [None]*self.n_estimators
    self.feat_ids_by_tree = []

    data = np.concatenate([X, y], axis = 1)
    samples = self._get_bootstrap_samples(data, self.n_estimators)

    for i in range(self.n_estimators):
      random_seed = self.random_state + i
      feat_ids = self._get_features(X, random_seed)
      sample = samples[i]
      X_train = sample[:, feat_ids]
      y_train = sample[:, -1]

      tree = DecisionTreeClassifier(random_state=random_seed, max_features=self.max_features, max_depth = self.max_depth)
      tree.fit(X_train, y_train)

      self.feat_ids_by_tree.append(feat_ids)
      self.jungle[i] = tree

    return self

  def predict_proba(self, X):
    y_pred = None

    for i in range(self.n_estimators):
      feat_ids = self.feat_ids_by_tree[i]
      tree = self.jungle[i]
      X_test = X.values[:, feat_ids]
      y_pred += tree.predict_proba(X_test)

    y_pred /= self.n_estimators

    return y_pred


  def _get_features(self, X, random_seed):
    np.random.seed(random_seed)
    n = X.shape[1]
    f_num = self.max_features
    ids = list(range(n))
    res = []

    while f_num>0:
      i = np.random.randint(0, len(ids))
      idx = ids[i]
      res.append(idx)
      ids.remove(idx)
      f_num -= 1

    return res

  def _get_bootstrap_samples(self, data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples




