from __future__ import division, print_function
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from collections import defaultdict
from tqdm import tqdm_notebook
from sklearn.datasets import load_files
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, validation_curve, learning_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline

def header(): return 'LECTURE 4: Linear models https://habrahabr.ru/company/ods/blog/323890/'

def run():

  #example_microchips()
  #example_movies()
  #example_telekom()
  homework()

  return

def example_microchips():
  data = pd.read_csv(utils.PATH.COURSE_FILE('microchip_tests.txt'),
                     header=None,
                     names=('test1', 'test2', 'released'))
  print(data.info())
  print(data.head())

  X = data.ix[:, :2].values
  y = data.ix[:, 2].values

  #plot_data(X, y)

  poly = PolynomialFeatures(degree=7)
  X_poly = poly.fit_transform(X)

  C = 0.9
  logit = LogisticRegression(C=C, n_jobs=-1, random_state=17)
  logit.fit(X_poly, y)

  #plot_boundary(logit, X, y, grid_step=0.01, poly_featurizer=poly)

  success = round(logit.score(X_poly, y), 3)
  print(success)

  # CV optimization
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
  c_values = np.logspace(-2, 3, 500)
  logit_searcher = LogisticRegressionCV(Cs = c_values, cv=skf, verbose=1, n_jobs=-1)
  logit_searcher.fit(X_poly, y)

  print(logit_searcher.scores_)

  plt.show()

def plot_data(X, y):
  plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', label='Released')
  plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Bad')
  plt.xlabel('Test 1')
  plt.ylabel('Test 2')
  plt.title('2 tests of microchips')
  plt.legend()

def plot_boundary(clf, X, y, grid_step=0.01, poly_featurizer=None):
  x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
  y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                       np.arange(y_min, y_max, grid_step))

  # каждой точке в сетке [x_min, m_max]x[y_min, y_max]
  # ставим в соответствие свой цвет
  Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
  Z = Z.reshape(xx.shape)
  plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
  plot_data(X, y)

def example_movies():
  reviews_train = load_files(utils.PATH.COURSE_PATH('aclImdb\\train'))
  text_train, y_train = reviews_train.data, reviews_train.target
  print(np.bincount(y_train))

  reviews_test = load_files(utils.PATH.COURSE_PATH('aclImdb\\test'))
  text_test, y_test = reviews_test.data, reviews_test.target
  print(np.bincount(y_test))

  cv = CountVectorizer()
  cv.fit(text_train)
  print(len(cv.vocabulary_))
  print(cv.get_feature_names()[:50])
  print(cv.get_feature_names()[50000:50050])

  X_train = cv.transform(text_train)
  X_test = cv.transform(text_test)

  logit = LogisticRegression(n_jobs=-1, random_state=17)
  logit.fit(X_train, y_train)
  score_train = logit.score(X_train, y_train)
  score_test = logit.score(X_test, y_test)
  print(round(score_train, 3), round(score_test, 3))

  feature_names = cv.get_feature_names()
  visualize_coefficients(logit, feature_names)

  # regularization

  text_pipeline_logit = make_pipeline(CountVectorizer(), LogisticRegression(n_jobs=-1, random_state=17))
  text_pipeline_logit.fit(text_train, y_train)
  score = text_pipeline_logit.score(text_test, y_test)
  print(score)

  param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)}
  grid_logit = GridSearchCV(text_pipeline_logit, param_grid_logit, cv=3, n_jobs=-1)
  grid_logit.fit(text_train, y_train)
  print(grid_logit.best_params_, grid_logit.best_score_)
  plot_grid_scores(grid_logit, 'logisticregression__C')
  score = text_pipeline_logit.score(text_test, y_test)
  print(score)

  plt.show()
  return

def visualize_coefficients(classifier, feature_names, n_top_features=25):
  # get coefficients with large absolute values
  coef = classifier.coef_.ravel()
  positive_coefficients = np.argsort(coef)[-n_top_features:]
  negative_coefficients = np.argsort(coef)[:n_top_features]
  interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
  # plot them
  plt.figure(figsize=(15, 5))
  colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
  plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
  feature_names = np.array(feature_names)
  plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=60, ha="right")

def plot_grid_scores(grid, param_name):
  plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_train_score'],
  color='green', label='train')
  plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_test_score'],
  color='red', label='test')
  plt.legend()

def example_telekom():
  data = pd.read_csv(utils.PATH.COURSE_FILE('telecom_churn.csv')).drop('State', axis=1)
  data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
  data['Voice mail plan'] = data['Voice mail plan'].map({'Yes': 1, 'No': 0})

  X = data.drop('Churn', axis=1).values
  y = data['Churn'].astype('int').values

  #ROC_curves(X, y)
  plot_learning_curve(X, y, degree=2, alpha=1)

  plt.show()
  return

def ROC_curves(X, y):
  alphas = np.logspace(-2, 0, 20)
  sgd_logit = SGDClassifier(loss='log', n_jobs=-1, random_state=17)
  logit_pipe = Pipeline([('scaler', StandardScaler()),
                         ('poly', PolynomialFeatures(degree=2)),
                         ('sgd_logit', sgd_logit)])
  val_train, val_test = validation_curve(logit_pipe, X, y, 'sgd_logit__alpha', alphas, cv=5, scoring='roc_auc')

  plot_with_err(alphas, val_train, label='training scores')
  plot_with_err(alphas, val_test, label='validation scores')
  plt.xlabel(r'$\alpha$')
  plt.ylabel('ROC AUC')
  plt.legend()

def plot_with_err(x, data, **kwargs):
  mu, std = data.mean(1), data.std(1)
  lines = plt.plot(x, mu, '-', **kwargs)
  plt.fill_between(x, mu - std, mu + std, edgecolor='none',
  facecolor=lines[0].get_color(), alpha=0.2)

def plot_learning_curve(X, y, degree=2, alpha=0.01):
  train_sizes = np.linspace(0.05, 1, 20)
  logit_pipe = Pipeline([('scaler', StandardScaler()),
                         ('poly', PolynomialFeatures(degree=degree)),
                         ('sgd_logit', SGDClassifier(n_jobs=-1, random_state=17, alpha=alpha))])
  N_train, val_train, val_test = learning_curve(logit_pipe, X, y, train_sizes=train_sizes, cv=5, scoring='roc_auc')

  plot_with_err(N_train, val_train, label='training scores')
  plot_with_err(N_train, val_test, label='validation scores')
  plt.xlabel('Training Set Size')
  plt.ylabel('AUC')
  plt.legend()

def homework():

  sample_file = utils.PATH.COURSE_FILE('stackoverflow_sample_125k.tsv', dir='stackoverflow')
  tags_file = utils.PATH.COURSE_FILE('top10_tags.tsv', dir='stackoverflow')

  top_tags = set()
  with open(tags_file, 'r') as f:
    for line in f:
      top_tags.add(line.strip())
  print(top_tags)

  model = LogRegressor(top_tags)
  top_n_train = 100000 # 100000
  total = 125000       # 125000
  slice = 90000        # 90000
  trunc_vocab = 10000  # 10000

  jaccard_mean = model.iterate_file(sample_file, top_n_train=top_n_train, total=total)
  print('Mean of the loss function on the last 10k train samples: %0.2f' % np.mean(model._loss[slice:top_n_train]))
  print('Jaccard on test: %0.2f' % jaccard_mean)

  #plt.plot(pd.Series(model._loss[:-25000]).rolling(10000).mean());

  model._vocab_inv = dict([(v, k) for (k, v) in model._vocab.items()])
  for tag in model._tags:
    print(tag, ':', ', '.join([model._vocab_inv[k] for (k, v) in
                               sorted(model._w[tag].items(),
                                      key=lambda t: t[1],
                                      reverse=True)[:5]]))

  model.filter_vocab(n=trunc_vocab)
  jaccard_mean = model.iterate_file(sample_file, update_vocab=False, learning_rate=0.01, top_n_train=top_n_train, total=total)
  print('Jaccard on test: %0.2f' % jaccard_mean)

  sentence = ("I want to improve my coding skills, so I have planned write " +
              "a Mobile Application.need to choose between Apple's iOS or Google's Android." +
              " my background: I have done basic programming in .Net,C/C++,Python and PHP " +
              "in college, so got OOP concepts covered. about my skill level, I just know " +
              "concepts and basic syntax. But can't write complex applications, if asked :(" +
              " So decided to hone my skills, And I wanted to know which is easier to " +
              "learn for a programming n00b. A) iOS which uses Objective C B) Android " +
              "which uses Java. I want to decide based on difficulty level").lower().replace(',', '')

  #print(sorted(model.predict_proba(sentence).items(), key=lambda t: t[1], reverse=True))
  y_pred = model.predict_proba(sentence)

  plt.plot(pd.Series(model._loss[:-25000]).rolling(10000).mean());

  plt.show()
  return

class LogRegressor():
  def __init__(self, tags):
    # словарь который содержит мапинг слов предложений и тегов в индексы (для экономии памяти)
    # пример: self._vocab['exception'] = 17 означает что у слова exception индекс равен 17
    self._vocab = {}

    # параметры модели: веса
    #   для каждого класса/тега нам необходимо хранить собственный вектор весов
    #   по умолчанию у нас все веса будут равны нулю
    #   мы заранее не знаем сколько весов нам понадобится
    #   поэтому для каждого класса мы создаем словарь изменяемого размера со значением по умолчанию 0
    # пример: self._w['java'][self._vocab['exception']] содержит вес для слова exception тега java
    self._w = dict([(t, defaultdict(int)) for t in tags])

    # параметры модели: смещения или вес w_0
    self._b = dict([(t, 0) for t in tags])

    self._tags = set(tags)



  def iterate_file1(self,
                    fname,
                    top_n_train=100000,
                    total=125000,
                    learning_rate=0.1,
                    tolerance=1e-16,
                    lmbda=0.01):

    self._loss = []
    n = 0
    self.jaccar = []
    with open(fname, 'r') as f:

        for line in tqdm_notebook(f, total=total, mininterval=1):
            pair = line.strip().split('\t')
            if len(pair) != 2:
                continue
            sentence, tags = pair
            sentence = sentence.split(' ')
            tags = set(tags.split(' '))

            sample_loss = 0

            if n>0 and n%1000==0:
              print(n, self._loss[n-1])

            predict = []

            for tag in self._tags:
                y = int(tag in tags)
                z = self._b[tag]

                for word in sentence:
                    if n >= top_n_train and word not in self._vocab:
                        continue
                    if word not in self._vocab:
                        self._vocab[word] = len(self._vocab)
                    z +=  self._w[tag][self._vocab[word]]

                if z > 70:sigma = 1.0-tolerance
                elif z<-70: sigma = tolerance
                else: sigma = 1.0/(1.0+np.exp(-z))
                sigma = np.clip(sigma, tolerance, 1.0-tolerance)
                sample_loss += -(y*np.log(sigma) + (1-y)*np.log(1-sigma))

                if n < top_n_train:
                    dLdw = (sigma-y)

                    vocabulary = []
                    for word in sentence:
                        reg = 0
                        if word not in vocabulary:
                            reg = -lmbda*self._w[tag][self._vocab[word]]
                            vocabulary.append( word )
                        self._w[tag][self._vocab[word]] += -learning_rate*dLdw + reg

                    self._b[tag] += -learning_rate*dLdw

                if (n >= top_n_train) and (sigma>0.9):
                    predict.append(tag)

            self._loss.append(sample_loss)
            if n>= top_n_train:
                predict = set(predict)
                self.jaccar.append(len(predict & tags)/len(predict | tags))
            n += 1
    return np.mean(self.jaccar)



  def predict_proba(self, sentence, tolerance=1e-16):
    sentence = sentence.split(' ')
    y_pred = {}

    for tag in self._tags:
      z = self._b[tag]

      for word in sentence:
        if word in self._vocab:
          z += self._w[tag][self._vocab[word]]

      sigma = round(self.calc_sigma(z, tolerance), 2)
      y_pred[tag] = sigma
      print(tag, sigma)

    return y_pred

  def calc_sigma(self, z, tolerance=1e-16):
    if z<-70:  sigma = tolerance
    elif z>70: sigma = 1.0-tolerance
    else:
      sigma = 1.0/(1.0 + math.exp(-z))
      sigma = np.clip(sigma, tolerance, 1 - tolerance)
    return sigma

  """
  Один прогон по датасету
  Параметры
  ----------
  fname:         имя файла с данными
  top_n_train:   первые top_n_train строк будут использоваться для обучения, остальные для тестирования
  total:         информация о количестве строк в файле для вывода прогресс бара
  learning_rate: скорость обучения для градиентного спуска
  tolerance:     используем для ограничения значений аргумента логарифмов
  """
  def iterate_file(self,
                   fname,
                   top_n_train=100000,
                   total=125000,
                   learning_rate=0.1,
                   tolerance=1e-16,
                   lmbda=0.0002,
                   gamma=0.1,
                   update_vocab=True):

    self._loss = []
    n = 0
    jaccard_coeff = []
    self._freqs = {}

    # откроем файл
    with open(fname, 'r') as f:

      for line in tqdm_notebook(f, total=total, mininterval=1):
        if n>=total:
          break

        pair = line.strip().split('\t')
        if len(pair) != 2:
          continue
        sentence, tags = pair
        sentence = sentence.split(' ')
        tags = set(tags.split(' '))

        ## ------ TEST ------
        #learning_rate = 1
        #if n==0:
        #  self._tags = ['one', 'two', 'three']
        #  self._w = dict([(t, defaultdict(int)) for t in self._tags])
        #  self._b = dict([(t, 0) for t in self._tags])
        #if n==0: sentence, tags = ['A', 'B', 'A'], ['one', 'three']
        #if n==1: sentence, tags = ['B', 'C'], ['one']
        #if n==2: sentence, tags = ['C', 'A', 'B', 'B'], ['two', 'three']
        #if n>2: return

        for word in sentence:
          if word in self._freqs:
            self._freqs[word] += 1
          else:
            self._freqs[word] = 1

        sample_loss = 0

        y_pred = []
        y_act  = []

        if n>0 and n%1000==0:
          print(n, self._loss[n-1])

        for tag in self._tags:
          y = int(tag in tags)
          z = self._b[tag]

          for word in sentence:
            if n >= top_n_train and word not in self._vocab:
              continue
            if word not in self._vocab and update_vocab:
              self._vocab[word] = len(self._vocab)
            if word in self._vocab:
              z += self._w[tag][self._vocab[word]]

          sigma = self.calc_sigma(z, tolerance)

          if n >= top_n_train:
            y_act.append(y)
            y_pred.append(sigma)

          sample_loss += -(y*math.log(sigma) + (1 - y)*math.log(1 - sigma))

          if n < top_n_train:
            dLdw = (y - sigma)

            word_set = []
            for word in sentence:
              if word not in self._vocab:
                continue;
              reg = 0.0
              if word not in word_set:
                wki = self._w[tag][self._vocab[word]]
                reg = lmbda*(gamma*2*wki + (1-gamma)*np.sign(wki))
                word_set.append(word)
              self._w[tag][self._vocab[word]] -= -learning_rate*dLdw + reg

            self._b[tag] -= -learning_rate*dLdw

        if n >= top_n_train:
          jaccard_coeff.append(self.jaccard(y_act, y_pred))

        self._loss.append(sample_loss)

        n += 1

    return np.mean(jaccard_coeff)

  def filter_vocab(self, n=10000):
    print('begin filter vocab')
    s = [x[0] for x in sorted(self._freqs.items(), key=lambda t: t[1], reverse=True)[:n]]
    self._vocab = {k:v for k,v in self._vocab.items() if k in s}
    print('end filter vocab')
    print('vocab length', len(self._vocab))

  def jaccard(self, y1, y2, level=0.9):
    n = len(y1)
    a = 0
    b = 0
    for i in range(n):
      t1 = 0 if y1[i]<=level else 1
      t2 = 0 if y2[i]<=level else 1
      if t1==1 and t2==1:
        a += 1
      if t1==1 or t2==1:
        b += 1

    return a/b


