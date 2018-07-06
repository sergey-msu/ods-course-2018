from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from l3_optional.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


def header(): return 'LECTURE 3: Classification https://habrahabr.ru/company/ods/blog/322534/'

def run():

  #informativity()
  #train_data, train_labels = syntetic()
  #decision_tree(train_data, train_labels)
  #regression_tree()
  #telekom_example()
  #complex_example()
  #mnist_example()

  #homework()
  homework_optional()

  return

def informativity():
  plt.rcParams['figure.figsize'] = (6,4)

  xx = np.linspace(0, 1, 50)
  plt.plot(xx, [2*x*(1 - x) for x in xx], label='gini')
  plt.plot(xx, [4*x*(1 - x) for x in xx], label='2*gini')
  plt.plot(xx, [-x*np.log2(x) - (1 - x)*np.log2(1 - x) for x in xx], label='entropy')
  plt.plot(xx, [1 - max(x, 1 - x) for x in xx], label='missclass')
  plt.plot(xx, [2 - 2*max(x, 1 - x) for x in xx], label='2*missclass')
  plt.xlabel('p+')
  plt.ylabel('criterion')
  plt.title('Criterions as fiunctions of p - binary classification)')
  plt.legend()
  plt.show()

  return

def syntetic():
  np.seed = 7

  # 1st class
  train_data = np.random.normal(size=(100, 2))
  train_labels = np.zeros(100)

  # 2nd class
  train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
  train_labels = np.r_[train_labels, np.ones(100)]

  plt.rcParams['figure.figsize'] = (10, 8)
  plt.scatter(train_data[:, 0], train_data[:, 1],
              c=train_labels, s=100,
              cmap='autumn', edgecolors='black', linewidth=1.5)
  plt.plot(range(-2, 5), range(4, -3, -1))
  plt.show()

  return train_data, train_labels

def get_grid(data, eps=0.01):
  x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
  y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
  return np.meshgrid(np.arange(x_min, x_max, eps), np.arange(y_min, y_max, eps))

def decision_tree(train_data, train_labels):

  clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
  clf_tree.fit(train_data, train_labels)

  xx, yy = get_grid(train_data)
  predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
  plt.pcolormesh(xx, yy, predicted, cmap='autumn')
  plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels,
              s=100, cmap='autumn', edgecolors='black', linewidth=1.5)
  plt.show()

  # show tree

  export_graphviz(clf_tree, feature_names=['x1', 'x2'], out_file='small_tree.dot', filled=True)





  return

def regression_tree():
  n_train = 50
  n_test  = 1000
  noise = 0.1

  def f(x):
    x = x.ravel()
    return np.exp(-x**2) + 1.5*np.exp(-(x - 2)**2)

  def generate(n_samples, noise):
    X = np.random.rand(n_samples)*10 - 5
    X = np.sort(X).ravel()
    y = f(X) + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))
    return X, y

  X_train, y_train = generate(n_samples=n_train, noise=noise)
  X_test,  y_test  = generate(n_samples=n_test, noise=noise)

  reg_tree = DecisionTreeRegressor(max_depth=5, random_state=17)
  reg_tree.fit(X_train, y_train)

  reg_tree_pred = reg_tree.predict(X_test)

  plt.figure(figsize=(10, 6))
  plt.plot(X_test, f(X_test), 'b')
  plt.scatter(X_train, y_train, c='b', s=20)
  plt.plot(X_test, reg_tree_pred, 'g', lw=2)
  plt.xlim([-5, 5])
  plt.title('Decision tree regressor, MSE = %.2f' % np.sum((y_test - reg_tree_pred)**2))
  plt.show()

  return

def telekom_example():

  # prepare data

  df = pd.read_csv(utils.PATH.COURSE_FILE('telecom_churn.csv'))
  df['International plan'] = pd.factorize(df['International plan'])[0]
  df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0]
  df['Churn'] = df['Churn'].astype('int')

  states = df['State']
  y = df['Churn']

  df.drop(['State', 'Churn'], axis=1, inplace=True)

  # prepare sample
  X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3, random_state=17)

  # KNN model

  knn = KNeighborsClassifier(n_neighbors=10)
  knn.fit(X_train, y_train)

  knn_pred = knn.predict(X_holdout)
  score = accuracy_score(y_holdout, knn_pred)
  print(score)

     # 5-times cross validation

  knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
  knn_params = {'knn__n_neighbors': range(1, 10)}
  knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)
  knn_grid.fit(X_train, y_train)

  best_params = knn_grid.best_params_
  best_score  = knn_grid.best_score_
  print(best_params, best_score)

  knn_grid_predict = knn_grid.predict(X_holdout)
  score = accuracy_score(y_holdout, knn_grid_predict)
  print(score)

  # Decision Tree model

  tree = DecisionTreeClassifier(max_depth=5, random_state=17)
  tree.fit(X_train, y_train)

  tree_pred = tree.predict(X_holdout)
  score = accuracy_score(y_holdout, tree_pred)
  print(score)

     # 5-times cross validation

  tree_params = { 'max_depth': range(1, 11), 'max_features': range(4, 19) }
  tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
  tree_grid.fit(X_train, y_train)

  best_params = tree_grid.best_params_
  best_score  = tree_grid.best_score_
  print(best_params, best_score)

  tree_grid_predict = tree_grid.predict(X_holdout)
  score = accuracy_score(y_holdout, tree_grid_predict)
  print(score)

  # Random Forest model

  forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=17)
  score = np.mean(cross_val_score(forest, X_train, y_train, cv=5))
  print(score)

  forest_params = {'max_depth': range(1, 11), 'max_features': range(4, 19)}
  forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1, verbose=True)
  forest_grid.fit(X_train, y_train)

  best_params = forest_grid.best_params_
  best_score  = forest_grid.best_score_
  print(best_params, best_score)

  forest_grid_predict = forest_grid.predict(X_holdout)
  score = accuracy_score(y_holdout, forest_grid_predict)
  print(score)

  return

def complex_example():

  def form_linearly_separable_data(n=500, x1_min=0, x1_max=30, x2_min=0, x2_max=30):
    data, target = [], []
    for i in range(n):
      x1, x2 = np.random.randint(x1_min, x1_max), np.random.randint(x2_min, x2_max)
      if np.abs(x1 - x2) > 0.5:
        data.append([x1, x2])
        target.append(np.sign(x1 - x2))
    return np.array(data), np.array(target)

  # generate data
  X, y = form_linearly_separable_data()
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolors='black')
  plt.show()

  # Decision Tree model
  tree = DecisionTreeClassifier(random_state=17).fit(X, y)

  xx, yy = get_grid(X, eps=0.05)
  predicted = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
  plt.pcolormesh(xx, yy, predicted, cmap='autumn')
  plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn', edgecolors='black', linewidth=1.5)
  plt.title('Easy task. Complex decision tree')
  plt.show()

  # KNN model
  knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)

  xx, yy = get_grid(X, eps=0.05)
  predicted = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
  plt.pcolormesh(xx, yy, predicted, cmap='autumn')
  plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='autumn', edgecolors='black', linewidth=1.5)
  plt.title('Easy task. Not bad KNN')
  plt.show()

  return

def mnist_example():
  data = load_digits()
  X, y = data.data, data.target

  print(X[0, :].reshape([8, 8]))

  f, axes = plt.subplots(1, 4, sharey=True, figsize=(16, 6))
  for i in range(4):
    axes[i].imshow(X[i, :].reshape([8, 8]))
  plt.show()

  X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)

  # Decision Tree model

  tree = DecisionTreeClassifier(max_depth=5, random_state=17)
  tree.fit(X_train, y_train)
  tree_pred = tree.predict(X_holdout)
  score = accuracy_score(y_holdout, tree_pred)
  print('Tree: ', score)

  tree_params = { 'max_depth': [1, 2, 3, 5, 10, 20, 25, 30, 40, 50, 64],
                  'max_features': [1, 2, 3, 5, 10, 20, 30, 50, 64] }
  tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
  tree_grid.fit(X_train, y_train)
  print(tree_grid.best_params_, tree_grid.best_score_)

  # KNN model

  knn = KNeighborsClassifier(n_neighbors=10)
  knn.fit(X_train, y_train)
  knn_pred = knn.predict(X_holdout)
  score = accuracy_score(y_holdout, knn_pred)
  print('KNN: ', score)

  score = cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv=5)
  print(np.mean(score))

  # Random Forest

  score = cross_val_score(RandomForestClassifier(random_state=17), X_train, y_train, cv=5)
  print(np.mean(score))


  return

def homework():

  X = np.linspace(-2, 2, 7)
  y = X**3
  T = np.linspace(-1.9, 1.9, 39*4)
  P = np.zeros_like(T)

  def simple_regression():
    plt.scatter(X, y)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    idx1 = np.where(X < 0)
    X1 = X[idx1]
    y1 = y[idx1]
    y1_mean = np.mean(y1)

    idx2 = np.where(X >= 0)
    X2 = X[idx2]
    y2 = y[idx2]
    y2_mean = np.mean(y2)

    y_pred = np.concatenate(([y1_mean]*len(idx1[0]), [y2_mean]*len(idx2[0])))

    plt.scatter(X, y_pred)
    return

  def regression_var_criterion():

    def regression(X, y, t):
      yl = y[X < t]
      yr = y[X >= t]
      q = y.var() - (yl.size/y.size)*yl.var() - (yr.size/y.size)*yr.var()
      return q

    Q = [regression(X, y, t) for t in T]

    plt.plot(T, Q)
    return

  def more_deep_tree():

    p11 = lambda x: (x<-1.5)
    p12 = lambda x: (x>=-1.5) & (x<0)
    p21 = lambda x: (x>=0) & (x<1.5)
    p22 = lambda x: (x>=1.5)

    y11 = np.mean(y[p11(X)])
    y12 = np.mean(y[p12(X)])
    y21 = np.mean(y[p21(X)])
    y22 = np.mean(y[p22(X)])

    idx = -1
    for t in T:
      idx += 1
      if p11(t): P[idx] = y11
      elif p12(t): P[idx] = y12
      elif p21(t): P[idx] = y21
      elif p22(t): P[idx] = y22

    plt.plot(T, P)

    return

  def heart_decease():
    df = pd.read_csv(utils.PATH.COURSE_FILE('mlbootcamp5_train.csv'), index_col='id', sep=';')

    df['age_years'] = (df['age']/365).astype('int')
    df_ch = pd.get_dummies(df['cholesterol'], prefix='cholesterol')
    df_gl = pd.get_dummies(df['gluc'], prefix='gluc')

    df = pd.concat([df, df_ch, df_gl], axis = 1).drop(['age', 'cholesterol', 'gluc'], axis=1)

    y = df['cardio'].as_matrix()
    X = df.drop('cardio', axis=1).as_matrix()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17)

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_valid)
    acc1 = accuracy_score(y_valid, y_pred)
    print(acc1)

    export_graphviz(tree, out_file='homework_tree.dot', filled=True)

    tree_params = { 'max_depth': range(2, 10) }
    tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
    tree_grid.fit(X_train, y_train)
    print(tree_grid.best_params_, tree_grid.best_score_)
    y_pred = tree_grid.predict(X_valid)
    acc2 = accuracy_score(y_valid, y_pred)

    acc_inc = (acc2 - acc1)/acc1*100
    print('acc_inc', acc_inc)

    depths_scores = list(zip(*[(score[0]['max_depth'], score[1]) for score in tree_grid.grid_scores_]))
    plt.plot(depths_scores[0], depths_scores[1])

    df_new = pd.DataFrame()
    df_new['age_1']   = df['age_years'].apply(lambda x: 1 if ((x>=45) & (x<50)) else 0)
    df_new['age_2']   = df['age_years'].apply(lambda x: 1 if ((x>=50) & (x<55)) else 0)
    df_new['age_3']   = df['age_years'].apply(lambda x: 1 if ((x>=55) & (x<60)) else 0)
    df_new['age_4']   = df['age_years'].apply(lambda x: 1 if ((x>=60) & (x<65)) else 0)
    df_new['ap_hi_1'] = df['ap_hi'].apply(lambda x: 1 if ((x>=120) & (x<140)) else 0)
    df_new['ap_hi_2'] = df['ap_hi'].apply(lambda x: 1 if ((x>=140) & (x<160)) else 0)
    df_new['ap_hi_3'] = df['ap_hi'].apply(lambda x: 1 if ((x>=160) & (x<180)) else 0)
    df_new['chol_1']  = df['cholesterol_1']
    df_new['chol_2']  = df['cholesterol_2']
    df_new['chol_3']  = df['cholesterol_3']
    df_new['gender']  = df['gender'].map({1: 0, 2: 1})

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
    tree.fit(df_new, y)

    export_graphviz(tree, out_file='homework_tree_final.dot', filled=True)


    plt.show()

    return

  #simple_regression()
  #regression_var_criterion()
  #more_deep_tree()
  heart_decease()

  plt.show()

  return

def homework_optional():

  ## TEST
  #X_train = np.array([[0, 2],
  #                    [1, 1],
  #                    [1, 2],
  #                    [1, 3],
  #                    [2, 2],
  #                    [3, 0],
  #                    [0, 0],
  #                    [0, 3],
  #                    [1, 0],
  #                    [2, 0],
  #                    [2, 1],
  #                    [3, 1],
  #                    [3, 2],
  #                    [3, 3]])
  #y_train = np.expand_dims(np.array([1, 1, 1, 1, 1, 1,
  #                                   0, 0, 0, 0, 0, 0, 0, 0]), axis=1)
  #tree = DecisionTree()
  #tree.fit(X_train, y_train)
  #
  #X_test = np.array([[0.5, 2],
  #                   [1.5, 0],
  #                   [1.5, 2],
  #                   [1.5, 3],
  #                   [2.5, 1]])
  #y_test = np.expand_dims(np.array([1, 0, 1, 1, 0]), axis=1)

  # MNIST
  data = load_digits()
  X, y = data.data, data.target
  y = np.expand_dims(y, axis=1).astype(int)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

  ## IRIS
  #data = utils.DATA.IRIS()
  #X, y = data[:,:-1], data[:, -1]
  #X_train = np.concatenate((X[0:40],  X[50:90],  X[100:140]))
  #X_test  = np.concatenate((X[40:50], X[90:100], X[140:150]))
  #y_train = np.expand_dims(np.concatenate((y[0:40],  y[50:90],  y[100:140])), axis=1).astype(int)
  #y_test  = np.expand_dims(np.concatenate((y[40:50], y[90:100], y[140:150])), axis=1).astype(int)

  tree = DecisionTree(max_depth=10, criterion='gini')
  tree.fit(X_train, y_train)
  y_pred = tree.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  print(round(acc*100, 2), '%')

  tree = DecisionTreeClassifier(max_depth=10, criterion='gini')
  tree.fit(X_train, y_train)
  y_pred = tree.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  print(round(acc*100, 2), '%')

  #export_graphviz(tree, out_file='_tree.dot', filled=True)
  #
  #plt.scatter(X[0:40, 0],    X[0:40, 1])
  #plt.scatter(X[50:90, 0],   X[50:90, 1])
  #plt.scatter(X[100:140, 0], X[100:140, 1])
  #plt.show()

  return