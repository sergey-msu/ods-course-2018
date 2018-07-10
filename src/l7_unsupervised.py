import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
import utils
from collections import Counter
from sklearn import decomposition
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist, pdist
from scipy.cluster import hierarchy

def header(): return 'LECTURE 7: Unsupervised Learning - PCA and Clustering https://habrahabr.ru/company/ods/blog/325654/';

def run():

  #pca()
  #clustering()
  homework()

  return

def pca():

  #iris()
  mnist()

  return

def iris():
  iris = datasets.load_iris()
  X = iris.data
  y = iris.target

  fig = plt.figure(1, figsize=(6, 5))
  plt.clf()
  ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

  plt.cla()

  for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

  y_clr = np.choose(y, [1, 2, 0]).astype(np.float)
  ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_clr)

  ax.w_xaxis.set_ticklabels([])
  ax.w_yaxis.set_ticklabels([])
  ax.w_zaxis.set_ticklabels([])
  plt.show()

  accur = classify(X, y)
  print('Accuracy: {0}'.format(accur))

  pca = decomposition.PCA(n_components=2)
  X_centered = X - X.mean(axis=0)
  pca.fit(X_centered)
  X_pca = pca.transform(X_centered)

  plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
  plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
  plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
  plt.legend(loc=0);
  plt.show()

  accur = classify(X_pca, y)
  print('Accuracy after PCA: {0}'.format(accur))

  for i, component in enumerate(pca.components_):
    print('{} component {}% of initial variance'.format(i+1, round(100*pca.explained_variance_ratio_[i], 2)) )
    print(' + '.join('%.3f x %s' % (value, name)
                     for value, name in zip(component, iris.feature_names)))

  return

def classify(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

  dt = DecisionTreeClassifier(max_depth=2, random_state=42)
  dt.fit(X_train, y_train)
  pred = dt.predict_proba(X_test)
  return accuracy_score(y_test, pred.argmax(axis=1))

def mnist():
  digits = datasets.load_digits()
  X = digits.data
  y = digits.target

  plt.figure(figsize=(16, 6))
  for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i,:].reshape([8,8]));
  plt.show()

  pca = decomposition.PCA(n_components=2)
  X_reduced = pca.fit_transform(X)

  print('Projecting {}-dimensional data to 2D'.format(X.shape[1]))

  plt.figure(figsize=(12, 10))
  plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
              edgecolor='none', alpha=0.7, s=40,
              cmap=plt.cm.get_cmap('nipy_spectral', 10))
  plt.colorbar()
  plt.title('MNIST. PCA projection')
  plt.show()

  #tsne = TSNE(random_state=17)
  #X_tsne = tsne.fit_transform(X)
  #
  #plt.figure(figsize=(12,10))
  #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
  #            edgecolor='none', alpha=0.7, s=40,
  #            cmap=plt.cm.get_cmap('nipy_spectral', 10))
  #plt.colorbar()
  #plt.title('MNIST. t-SNE projection')
  #plt.show()

  pca = decomposition.PCA().fit(X)

  plt.figure(figsize=(10, 7))
  plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
  plt.xlabel('Number of components')
  plt.ylabel('Total explained variance')
  plt.xlim(0, 63)
  plt.yticks(np.arange(0, 1.1, 0.1))
  plt.axvline(21, c='b')
  plt.axhline(0.9, c='r')
  plt.show()

  return

def clustering():

  #k_means_simple()
  #k_means_clustnum()
  #affinity_propagation()
  #spectral_clustering()
  #agglomerative_clustering()
  clustering_metrics()

  return

def get_3_clusers():
  X = np.zeros([150, 2])

  np.random.seed(seed=42)

  X[:50, 0] = np.random.normal(loc=0.0, scale=0.3, size=50)
  X[:50, 1] = np.random.normal(loc=0.0, scale=0.3, size=50)

  X[50:100, 0] = np.random.normal(loc=2.0, scale=0.5, size=50)
  X[50:100, 1] = np.random.normal(loc=-1.0, scale=0.2, size=50)

  X[100:150, 0] = np.random.normal(loc=-1.0, scale=0.2, size=50)
  X[100:150, 1] = np.random.normal(loc=2.0, scale=0.5, size=50)

  return X

def k_means_simple():

  X = get_3_clusers()

  plt.figure(figsize=(5, 5))
  plt.plot(X[:, 0], X[:, 1], 'bo')
  plt.show()

  np.random.seed(seed=42)
  centroids = np.random.normal(loc=0.0, scale=1.0, size=6)
  centroids = centroids.reshape([3, 2])
  cent_history = []
  cent_history.append(centroids)

  for i in range(3):
    distances = cdist(X, centroids)
    labels = distances.argmin(axis=1)

    centroids = centroids.copy()
    centroids[0, :] = np.mean(X[labels==0, :], axis=0)
    centroids[1, :] = np.mean(X[labels==1, :], axis=0)
    centroids[2, :] = np.mean(X[labels==2, :], axis=0)

    cent_history.append(centroids)

  plt.figure(figsize=(8, 8))
  for i in range(4):
    distances = cdist(X, cent_history[i])
    labels = distances.argmin(axis=1)

    plt.subplot(2, 2, i+1)
    plt.plot(X[labels==0, 0], X[labels==0, 1], 'bo', label='clister #1')
    plt.plot(X[labels==1, 0], X[labels==1, 1], 'mo', label='clister #2')
    plt.plot(X[labels==2, 0], X[labels==2, 1], 'co', label='clister #3')
    plt.plot(cent_history[i][:, 0], cent_history[i][:, 1], 'rX')
    plt.legend(loc=0)
    plt.title('Step {0}'.format(i+1))
  plt.show()

  return

def k_means_clustnum():
  X = get_3_clusers()
  inertia = []

  for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))

  plt.plot(range(1, 8), inertia, marker='s')
  plt.xlabel('$k$')
  plt.ylabel('$J(C_k)$')
  plt.show()

  return

def affinity_propagation():

  return

def spectral_clustering():

  return

def agglomerative_clustering():
  X = get_3_clusers()

  distance_mat = pdist(X)
  Z = hierarchy.linkage(distance_mat, 'single')
  plt.figure(figsize=(10, 5))
  dh = hierarchy.dendrogram(Z, color_threshold=0.5)
  plt.show()

  return

def clustering_metrics():
  data = datasets.load_digits()
  X, y = data.data, data.target

  algorithms = []
  algorithms.append(KMeans(n_clusters=10, random_state=1))
  algorithms.append(AffinityPropagation())
  algorithms.append(SpectralClustering(n_clusters=10, random_state=1, affinity='nearest_neighbors'))
  algorithms.append(AgglomerativeClustering(n_clusters=10))

  data = []

  for algo in algorithms:
    algo.fit(X)
    data.append(({
      'ARI': metrics.adjusted_rand_score(y, algo.labels_),
      'AMI': metrics.adjusted_mutual_info_score(y, algo.labels_),
      'Homogenity': metrics.homogeneity_score(y, algo.labels_),
      'Completeness': metrics.completeness_score(y, algo.labels_),
      'V-measure': metrics.v_measure_score(y, algo.labels_),
      'Silhouette': metrics.silhouette_score(X, algo.labels_)
      }))

  result = pd.DataFrame(data=data,
                        columns=['ARI', 'AMI', 'Homogenity', 'Completeness', 'V-measure', 'Silhouette'],
                        index=['K-means', 'Affinity', 'Spectral', 'Agglomerative'])

  print(result)

  return

def homework():
  plt.style.use(['seaborn-darkgrid'])
  plt.rcParams['figure.figsize'] = (12, 9)
  plt.rcParams['font.family'] = 'DejaVu Sans'
  RANDOM_STATE = 1

  X_train = np.loadtxt(utils.PATH.COURSE_FILE('samsung_train.txt', 'samsung_HAR'))
  y_train = np.loadtxt(utils.PATH.COURSE_FILE('samsung_train_labels.txt', 'samsung_HAR')).astype(int)
  print(X_train.shape)
  print(y_train.shape)

  X_test = np.loadtxt(utils.PATH.COURSE_FILE('samsung_test.txt', 'samsung_HAR'))
  y_test = np.loadtxt(utils.PATH.COURSE_FILE('samsung_test_labels.txt', 'samsung_HAR')).astype(int)
  print(X_test.shape)
  print(y_test.shape)

  X = np.concatenate([X_train, X_test], axis=0)
  y = np.concatenate([y_train, y_test], axis=0)

  #labels = np.unique(y)
  #n_classes = labels.size
  #print(labels)
  labels = dict({
    1: 'ходьба',
    2: 'подъем вверх по лестнице',
    3: 'спуск по лестнице',
    4: 'сидение',
    5: 'стояние',
    6: 'лежание'
    })
  #
  #scaler = StandardScaler()
  #X_scaled = scaler.fit_transform(X)
  #
  #pca = decomposition.PCA().fit(X_scaled)
  #
  ## Q1
  #print('--------------------- Q1')

  #plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
  #plt.xlabel('Number of components')
  #plt.ylabel('Total explained variance')
  #plt.xlim(0, 100)
  #plt.yticks(np.arange(0, 1.1, 0.1))
  #plt.axvline(63, c='b')
  #plt.axhline(0.9, c='r')
  #plt.show()
  #
  ## Q2
  #print('--------------------- Q2')

  #print(pca.explained_variance_ratio_[0])
  #
  ##Q3
  #print('--------------------- Q3')

  #X_reduct = pca.transform(X_scaled)
  #print(X_reduct.shape)
  #
  #X_2D = X_reduct[:,:2]
  #
  #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
  #
  #colors = ['b', 'c', 'y', 'm', 'r', 'g']
  #xs = []
  #for i in range(6):
  #  x = axes[0].scatter(X_2D[y==i+1][:,0], X_2D[y==i+1][:,1], s=5, color=colors[i])
  #  xs.append(x)
  #axes[0].legend(xs, labels.values())
  #
  ##Q4
  #print('--------------------- Q4')

  #kmeans = KMeans(n_clusters=6, n_init=100, random_state=RANDOM_STATE).fit(X_reduct[:, :63])
  #print(kmeans.cluster_centers_[:, :2])
  #
  #colors = ['b', 'c', 'y', 'm', 'r', 'g']
  #xs = []
  #cluster_labels = kmeans.predict(X_reduct[:, :63])
  #for i in range(6):
  #  x_label = X_reduct[cluster_labels==i]
  #  x = axes[1].scatter(x_label[:,0], x_label[:,1], s=5, color=colors[i])
  #  xs.append(x)
  #for i in range(6):
  #  c_c = kmeans.cluster_centers_[i, :]
  #  axes[1].scatter(c_c[0], c_c[1], s=40, edgecolor='black', linewidth='1', color=colors[i])
  #plt.show()
  #
  #tab = pd.crosstab(y, cluster_labels, margins=True, normalize='index')
  #tab.index = ['ходьба', 'подъем вверх по лестнице',
  #             'спуск по лестнице', 'сидение', 'стояние', 'лежание', 'все']
  #tab.columns = ['cluster' + str(i + 1) for i in range(6)] #+ ['все']
  #print(tab)
  #
  ## Q5
  #print('--------------------- Q5')

  #inertia = []
  #for k in range(1, n_classes+1):
  #  kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(X_reduct[:, :63])
  #  inertia.append(np.sqrt(kmeans.inertia_))
  #  print('{0}-th inertia calculated'.format(k))
  #plt.plot(range(1, n_classes+1), inertia, marker='s')
  #plt.xlabel('$k$')
  #plt.ylabel('$J(C_k)$')
  #
  ## Q6
  #print('--------------------- Q6')

  #ad = AgglomerativeClustering(n_clusters=n_classes, linkage='ward').fit(X_reduct[:, :63])
  #kmeans = KMeans(n_clusters=6, n_init=100, random_state=RANDOM_STATE).fit(X_reduct[:, :63])
  #ad_score = metrics.adjusted_rand_score(y, ad.labels_)
  #kmeans_score = metrics.adjusted_rand_score(y, kmeans.labels_)
  #
  #print('ad_score:', ad_score)
  #print('kmeans_score:', kmeans_score)
  #
  #plt.show()

  # Q7
  print('--------------------- Q7')

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled  = scaler.transform(X_test)

  svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
  svc = GridSearchCV(LinearSVC(random_state=RANDOM_STATE), svc_params, cv=3, n_jobs=-1)
  svc.fit(X_train_scaled, y_train)

  print(svc.best_score_)
  print(svc.best_params_)

  # Q8
  print('--------------------- Q8')

  y_pred = svc.predict(X_test_scaled)
  tab = pd.crosstab(y_test, y_pred, margins=True)
  tab.index = ['ходьба', 'подъем вверх по лестнице', 'спуск по лестнице',
               'сидение', 'стояние', 'лежание', 'все']
  tab.columns = tab.index
  print(tab)

  precision_and_recall(y_test, y_pred, labels)

  # Q9
  print('--------------------- Q9')

  pca = decomposition.PCA().fit(X_train_scaled)
  X_reduct_scaled = pca.transform(X_train_scaled)[:, :63]
  svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
  svc = GridSearchCV(LinearSVC(random_state=RANDOM_STATE), svc_params, cv=3, n_jobs=-1)
  print('-----fit')
  svc.fit(X_reduct_scaled, y_train)

  print(svc.best_score_)
  print(svc.best_params_)

  return

def precision_and_recall(y_test, y_pred, labels):
  for i in range(1, 7):
    test_true  = set(np.where(y_test==i)[0])
    test_false = set(np.where(y_test!=i)[0])
    pred_true  = set(np.where(y_pred==i)[0])
    pred_false = set(np.where(y_pred!=i)[0])

    tp = len(y_pred[list(test_true.intersection(pred_true))])
    fp = len(y_pred[list(test_false.intersection(pred_true))])
    tn = len(y_pred[list(test_false.intersection(pred_false))])
    fn = len(y_pred[list(test_true.intersection(pred_false))])
    precision = tp/(tp + fp)
    recall    = tp/(tp + fn)
    print('{0}: precision={1}, recall={2}'.format(labels[i], round(precision, 2), round(recall, 2)))
  return