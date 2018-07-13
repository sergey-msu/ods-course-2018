import numpy as np
from l3_optional.utils import *
from l3_optional.criterion import criterion_by_name
from l3_optional.node import InnerNode, LeafNode
from sklearn.base  import BaseEstimator

class DecisionTree(BaseEstimator):
  def __init__(self, max_depth=np.inf, min_samples_split=2, criterion='gini', debug=False):
    self.__max_depth = max(max_depth, 0)
    self.__min_samples_split = min_samples_split
    self.__criterion = criterion_by_name(criterion)
    self.__root = None

  def predict(self, X):
    if self.__root is None:
      raise Exception('Train your tree first')

    n = X.shape[0]
    y_pred = np.zeros([n])
    for i in range(n):
      y_pred[i] = self.__root.predict(X[i,:])

    return y_pred

  def predict_proba(self, X):
    # ???
    pass

  def fit(self, X, y):
    S = np.concatenate((X, y), axis=1)
    self.__root = self.__build_subtree(S, -1);
    return self

  def __build_subtree(self, S, depth):
    depth += 1

    if ((self.__max_depth == depth) or
        (S.shape[0]<=self.__min_samples_split)):
      return self.__build_leaf(S[:,-1])

    Sl, Sr, pred = self.__optimize_Q(S)

    if Sl.shape[0]==0 or Sr.shape[0]==0:
      return self.__build_leaf(S[:,-1])

    lnode = self.__build_subtree(Sl, depth)
    rnode = self.__build_subtree(Sr, depth)

    return InnerNode(pred, lnode, rnode)

  def __optimize_Q(self, S):
    j_opt  = -1
    t_opt  = -np.infty
    q_opt  = -np.infty
    Sl_opt = None
    Sr_opt = None

    for j in range(S.shape[1] - 1):
      q, t, Sl, Sr = self.__maximize_Q(S, j)
      if q>q_opt:
        q_opt = q
        j_opt = j
        t_opt = t
        Sl_opt = Sl
        Sr_opt = Sr

    return Sl_opt, Sr_opt, lambda x: x[j_opt] < t_opt

  def __maximize_Q(self, S, j):
    xjs = np.sort(np.unique(np.copy(S[:, j])))
    q_opt = -np.infty
    t_opt = -np.infty
    Sl_opt = None
    Sr_opt = None

    for t in xjs:
      Sl = S[np.where(S[:, j] >= t)]
      Sr = S[np.where(S[:, j] < t)]
      q = self.__Q(S, Sl, Sr)
      if q>q_opt:
        q_opt = q
        t_opt = t
        Sl_opt = Sl
        Sr_opt = Sr

    return q_opt, t_opt, Sl_opt, Sr_opt

  def __Q(self, S, Sl, Sr):
    n  = S.shape[0]
    nl = Sl.shape[0]
    nr = Sr.shape[0]

    d = self.__criterion.calc(S[:,-1])
    dl = nl/n*self.__criterion.calc(Sl[:,-1]) if nl>0 else 0
    dr = nr/n*self.__criterion.calc(Sr[:,-1]) if nr>0 else 0

    return d - dl - dr

  def __build_leaf(self, y):
    if self.__criterion.is_regression:
      value = y.mean()
      return LeafNode(value)
    else:
      hs = hist(y)
      h_max = max(hs.values())
      values = [k for k,v in hs.items() if v==h_max]
      return LeafNode(values[0])