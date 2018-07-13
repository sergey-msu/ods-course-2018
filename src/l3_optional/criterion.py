from l3_optional.utils import *
import numpy as np

class CriterionBase:
  def calc(self, y):
    pass

  @property
  def is_regression(self):
    pass


class GiniCriterion(CriterionBase):
  @property
  def is_regression(self):
    return False

  def calc(self, y):
    fs = freqs(y)
    return 1 - np.sum([fs[cls]**2 for cls in fs])


class EntropyCriterion(CriterionBase):
  @property
  def is_regression(self):
    return False

  def calc(self, y):
    fs = freqs(y)
    return -np.sum(fs[cls]*np.log2(fs[cls]) for cls in fs)


class VarianceCriterion(CriterionBase):
  @property
  def is_regression(self):
    return True

  def calc(self, y):
    return y.var()


class MadMedianCriterion(CriterionBase):
  @property
  def is_regression(self):
    return True

  def calc(self, y):
    m = y.median()
    return 1/len(y)*np.sum([np.abs(yi - m) for yi in y])


# shortcuts

gini = GiniCriterion()
entropy = EntropyCriterion()
variance = VarianceCriterion()
madmedian = MadMedianCriterion()

def criterion_by_name(name):
  if name=='gini':
    return gini
  if name=='entropy':
    return entropy
  if name=='variance':
    return variance
  if name=='madmedian':
    return madmedian
  raise Exception('Unknown criterion name')