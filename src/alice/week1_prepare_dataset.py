from __future__ import division, print_function
import os
import pickle
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
from tqdm import tqdm_notebook
from glob import glob
from scipy.sparse import csr_matrix
from alice import data_utils
import utils

def header(): return 'Week 1: Prepare dataset'

def run():

  run_for('3users', 10, True)
  run_for('10users', 10)
  run_for('150users', 10)

  return

def run_for(dir, session_length, details=False):
  train_data, site_freq, X_sparse, y = data_utils.prepare_train_set(dir, session_length)

  print('----------------'+dir+'----------------')
  print()
  print(train_data.head())
  if details:
    print(site_freq)
    print(X_sparse.todense())
  print(len(train_data))
  print(len(site_freq))
  if details:
    X, y = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    print(X), print(y)

  print('TOP 10 sites:')
  for i, site in enumerate(site_freq):
    if i>=10: break
    print(site)

  print()