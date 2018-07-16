import numpy as np
import pandas as pd
import itertools
import utils
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from scipy import stats

from alice import data_utils

def header(): return 'Week 2: Initial data analysis'

def run():

  #run_for(data_utils.DIR_3USERS, 5, 3, details=True)
  #run_for(data_utils.DIR_10USERS, 10, 10, details=True)
  #run_for(data_utils.DIR_150USERS, 10, 10)

  part1_prepare()

  part2_hypotheses()

  return

def run_for(dir, session_length, window_size, details=False):
  X_sparse, y = data_utils.ensure_sparse_train_set_window(dir, session_length, window_size)

  print('---------------- '+dir+' ----------------')
  print(X_sparse.shape)
  print(X_sparse.todense())
  print(y)
  print()

def part1_prepare():
  data_lengths = []
  for dir in ['10users', '150users']:
    for window_size, session_length in itertools.product([10, 7, 5], [15, 10, 7, 5]):
      if window_size <= session_length: # and (window_size, session_length) != (10, 10):
        X_sparse, y = data_utils.ensure_sparse_train_set_window(dir, session_length, window_size)
        data_lengths.append(X_sparse.shape[0])

  data_lengths = set(data_lengths)
  print(data_lengths)
  print('distinct:', len(data_lengths))

def part2_hypotheses():

  fpath = utils.PATH.STORE_FOR('alice', 'train_data_{}.csv'.format('10users'))
  train_df = pd.read_csv(fpath, index_col='session_id')
  print(train_df.head())
  print(train_df.info())
  print()
  print(train_df['user_id'].value_counts())

  num_unique_sites = [np.unique(train_df.values[i, :-1]).shape[0] for i in range(train_df.shape[0])]
  unique_sites = pd.Series(num_unique_sites)
  unique_counts = unique_sites.value_counts()
  print(unique_counts)

  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))

  unique_sites.hist(ax=axes[0])

  norm_distr_test_result = stats.shapiro(unique_counts)
  print(norm_distr_test_result)
  stats.probplot(unique_counts, dist='norm', plot=axes[1])

  # hypothesis check

  has_two_similar = (np.array(num_unique_sites) < 10).astype('int')
  all = len(has_two_similar)
  success  = has_two_similar.sum()
  failures = all - success
  pi_val = stats.binom_test([success, failures], p=0.95, alternative='greater')
  print(pi_val)

  binomial_interval = proportion_confint(success, all, alpha=0.05, method='wilson')
  print(binomial_interval)

  all_site_freqs = data_utils.prepare_site_freqs('10users')
  site_freqs = [ all_site_freqs[key][1] for key in all_site_freqs
                 if all_site_freqs[key][1]>=1000 and all_site_freqs[key][0] != 0 ]
  pd.Series(site_freqs).hist(ax=axes[2])

  all_site_freqs = [ all_site_freqs[key][1] for key in all_site_freqs if all_site_freqs[key][0] != 0 ]
  n_samples = len(all_site_freqs)

  site_freq_scores = [np.mean(sample) for sample in get_bootstrap_samples(all_site_freqs, n_samples)]
  intervals = stat_intervals(site_freq_scores, 0.05)
  print(intervals)

  plt.show()

  return

def get_bootstrap_samples(data, n_samples, random_seed=17):
  np.random.seed(random_seed)
  indices = np.random.randint(0, len(data), (n_samples, len(data)))
  samples = np.array(data)[indices]
  return samples

def stat_intervals(stat, alpha):
  return np.percentile(stat, [ 100*alpha/2.0, 100*(1 - alpha/2.0) ])