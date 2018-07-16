import os
import datetime as dt
import glob
import math
import numpy as np
import pandas as pd
import pickle
import utils
from collections import Counter
from scipy.sparse import csr_matrix

DIR_3USERS   = '3users'
DIR_10USERS  = '10users'
DIR_150USERS = '150users'
ALICE_DATA   = os.path.join(utils.PATH.COURSE_DATA, 'alice')

# -------------- GENERAL --------------

def ALICE_DIR(dir):
  return os.path.join(ALICE_DATA, dir)

def ALICE_FILE(dir, fn):
  return os.path.join(ALICE_DIR(dir), fn)

def get_files(dir):
  path = ALICE_DIR(dir)
  pattern = path+'/user*.csv'
  return glob.glob(pattern)

def load_user_data(fpath):
  fname = os.path.split(fpath)[1]
  id = int(fname[4:8])
  return id, pd.read_csv(fpath)

# -------------- WEEK 1 --------------

def get_train_data_files(dir):
  train_data_fname = utils.PATH.STORE_FOR('alice', 'train_data_{}.csv'.format(dir))
  X_sparse_fname = utils.PATH.STORE_FOR('alice', 'X_sparse_{}.pkl'.format(dir))
  y_fname = utils.PATH.STORE_FOR('alice', 'y_{}.pkl'.format(dir))

  return train_data_fname, X_sparse_fname, y_fname

def prepare_site_freqs(dir):
  # try load file with frequences
  site_freq_fname  = utils.PATH.STORE_FOR('alice', 'site_freq_{}.pkl'.format(dir))
  if os.path.exists(site_freq_fname):
    try:
      with open(site_freq_fname, 'rb') as f:
        return pickle.load(f)
    except:
      os.remove(site_freq_fname)

  # extract frequences
  freq_dict = {}
  for f in get_files(dir):
    user_id, user_df = load_user_data(f)
    sites = user_df['site']
    for s in sites:
      freq_dict[s] = freq_dict.get(s, 0) + 1

  site_freqs = {}
  for i, kv in enumerate(sorted(freq_dict.items(), key=lambda t: t[1], reverse=True)):
    site_freqs[kv[0]] = (i+1, kv[1])

  #save file with frequences
  with open(site_freq_fname, 'wb') as f:
    pickle.dump(site_freqs, f, protocol=2)

  return site_freqs

def sparsify(X):
  indprt = [0]
  indices = []
  data = []
  vocabulary = {}

  for d in X:
    for site_id in d:
      if site_id==0: continue
      idx = vocabulary.setdefault(site_id, len(vocabulary))
      indices.append(idx)
      data.append(1)
    indprt.append(len(indices))

  return csr_matrix((data, indices, indprt), dtype=int)

def load_train_set(dir):
  train_data_fname, X_sparse_fname, y_fname = get_train_data_files(dir)
  if os.path.exists(train_data_fname) and \
     os.path.exists(X_sparse_fname) and \
     os.path.exists(y_fname):
    try:
      train_data = pd.read_csv(train_data_fname, index_col='session_id')
      with open(X_sparse_fname, 'rb') as f:
        X_sparse = pickle.load(f)
      with open(y_fname, 'rb') as f:
        y = pickle.load(f)
      return train_data, X_sparse, y
    except:
      os.remove(train_data_fname)
      os.remove(X_sparse_fname)
      os.remove(y_fname)

  return None, None, None

def save_train_set(train_data, X_sparse, y, dir):
  train_data_fname, X_sparse_fname, y_fname = get_train_data_files(dir)
  train_data.to_csv(train_data_fname, index_label='session_id', float_format='%d')
  with open(X_sparse_fname, 'wb') as f:
    pickle.dump(X_sparse, f, protocol=2)
  with open(y_fname, 'wb') as f:
    pickle.dump(y, f, protocol=2)

def prepare_train_set(dir, session_length):
  # try load saved files
  train_data, X_sparse, y = load_train_set(dir)
  site_freq = prepare_site_freqs(dir)
  if train_data is not None and \
     X_sparse is not None and \
     y is not None:
    return train_data, site_freq, X_sparse, y

  # preprocess data #1: digitize user session history
  data = np.zeros([0, session_length+1])
  for f in get_files(dir):
    user_id, user_df = load_user_data(f)
    site_ids = user_df['site'].apply(lambda s: site_freq[s][0]).values
    n = len(site_ids)
    if (n%session_length>0):
      zero_tail = [0]*(session_length - n%session_length)
      site_ids = np.concatenate([site_ids, zero_tail])

    user_data = np.reshape(site_ids, [len(site_ids)//session_length, session_length])
    user_data = np.concatenate([user_data, [[user_id]]*user_data.shape[0]], axis=1)

    data = np.concatenate([data, user_data]).astype(int)

  cols = [ 'site{0}'.format(j) for j in range(1, session_length+1) ] + ['user_id']
  train_data = pd.DataFrame(data, columns=cols)

  # preprocess data #2: put user session history into sparse matrix
  X = train_data.iloc[:, :-1].values
  y = train_data.iloc[:, -1].values
  X_sparse = sparsify(X)

  # save data files
  save_train_set(train_data, X_sparse, y, dir)

  return train_data, site_freq, X_sparse, y

# -------------- WEEK 2 --------------

def ensure_sparse_train_set_window(dir, session_length, window_size):

  X_sparse_fname = utils.PATH.STORE_FOR('alice', 'X_sparse_{0}_s{1}_w{2}.pkl'.format(dir, session_length, window_size))
  y_fname = utils.PATH.STORE_FOR('alice', 'y_{0}_s{1}_w{2}.pkl'.format(dir, session_length, window_size))

  if os.path.exists(X_sparse_fname) and os.path.exists(y_fname):
    try:
      with open(X_sparse_fname, 'rb') as f:
        X_sparse = pickle.load(f)
      with open(y_fname, 'rb') as f:
        y = pickle.load(f)
      return X_sparse, y
    except:
      os.remove(X_sparse_fname)
      os.remove(y_fname)

  X_sparse, y = prepare_sparse_train_set_window(dir, session_length, window_size)

  with open(X_sparse_fname, 'wb') as f:
    pickle.dump(X_sparse, f, protocol=2)
  with open(y_fname, 'wb') as f:
    pickle.dump(y, f, protocol=2)

  return X_sparse, y

def prepare_sparse_train_set_window(dir, session_length, window_size):

  site_freq = prepare_site_freqs(dir)
  data = np.zeros([0, session_length+1])

  for f in get_files(dir):
    user_id, user_df = load_user_data(f)
    site_ids = user_df['site'].apply(lambda s: site_freq[s][0])
    n = len(site_ids)
    snum = math.ceil(n/window_size)
    user_data = np.zeros([snum, session_length+1])
    for i in range(snum):
      i_begin = i*window_size
      i_end   = min(i_begin + session_length, n)
      user_data[i, 0: i_end-i_begin] = site_ids.iloc[i_begin:i_end]
      user_data[i, session_length] = user_id
    data = np.append(data, user_data, axis=0)

  cols = [ 'site{0}'.format(j) for j in range(1, session_length+1) ] + ['user_id']
  train_data = pd.DataFrame(data, columns=cols)
  X = train_data.iloc[:, :-1].values
  y = train_data.iloc[:, -1].values
  X_sparse = sparsify(X)

  return X_sparse, y

# -------------- WEEK 3 --------------

def ensure_train_set_with_fe(dir, session_length, window_size):
  sparse_data_fname = utils.PATH.STORE_FOR('alice', 'sparse_data_{0}_s{1}_w{2}.pkl'.format(dir, session_length, window_size))
  extra_data_fname  = utils.PATH.STORE_FOR('alice', 'extra_data_{0}_s{1}_w{2}.csv'.format(dir, session_length, window_size))

  if os.path.exists(sparse_data_fname) and os.path.exists(extra_data_fname):
    try:
      with open(sparse_data_fname, 'rb') as f:
        sparse_data = pickle.load(f)
      extra_data = pd.read_csv(extra_data_fname)
      return sparse_data, extra_data
    except:
      os.remove(sparse_data)
      os.remove(extra_data)

  print('prepare train set for', dir)
  sparse_data, extra_data = prepare_train_set_with_fe(dir, session_length, window_size)

  with open(sparse_data_fname, 'wb') as f:
    pickle.dump(sparse_data, f, protocol=2)
  extra_data.to_csv(extra_data_fname)

  return sparse_data, extra_data

def prepare_train_set_with_fe(dir, session_length, window_size):
  site_freqs = prepare_site_freqs(dir)
  top_sites_cnt = 30
  top_sites = { site_freqs[k][0]:site_freqs[k][1] for k in site_freqs }
  top_sites = list(Counter(top_sites).most_common(top_sites_cnt))
  top_sites_cnt = min(top_sites_cnt, len(top_sites))
  extra_feat_num = 12 + top_sites_cnt

  sparse_data = np.zeros([0, session_length])
  extra_data  = np.zeros([0, extra_feat_num + 1])

  for f in get_files(dir):
    user_id, user_df = load_user_data(f)
    site_ids = user_df['site'].apply(lambda s: site_freqs[s][0])
    site_tms = pd.to_datetime(user_df['timestamp'])
    n = len(site_ids)
    row_num = math.ceil(n/window_size)

    user_sparse_data = np.zeros([row_num, session_length])
    user_extra_data = np.zeros([row_num, extra_feat_num + 1])

    for i in range(row_num):
      i_begin = i*window_size
      i_end   = min(i_begin + session_length, n)
      sites   = site_ids[i_begin:i_end]
      times   = site_tms.iloc[i_begin:i_end]

      # Session site IDs to sparsify

      user_sparse_data[i, 0: i_end-i_begin] = site_ids.iloc[i_begin:i_end]

      # Extra features: feature engineering

      # time features
      start_time = times.iloc[0]
      user_extra_data[i, 0] = (times.max() - times.min()).total_seconds()  #duration
      user_extra_data[i, 1] = start_time.hour  #hour
      user_extra_data[i, 2] = start_time.weekday() #week_day
      user_extra_data[i, 3] = start_time.day   #day
      user_extra_data[i, 4] = start_time.month #month
      user_extra_data[i, 5] = start_time.year  #year
      user_extra_data[i, 6] = 1 if (start_time.hour >= 9  and start_time.hour < 13) or \
                             (start_time.hour >= 14 and start_time.hour <= 18) \
                          else 0 #work_time
      user_extra_data[i, 7]  = 1 if (start_time.hour >= 5  and start_time.hour < 10) else 0 #is morning
      user_extra_data[i, 8]  = 1 if (start_time.hour >= 10 and start_time.hour < 18) else 0 #is day
      user_extra_data[i, 9]  = 1 if (start_time.hour >= 18 and start_time.hour < 22) else 0 #is evening
      user_extra_data[i, 10] = 1 if (start_time.hour >= 22 or  start_time.hour < 5)  else 0 #is night

      # sites features
      user_extra_data[i, 11] = len(set(sites)) #unique_sites
      col_idx = 12
      for top_site in top_sites:
        user_extra_data[i, col_idx] = 1 if top_site[0] in sites else 0
        col_idx += 1

      # target column: user_id
      user_extra_data[i, -1] = user_id

    sparse_data = np.append(sparse_data, user_sparse_data, axis=0).astype(int)
    extra_data  = np.append(extra_data, user_extra_data, axis=0)

  train_data = pd.DataFrame(dtype='int')

  train_data['duration'] = extra_data[:, 0]
  train_data = pd.concat([train_data, pd.get_dummies(extra_data[:, 1].astype(int), prefix='hour',     prefix_sep='')], axis=1)
  train_data = pd.concat([train_data, pd.get_dummies(extra_data[:, 2].astype(int), prefix='week_day', prefix_sep='')], axis=1)
  train_data = pd.concat([train_data, pd.get_dummies(extra_data[:, 3].astype(int), prefix='day',      prefix_sep='')], axis=1)
  train_data = pd.concat([train_data, pd.get_dummies(extra_data[:, 4].astype(int), prefix='month',    prefix_sep='')], axis=1)
  train_data = pd.concat([train_data, pd.get_dummies(extra_data[:, 5].astype(int), prefix='year',     prefix_sep='')], axis=1)
  train_data['work_time']  = extra_data[:, 6]
  train_data['is_morning'] = extra_data[:, 7]
  train_data['is_day']     = extra_data[:, 8]
  train_data['is_evening'] = extra_data[:, 9]
  train_data['is_night']   = extra_data[:, 10]
  train_data['unique_sites'] = extra_data[:, 11]
  train_data = pd.concat([train_data, pd.DataFrame(data=extra_data[:, 12:12+top_sites_cnt],
                                                   columns=[ 'top_site{0}'.format(i) for i in range(top_sites_cnt) ])], axis=1)
  train_data['target'] = extra_data[:, -1]

  train_data.index.rename('session_id', inplace=True)
  train_data = train_data.astype(int)

  sparse_data = sparsify(sparse_data)

  return sparse_data, train_data
