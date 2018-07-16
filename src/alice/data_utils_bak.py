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

  site_freq = {}
  for i, kv in enumerate(sorted(freq_dict.items(), key=lambda t: t[1], reverse=True)):
    site_freq[kv[0]] = (i+1, kv[1])

  #save file with frequences
  with open(site_freq_fname, 'wb') as f:
    pickle.dump(site_freq, f, protocol=2)

  return site_freq

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

def ensure_train_set_with_fe(dir, session_length, window_size,
                             use_general=True, use_timespans=False, use_secondary=False, use_extra=False):

  feat_str_mask = '{0}{1}{2}{3}'.format('g' if use_general else '',
                                        't' if use_timespans else '',
                                        's' if use_secondary else '',
                                        'e' if use_extra else '')
  train_fname = utils.PATH.STORE_FOR('alice', 'train_{0}_s{1}_w{2}_f{3}.pkl'.format(dir, session_length, window_size, feat_str_mask))

  if os.path.exists(train_fname):
    try:
      with open(train_fname, 'rb') as f:
        train_data = pickle.load(f)
      return train_data
    except:
      os.remove(train_fname)

  train_data = prepare_train_set_with_fe(dir, session_length, window_size, use_general, use_timespans, use_secondary, use_extra)

  with open(train_fname, 'wb') as f:
    pickle.dump(train_data, f, protocol=2)

  return train_data

def prepare_train_set_with_fe(dir, session_length, window_size,
                              use_general=True, use_timespans=False, use_secondary=False, use_extra=False, top_sites=30):
  site_freq = prepare_site_freqs(dir)

  general_feat_num   = session_length if use_general else 0
  timespans_feat_num = (session_length-1) if use_timespans else 0
  secondary_feat_num = 4 if use_secondary else 0
  extra_feat_num     = 4+30 if use_extra else 0
  total_feat_num     = general_feat_num + timespans_feat_num + secondary_feat_num + extra_feat_num + 1

  data = np.zeros([0, total_feat_num])

  for f in get_files(dir):
    user_id, user_df = load_user_data(f)
    site_ids = user_df['site'].apply(lambda s: site_freq[s][0])
    site_tms = pd.to_datetime(user_df['timestamp'])
    n = len(site_ids)
    row_num = math.ceil(n/window_size)
    user_data = np.zeros([row_num, total_feat_num])

    for i in range(row_num):
      i_begin = i*window_size
      i_end   = min(i_begin + session_length, n)
      sites   = site_ids.iloc[i_begin:i_end]
      times   = site_tms.iloc[i_begin:i_end]
      start_time = times.iloc[0]
      col_idx = 0

      # general info: site ids in session
      if use_general:
        user_data[i, 0: i_end-i_begin] = sites
        col_idx += general_feat_num

      # sites timespans in session
      if use_timespans:
        tspans  = [ (t1 - t0).total_seconds() for t0,t1 in zip(times, times.iloc[1:])]
        if len(tspans)==0:
          tspans.append(1)
        user_data[i, col_idx : col_idx + i_end-i_begin - 1] = tspans
        col_idx += timespans_feat_num

      # secondary features
      if use_secondary:
        user_data[i, col_idx]   = max((times.iloc[-1] - times.iloc[0]).total_seconds(), 1)  #session_timespan
        user_data[i, col_idx+1] = len(set(sites))      #unique_sites
        user_data[i, col_idx+2] = start_time.hour      #start_hour
        user_data[i, col_idx+3] = start_time.weekday() #day_of_week
        col_idx += 4

      # extra features (feature engineering)
      if use_extra:
        top30_sites = Counter(site_freq).most_common(30)
        extra_feat_cnt = 0

        user_data[i, col_idx]   = start_time.year
        user_data[i, col_idx+1] = start_time.month
        user_data[i, col_idx+2] = start_time.day
        user_data[i, col_idx+3] = start_time.hour//4
        extra_feat_cnt += 4

        for top_site in top30_sites:
          user_data[i, col_idx+extra_feat_cnt] = 1 ###################????
          extra_feat_cnt += 1

        # index safeguard
        if extra_feat_cnt != extra_feat_num:
          raise Exception('Index error: extra_feat_cnt={0}, extra_feat_num={1}'.format(extra_feat_cnt, extra_feat_num))

        col_idx += extra_feat_num

      # target column: user_id
      user_data[i, col_idx] = user_id

      # index safeguard
      if col_idx+1 != total_feat_num:
        raise Exception('Index error: last col_idx={0}, total_feat_num={1}'.format(col_idx, total_feat_num))

    data = np.append(data, user_data, axis=0)

  data = data.astype('int')

  cols = []
  if use_general:
    cols += [ 'site{0}'.format(j) for j in range(1, session_length+1) ]
  if use_timespans:
    cols += [ 'time_diff{0}'.format(j) for j in range(1, session_length) ]
  if use_secondary:
    cols += [ 'session_timespan', '#unique_sites', 'start_hour', 'day_of_week' ]
  cols += [ 'target' ]

  train_data = pd.DataFrame(data, columns=cols)
  #X = train_data.iloc[:, :-1].values
  #X_sparse = sparsify(X)

  return train_data
