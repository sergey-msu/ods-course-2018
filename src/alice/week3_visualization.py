import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from alice import data_utils

def header(): return 'Week 3: Visualization anf Features'

def run():

  ensure_files()
  #features()
  #visualization()
  #feature_engineering()

  return

def ensure_files():
  data_utils.ensure_train_set_with_fe(data_utils.DIR_3USERS, 10, 10)
  data_utils.ensure_train_set_with_fe(data_utils.DIR_10USERS, 10, 10)
  data_utils.ensure_train_set_with_fe(data_utils.DIR_150USERS, 10, 10)
  return

def features():
  train_data_3users = data_utils.ensure_train_set_with_fe(data_utils.DIR_3USERS, 10, 10)
  train_data_10users = data_utils.ensure_train_set_with_fe(data_utils.DIR_10USERS, 10, 10)
  train_data_150users = data_utils.ensure_train_set_with_fe(data_utils.DIR_150USERS, 10, 10)

  session_timespan_median = train_data_10users['session_timespan'].median()
  day_of_week_median = train_data_10users['day_of_week'].median()
  print(session_timespan_median)
  print(day_of_week_median)

  start_hour_median = train_data_150users['start_hour'].median()
  unique_sites_median = train_data_150users['#unique_sites'].median()
  print(start_hour_median)
  print(unique_sites_median)
  return

def visualization():
  id_name_dict = {
    128: 'Mary-Kate',
    39:  'Ashley',
    207: 'Lindsey',
    127: 'Naomi',
    237: 'Avril',
    33:  'Bob',
    50:  'Bill',
    31:  'John',
    100: 'Dick',
    241: 'Ed' }

  train_data_10users = data_utils.ensure_train_set_with_fe(data_utils.DIR_10USERS, 10, 10)
  train_data_10users['target'] = train_data_10users['target'].map(id_name_dict)

  color_dic = {
    'Mary-Kate': 'pink',
    'Ashley':    'darkviolet',
    'Lindsey':   'blueviolet',
    'Naomi':     'hotpink',
    'Avril':     'orchid',
    'Bob':       'firebrick',
    'Bill':      'gold',
    'John':      'forestgreen',
    'Dick':      'slategrey',
    'Ed':        'brown'}

  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

  session_data = train_data_10users[train_data_10users['session_timespan'] <= 200][['session_timespan']]
  session_data.hist(color='darkviolet', ax=axes[0][0])
  axes[0][0].set_xlabel('Session Timespan')
  axes[0][0].set_ylabel('Counts')

  unique_sites_data = train_data_10users['#unique_sites']
  unique_sites_data.hist(color='aqua', ax=axes[0][1])
  axes[0][1].set_xlabel('Number of Unique Sites')
  axes[0][1].set_ylabel('Counts')

  unique_sites_data = train_data_10users['start_hour']
  unique_sites_data.hist(color='darkgreen', ax=axes[1][0])
  axes[1][0].set_xlabel('Start Hour')
  axes[1][0].set_ylabel('Counts')

  axis = axes[1][1]
  axis.set_xlim(left=0, right=6)
  axis.set_xticklabels(['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])
  axis.set_xlabel('Day of Week')
  axis.set_ylabel('Counts')
  unique_sites_data = train_data_10users['day_of_week']
  unique_sites_data.hist(color='sienna', ax=axis, bins=7)
  plt.show()

  plot_users_hist(train_data_10users, '#unique_sites', color_dic)
  plot_users_hist(train_data_10users, 'start_hour', color_dic)
  plot_users_hist(train_data_10users, 'day_of_week', color_dic,
                  x_min=0, x_max=6, infer_bins=True, x_ticks=np.arange(7), stick_labels=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])

  site_freqs = dict([ (k, v[1]) for k,v in data_utils.prepare_site_freqs(data_utils.DIR_10USERS).items() ])
  top10_sites = Counter(site_freqs).most_common(10)
  print(top10_sites)
  labels, values = zip(*top10_sites)
  g = sns.barplot(x=labels, y=values)
  g.set_xticklabels(labels=labels, rotation=80)

  plt.show()
  return

def plot_users_hist(train_data_10users, col_name, color_dic,
                    x_min=None, x_max=None, stick_labels=None, x_ticks=None, infer_bins=False):
  fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 10))
  i = 0
  for name in color_dic:
    color = color_dic[name]
    axis = axes[i//5][i%5]
    axis.set_ylabel('')
    if stick_labels is not None:
      axis.set_xticklabels(stick_labels)
    if x_ticks is not None:
      axis.set_xticks(x_ticks)
    if x_min is not None and x_max is not None:
      axis.set_xlim(left=x_min, right=x_max)
    leg_handle = mpatches.Patch(color=color, label=name)
    axis.legend(handles=[leg_handle])

    user_unique_sites_data = train_data_10users[train_data_10users['target'] == name][col_name]
    bins = len(user_unique_sites_data.value_counts()) if infer_bins else 10
    user_unique_sites_data.hist(color=color, ax=axis, bins=bins)

    i += 1

  plt.show()
  return

def feature_engineering():


  return
