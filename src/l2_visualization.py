from __future__ import (absolute_import, division, print_function, unicode_literals)
import math
import numpy as np
import pandas as pd
import seaborn as sns
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from pylab import rcParams
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def header(): return 'LECTURE 2: Visualization https://habrahabr.ru/company/ods/blog/323210/'

def run():

  #df = pd.DataFrame(
  #  {"Success": 20*["Yes"] + 20*["No"],
  #   "B": np.random.randint(1, 7, 40)})
  #
  #print(df.head(-1))
  #
  #print('***********************')
  #
  ##df = pd.melt(df, value_vars=['A'], id_vars='Success')
  #print(df.head(-1))
  #sns.violinplot(y='B', x='Success', hue='Success', data=df)
  #plt.show()


  lec_notes()
  homework()

  return

def homework():

  sns.set_context('notebook',
                  font_scale=1.5,
                  rc={
                    'figure.figsize': (12, 9),
                    'axes.titlesize': 18})

  train = pd.read_csv(utils.PATH.COURSE_FILE('mlbootcamp5_train.csv'), sep=';', index_col='id')
  print(train.head())
  print('dataset size:', train.shape)

  # -----------------------------

  train_uniques = pd.melt(frame=train,
                          value_vars=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'])
  train_uniques = pd.DataFrame(train_uniques.groupby(['variable', 'value'])['value'].count()) \
                    .sort_index(level=[0, 1]) \
                    .rename(columns={'value': 'count'}) \
                    .reset_index()
  print(train_uniques.head(-1))

  sns.factorplot(x='variable', y='count', hue='value',
                 data=train_uniques, kind='bar', size=12)
  plt.show()

  # -----------------------------

  train_uniques = pd.melt(frame=train,
                          value_vars=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
                          id_vars=['cardio'])
  train_uniques = pd.DataFrame(train_uniques.groupby(['variable', 'value', 'cardio'])['value'].count()) \
                    .sort_index(level=[0, 1]) \
                    .rename(columns={'value': 'count'}) \
                    .reset_index()
  print(train_uniques.head(-1))

  sns.factorplot(x='variable', y='count', hue='value',
                 col='cardio', data=train_uniques, kind='bar', size=9)
  plt.show()

  print('*************************')

  for c in train.columns:
    n = train[c].nunique()
    print(c)

    if (n<=3):
      print(n, sorted(train[c].value_counts().to_dict().items()))
    else:
      print(n)
    print(10*'-')

  print('*************************')

  corr_matrix_pearson = train.corr(method='pearson')
  sns.heatmap(corr_matrix_pearson)
  plt.show()

  sns.violinplot(x='gender', y='height', hue='gender', scale='count', split=True, data=train)
  plt.show()

  df_woman = train[train.gender==1]['height']
  df_man   = train[train.gender==2]['height']
  ax = sns.kdeplot(df_woman, legend=True, bw=0.5)
  sns.kdeplot(df_man, legend=True, bw=0.5)
  plt.show()

  corr_matrix_spearman = train.corr(method='spearman')
  sns.heatmap(corr_matrix_spearman)
  plt.show()

  df_cleared = train[(train['ap_hi']>train['ap_lo']) &
                     (train['ap_hi']<400) &
                     (train['ap_lo']<400) &
                     (train['ap_hi']>0) &
                     (train['ap_lo']>0)]
  ap_hi = np.log(df_cleared['ap_hi'])
  ap_lo = np.log(df_cleared['ap_lo'])

  ax = sns.jointplot(ap_hi, ap_lo)
  plt.show()

  train['age_years'] = (train['age'] // 365.25).astype(int)
  sns.countplot(x='age_years', hue='cardio', data=train)
  plt.show()

  return


def lec_notes():

  rcParams['figure.figsize'] = 8, 5
  df = pd.read_csv(utils.PATH.COURSE_FILE('video_games_sales.csv'))
  print(df.info())
  print(len(df))

  df = df.dropna()

  df['User_Score']      = df.User_Score.astype('float64')
  df['Year_of_Release'] = df.Year_of_Release.astype('int64')
  df['User_Count']      = df.User_Count.astype('int64')
  df['Critic_Count']    = df.Critic_Count.astype('int64')

  print(df.shape)

  useful_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Global_Sales',
                 'Critic_Score', 'Critic_Count',
                 'User_Score', 'User_Count', 'Rating']
  print(df[useful_cols].head())

  pandas(df)
  seaborn(df)
  plotly(df)

  df = pd.read_csv(utils.PATH.COURSE_FILE('telecom_churn.csv'))
  #print(df.head())
  print(df.info())
  print(df.shape)

  #visual_analysis(df)
  t_sne(df)

  return

def pandas(df):
  sales_df = df[[x for x in df.columns if 'Sales' in x] + ['Year_of_Release']]
  sales_df.groupby('Year_of_Release').sum().plot()
  plt.show()

  sales_df.groupby('Year_of_Release').sum().plot(kind='bar', rot=45)
  plt.show()

  return

def seaborn(df):
  cols = ['Global_Sales',
          'Critic_Score', 'Critic_Count',
          'User_Score', 'User_Count']
  sns_plot = sns.pairplot(df[cols])
  sns_plot.savefig('pairplot.png')

  sns.distplot(df.Critic_Score)
  plt.show()

  sns.jointplot(df['Critic_Score'], df['User_Score'])
  plt.show()

  top_platforms = df.Platform.value_counts().sort_values(ascending=False).head(5).index.values
  sns.boxplot(y='Platform', x='Critic_Score', data=df[df.Platform.isin(top_platforms)], orient='h')
  plt.show()

  platform_genre_sales = df.pivot_table(index='Platform',
                                        columns='Genre',
                                        values='Global_Sales',
                                        aggfunc=sum).fillna(0).applymap(float)
  sns.heatmap(platform_genre_sales, annot=True, fmt='0.1f', linewidths=0.5)
  plt.show()

  return

def plotly(df):
  init_notebook_mode(connected=True)

  #1 посчитаем число вышедших игр и проданных копий по годам

  df_sales = df.groupby('Year_of_Release')[['Global_Sales']].sum()
  df_cnts  = df.groupby('Year_of_Release')[['Name']].count()
  df_years = df_sales.join(df_cnts)
  df_years.columns = ['Global_Sales', 'Number_of_Games']

  trace0 = go.Scatter(x=df_years.index,
                      y=df_years.Global_Sales,
                      name='Global Sales')

  trace1 = go.Scatter(x=df_years.index,
                      y=df_years.Number_of_Games,
                      name='Number of games released')

  data = [trace0, trace1]
  layout = {'title': 'Statistics of video games'}

  fig = go.Figure(data=data, layout=layout)
  plot(fig, filename='years_stats.html', show_link=False)

  #2 считаем число проданных и вышедших игр по платформам

  df_platforms_sales = df.groupby('Platform')[['Global_Sales']].sum()
  df_platforms_cnt   = df.groupby('Platform')[['Name']].count()
  df_platforms = df_platforms_sales.join(df_platforms_cnt)
  df_platforms.columns = ['Global_Sales', 'Number_of_Games']
  df_platforms.sort_values('Global_Sales', ascending=False, inplace=True)

  trace0 = go.Bar(x=df_platforms.index,
                  y=df_platforms.Global_Sales,
                  name='Global Sales')

  trace1 = go.Bar(x=df_platforms.index,
                  y=df_platforms.Number_of_Games,
                  name='Numer of games released')

  data = [trace0, trace1]
  layout = {'title': 'Share of platforms', 'xaxis': {'title': 'platform'}}

  fig = go.Figure(data=data, layout=layout)
  plot(fig, show_link=False)

  #3 создаем Box trace для каждого жанра из наших данных

  data = []
  for genre in df.Genre.unique():
    data.append(go.Box(y=df[df.Genre==genre].Critic_Score, name=genre))

  plot(data, show_link=False)

  return

def visual_analysis(df):

  print(df['Churn'].value_counts())

  df['Churn'].value_counts().plot(kind='bar', label='Churn')
  plt.legend()
  plt.title('Ottok clientov')
  plt.show()

  corr_matrix = df.drop(['State', 'International plan', 'Voice mail plan', 'Area code'], axis=1).corr()
  sns.heatmap(df.corr())
  plt.show()

  features = list(set(df.columns)-set(['State', 'International plan', 'Voice mail plan', 'Area code',
                                       'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge',
                                       'Churn']))
  df[features].hist(figsize=(20,12))
  plt.show()

  #sns.pairplot(df[features + ['Churn']], hue='Churn')
  #plt.show()

  fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))

  for idx, feat in enumerate(features):
    ax_row = int(idx/4)
    ax_col = int(idx%4)
    sns.boxplot(x='Churn', y=feat, data=df, ax=axes[ax_row, ax_col])
    axes[ax_row, ax_col].legend()
    axes[ax_row, ax_col].set_xlabel('Churn')
    axes[ax_row, ax_col].set_ylabel(feat)
  plt.show()

  _, axes = plt.subplots(2, 2, figsize=(16, 6))

  sns.boxplot(x='Churn', y='Total day minutes', data=df, ax=axes[0, 0])
  sns.violinplot(x='Churn', y='Total day minutes', data=df, ax=axes[0, 1])
  sns.countplot(x='International plan', hue='Churn', data=df, ax=axes[1, 0]);
  sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=axes[1, 1]);
  plt.show()

  sns.countplot(x='Customer service calls', hue='Churn', data=df)
  plt.show()

  st = df.groupby(['State'])['Churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T
  print(st)

  return

def t_sne(df):

  X = df.drop(['State'], axis=1)
  X['International plan'] = pd.factorize(X['International plan'])[0]
  X['Voice mail plan'] = pd.factorize(X['Voice mail plan'])[0]

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  tsne = TSNE(random_state=17)
  tsne_representation = tsne.fit_transform(X_scaled)

  plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1])
  plt.show()


  return