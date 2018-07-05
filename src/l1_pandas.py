import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

heading = utils.PRINT.HEADER


def header():
    return 'LECTURE 1: Pandas https://habrahabr.ru/company/ods/blog/322626/'


def run():

    #df = pd.read_csv(utils.PATH.COURSE_FILE('telecom_churn.csv'))
    #basics(df)
    #churn_forecasting(df)
    homework()
    #pandas10min()

    return

def homework():

    df = pd.read_csv(utils.PATH.COURSE_FILE('mlbootcamp5_train.csv'),
                     sep=';',
                     index_col='id')
    print(df.head())

    r = df.groupby(['gender'])[['height', 'gender']].agg(['mean', 'count']).sort_values(by=[('height', 'mean')], ascending=True)
    gm  = r['gender']['mean']
    hm  = r['height']['mean']
    cnt = r['height']['count']
    female_id = gm.iloc[0]
    male_id   = gm.iloc[1]
    print("female: {}, av_height: {}, count: {}".format(female_id, hm.iloc[0], cnt.iloc[0]))
    print("male:   {}, av_height: {}, count: {}".format(male_id,   hm.iloc[1], cnt.iloc[1]))
    print("total: ", cnt.iloc[0]+cnt.iloc[1])

    r = df.groupby(['gender'])[['alco']].agg(['mean'])
    print(r)

    r = df.groupby(['gender'])[['smoke']].agg(['mean'])
    r['smoke_pct'] = round(r[('smoke', 'mean')]*100, 2)
    print(r)
    f = round(r['smoke_pct'].iloc[1]/r['smoke_pct'].iloc[0])
    print(f)

    r = df.groupby(['smoke'])[['age']].agg(['median'])
    m = (r.iloc[0] - r.iloc[1])/30
    print(m)

    df['age_years'] = round(df['age']/365)
    so = df[(df['smoke']==1) & (df['age_years']>=60) & (df['age_years']<=64)]

    so_low = so[(so['ap_hi']<120) & (so['cholesterol']==1)]
    so_hi  = so[(so['ap_hi']>=160) & (so['ap_hi']<180) & (so['cholesterol']==3)]

    print('----------------------')

    so_low_cnt = so_low['cardio'].mean()
    so_hi_cnt  = so_hi['cardio'].mean()
    print(round(so_hi_cnt/so_low_cnt))

    df['BMI'] = df['weight']/((df['height']/100)**2)
    print(df['BMI'][:5])

    m = df['BMI'].median()
    print(m)

    r = df.groupby(['gender'])[['BMI']].agg(['mean'])
    print(r)

    r = df.groupby(['cardio'])[['BMI']].agg(['mean'])
    print(r)

    r = df.groupby(['gender', 'alco', 'cardio'])[['BMI']].agg(['mean'])
    print(r)

    print('----------------------');

    h_min = df['height'].quantile(0.025)
    h_max = df['height'].quantile(0.975)
    w_min = df['weight'].quantile(0.025)
    w_max = df['weight'].quantile(0.975)
    clear = df[(df['ap_lo']<=df['ap_hi']) & (df['height']>=h_min) & (df['height']<=h_max) & (df['weight']>=w_min) & (df['weight']<=w_max)]
    print(len(df.index))
    print(len(clear.index))
    print(round(100*len(clear.index)/len(df.index)))

    return

def churn_forecasting(df):

    r = pd.crosstab(df['Churn'], df['International plan'], margins=True)
    print(r)

    r = pd.crosstab(df['Churn'], df['Customer service calls'], margins=True)
    print(r)

    df['Many_service_calls'] = (df['Customer service calls']>3).astype('int')
    r = pd.crosstab(df['Many_service_calls'], df['Churn'], margins=True)
    print(r)

    r = pd.crosstab(df['Many_service_calls'] & df['International plan'] , df['Churn'])
    print(r)

    return

def basics(df):
    h = df.head()
    #print(h)

    print(df.shape)
    print(df.columns)
    #print(df.info())

    df['Churn'] = df['Churn'].astype('int64')

    print(df.info())

    print(df.describe())
    print(df.describe(include=['object', 'bool']))
    print(df['Churn'].value_counts()) # values distribution
    print(df['Area code'].value_counts(normalize=True))

    # ----------- SORTING -----------

    h = df.sort_values(by=['Churn', 'Total day charge'], ascending=[True, False]).head()
    print(h)

    # ----------- INDEXING -----------

    print(df['Churn'].mean())
    print(df[df['Churn'] == 1].mean())

    print(df[df['Churn'] == 1]['Total day minutes'].mean())
    print(df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max())

    x = df.loc[0:5, 'State':'Area code']
    print(x)

    x = df.iloc[0:5, 0:3]
    print(x)

    print(df[:2])  # first two rows
    print(df[-1:]) # last row

    # ----------- FUNC APPLYING -----------

    x = df.apply(np.max)  # apply max to each column
    print(x)

    d = {'No': False, 'Yes': True}
    df['International plan'] = df['International plan'].map(d)
    print(df.head())

    df.replace({'Voice mail plan': d})  # ??? seems not workings
    print(df.head())

    # ----------- GROUPING -----------

    r = df.groupby(['Churn'])[['Total day minutes', 'Total eve minutes', 'Total night minutes']].describe(percentiles=[])
    print(r)

    # the same with agg function
    r = df.groupby(['Churn'])[['Total day minutes', 'Total eve minutes', 'Total night minutes']].agg([np.mean, np.std, np.max])
    print(r)

    # ----------- SUMMARY(PIVOT) TABLES -----------

    r = pd.crosstab(df['Churn'], df['International plan'])
    print(r)

    r = pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True)
    print(r)

    r = df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
                       ['Area code'],
                       aggfunc='mean').head(10)
    print(r)

    # ----------- DATAFRAME TRANSORMATIONS -----------

    # add new column
    total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
    df.insert(loc=len(df.columns), column='Total calls', value = total_calls)

    df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
    print(df.head())

    # drop column
    df = df.drop(['Total charge', 'Total calls'], axis=1)

    # drop rows
    df = df.drop([1, 2])

    print(df.head())

    return

def pandas10min():
    '''http://pandas.pydata.org/pandas-docs/stable/10min.html'''

    heading('Series & DataFrames')

    s = pd.Series([1, 3, 5, np.nan, 6, 8])
    dates = pd.date_range('20130101', periods=6, freq='D')
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df)

    df2 = pd.DataFrame({ 'A': 1.0, \
                         'B': pd.Timestamp('20130101'), \
                         'C': pd.Series(1, index=list(range(4)), dtype='float32'), \
                         'D': np.array([3]*4, dtype='int32'), \
                         'E': pd.Categorical(['test', 'train', 'test', 'train']), \
                         'F': 'foo'})
    print(df2)
    print(df2.dtypes)

    print(df.head(3))
    print(df.tail(3))

    heading('rows/columns headers, data (numpy array)')
    print(df.index)
    print(df.columns)
    print(df.values)

    heading('quick statistic summary')
    print(df.describe())

    heading('transpose')
    print(df.T)

    heading('sort by axis header')
    print(df.sort_index(axis=1, ascending=False))

    heading('sort by values')
    print(df.sort_values(by='B'))

    heading('selecting single column')
    print(df['A'])

    heading('selecting single column #2')
    print(df.A)

    heading('slicing rows')
    print(df[1:3])

    heading('slicing rows #2')
    print(df['20130101':'20130104'])

    heading('row by label')
    print(df.loc[dates[0]])

    heading('multi-axis label')
    print(df.loc[:,['A','B']]) # all rows, two columns
    print(df.loc['20130102':'20130105', ['B', 'C']])

    heading('dimension reduction')
    print(df.loc['20130105', ['A', 'C']])

    heading('scalar value')
    print(df.loc[dates[0], 'A'])
    print(df.at[dates[0], 'A']) # fast access

    heading('row by position')
    print(df.iloc[3])

    heading('slicing')
    print(df.iloc[3:5, 0:2])
    print(df.iloc[[1, 2, 5], [2, 3]])
    print(df.iloc[1:3, :])
    print(df.iloc[:, 1:3])

    heading('getting value')
    print(df.iloc[1,2])
    print(df.iat[1,2])

    heading('boolean indexing')
    print(df[df.A > 0])

    heading('apply boolead condition')
    print(df[df>0])

    heading('isin() filtering')
    df2 = df.copy()
    df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
    print(df2)
    print(df2[df2['E'].isin(['two', 'four'])])

    heading('setting new column')
    s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130101', periods=6))
    df['F'] = s1
    print(s1)
    print(df)

    heading('setting by label/index')
    df.at[dates[0], 'A'] = 0
    df.iat[1, 2] = 1

    heading('setting by numpy array')
    df.loc[:, 'D'] = np.array([5]*len(df))
    print(df)

    heading('setting with where')
    df2 = df.copy()
    df[df > 0] = -df2
    print(df)

    return
