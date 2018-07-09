import json
import numpy as np
import pandas as pd
import utils
import pytesseract
import requests
import reverse_geocoder as revgc
#import statsmodels.api as sm
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, f_classif
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import beta
from scipy.stats import shapiro
from scipy.stats import lognorm
from scipy.spatial.distance import euclidean
from scipy.misc import face
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from PIL import Image
from io import BytesIO


def header(): return "LECTION 6: Feature Extraction, Engineering, Transformation and Selection https://habrahabr.ru/company/ods/blog/325422/"

def run():

  with open(utils.PATH.COURSE_FILE('train.json', 'renthop')) as f:
    data = json.load(f)
    df = pd.DataFrame(data)
  #print(df.head())

  print('Feature Extraction')

  #bag_of_words()
  #n_grams()
  #images_fine_tuning()
  #read_text_in_image()
  #geodata(df)
  #date_time(df)

  print('Feature Transformation')

  #scaling()
  #lognormal()
  #qq_plot(df)

  print('Feature Engineering')

  #feat_engineering(df)
  #interactions(df)

  print('Feature Selection')

  #statistical()
  #baseline_modelling(df)
  selector(df)

  return

def vectorize(text, dict):
  vector = np.zeros(len(dict))
  for i, word in dict:
    num = 0
    for w in text:
      if w==word:
        num += 1
    if num>0:
      vector[i] = num
  return vector

def bag_of_words():
  texts = [['i', 'have', 'a', 'cat'],
           ['he', 'have', 'a', 'dog'],
           ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]
  dict = list(enumerate(set(reduce(lambda x, y: x + y, texts))))
  print(dict)
  for t in texts:
    print(vectorize(t, dict))

def n_grams():
  vect = CountVectorizer(ngram_range=(1,1))
  bag_of_words = vect.fit_transform(['no i have cows', 'i have no cows']).toarray()
  print(vect.vocabulary_)
  print(bag_of_words)

  vect = CountVectorizer(ngram_range=(1,2))
  ngram = vect.fit_transform(['no i have cows', 'i have no cows']).toarray()
  print(vect.vocabulary_)
  print(ngram)

  vect = CountVectorizer(ngram_range=(3,3), analyzer='char_wb') # char n-grams
  n1, n2, n3, n4 = vect.fit_transform([ 'иванов', 'петров', 'петренко', 'смит' ]).toarray()
  print(euclidean(n1, n2))
  print(euclidean(n2, n3))
  print(euclidean(n3, n4))

def images_fine_tuning():
  resnet_settings = { 'include_top': False, 'weights': 'imagenet' }
  resnet = ResNet50(**resnet_settings)

  img = image.array_to_img(face())
  img = img.resize((224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  features = resnet.predict(x)
  print(features)

def read_text_in_image():
  img = 'http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg'
  img = requests.get(img)
  with Image.open(BytesIO(img.content)) as img:
    text = pytesseract.image_to_string(img) # throws file not found error. No time to find out the problem origin
    print(text) # Google

def geodata(df):
  gdata = revgc.search((df.latitude.iloc[0], df.longitude.iloc[0]))
  print(gdata)

def date_time(df):
  df['dow'] = pd.to_datetime(df['created']).apply(lambda x: x.date().weekday())
  df['is_weekend'] = df['dow'].apply(lambda x: 1 if x in (5, 6) else 0)

  print(df.head())

  # hours - circle projection
  def make_harmonic_features(value, period=24):
    value *= 2*np.pi/period
    return np.cos(value), np.sin(value)

  print(euclidean(make_harmonic_features(23), make_harmonic_features(1)))
  print(euclidean(make_harmonic_features(9), make_harmonic_features(11)))

  return

def scaling():

  data = beta(1, 10).rvs(1000).reshape(-1, 1)
  print(shapiro(data)) # stats + p-value
  print(shapiro(StandardScaler().fit_transform(data)))

  data = np.array([1,1,0,-1,2,1,2,3,-2,4,100]).reshape(-1, 1).astype(np.float64)
  scaled = StandardScaler().fit_transform(data)
  print(scaled)
  print( (data-data.mean())/data.std() )

  scaled = MinMaxScaler().fit_transform(data)
  print(scaled)
  print( (data - data.min())/(data.max() - data.min()) )
  return

def lognormal():
  data = lognorm(s = 1).rvs(1000)
  print(shapiro(data))
  print(shapiro(np.log(data)))
  return

def qq_plot(df):
  # not working for some reasom. No time to find out
  price = df.price[(df.price <= 2000) & (df.price > 500)]
  price_log = np.log(price)
  price_mm = MinMaxScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
  price_z = StandardScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()

  sm.qqplot(price_log, loc=price_log.mean(), scale=price_log.std()).savefig('qq_price_log.png')
  sm.qqplot(price_mm, loc=price_mm.mean(), scale=price_mm.std()).savefig('qq_price_mm.png')
  sm.qqplot(price_z, loc=price_z.mean(), scale=price_z.std()).savefig('qq_price_z.png')

  return

def feat_engineering(df):
  x_data, y_data = get_data(df)
  print(x_data.head())

  x_data = x_data.values

  score = cross_val_score(LogisticRegression(), x_data, y_data, scoring='neg_log_loss').mean()
  print(score)

  score = cross_val_score(LogisticRegression(), StandardScaler().fit_transform(x_data), y_data, scoring='neg_log_loss').mean()
  print(score) # it works!

  score = cross_val_score(LogisticRegression(), MinMaxScaler().fit_transform(x_data), y_data, scoring='neg_log_loss').mean()
  print(score) # not this time :(

  return

def interactions(df):
  rooms = df["bedrooms"].apply(lambda x: max(x, .5))
  # избегаем деления на ноль; .5 выбран более или менее произвольно
  df["price_per_bedroom"] = df["price"] / rooms
  return

EPSILON = 1e-5

class FeatureEngineer(TransformerMixin):

  def apply(self, df, k, condition):
    df[k] = df['features'].apply(condition)
    df[k] = df[k].astype(np.int8)

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X, y=None):
    df = X.copy()

    df.features = df.features.apply(lambda x: ' '.join([y.replace(' ', '_') for y in x]))
    df.features = df.features.apply(lambda x: x.lower())
    df.features = df.features.apply(lambda x: x.replace('-', '_'))

    for k, condition in (('dishwasher', lambda x: 'dishwasher' in x),
                         ('doorman', lambda x: 'doorman' in x or 'concierge' in x),
                         ('pets', lambda x: "pets" in x or "pet" in x or "dog" in x or "cats" in x and "no_pets" not in x),
                         ('air_conditioning', lambda x: 'air_conditioning' in x or 'central' in x),
                         ('parking', lambda x: 'parking' in x),
                         ('balcony', lambda x: 'balcony' in x or 'deck' in x or 'terrace' in x or 'patio' in x),
                         ('bike', lambda x: 'bike' in x),
                         ('storage', lambda x: 'storage' in x),
                         ('outdoor', lambda x: 'outdoor' in x or 'courtyard' in x or 'garden' in x),
                         ('roof', lambda x: 'roof' in x),
                         ('gym', lambda x: 'gym' in x or 'fitness' in x),
                         ('pool', lambda x: 'pool' in x),
                         ('backyard', lambda x: 'backyard' in x),
                         ('laundry', lambda x: 'laundry' in x),
                         ('hardwood_floors', lambda x: 'hardwood_floors' in x),
                         ('new_construction', lambda x: 'new_construction' in x),
                         ('dryer', lambda x: 'dryer' in x),
                         ('elevator', lambda x: 'elevator' in x),
                         ('garage', lambda x: 'garage' in x),
                         ('pre_war', lambda x: 'pre_war' in x or 'prewar' in x),
                         ('post_war', lambda x: 'post_war' in x or 'postwar' in x),
                         ('no_fee', lambda x: 'no_fee' in x),
                         ('low_fee', lambda x: 'reduced_fee' in x or 'low_fee' in x),
                         ('fire', lambda x: 'fireplace' in x),
                         ('private', lambda x: 'private' in x),
                         ('wheelchair', lambda x: 'wheelchair' in x),
                         ('internet', lambda x: 'wifi' in x or 'wi_fi' in x or 'internet' in x),
                         ('yoga', lambda x: 'yoga' in x),
                         ('furnished', lambda x: 'furnished' in x),
                         ('multi_level', lambda x: 'multi_level' in x),
                         ('exclusive', lambda x: 'exclusive' in x),
                         ('high_ceil', lambda x: 'high_ceil' in x),
                         ('green', lambda x: 'green_b' in x),
                         ('stainless', lambda x: 'stainless_' in x),
                         ('simplex', lambda x: 'simplex' in x),
                         ('public', lambda x: 'public' in x),
                         ):
      self.apply(df, k, condition)

    df['bathrooms'] = df['bathrooms'].apply(lambda x: x if x < 5 else 5)
    df['bedrooms'] = df['bedrooms'].apply(lambda x: x if x < 5 else 5)
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    created = pd.to_datetime(df.pop("created"))
    df["listing_age"] = (pd.to_datetime('today') - created).apply(lambda x: x.days)
    df["room_dif"] = df["bedrooms"] - df["bathrooms"]
    df["room_sum"] = df["bedrooms"] + df["bathrooms"]
    df["price_per_room"] = df["price"] / df["room_sum"].apply(lambda x: max(x, .5))
    df["bedrooms_share"] = df["bedrooms"] / df["room_sum"].apply(lambda x: max(x, .5))
    df['price'] = df['price'].apply(lambda x: np.log(x + EPSILON))

    key_types = df.dtypes.to_dict()
    for k in key_types:
      if key_types[k].name not in ('int64', 'float64', 'int8'):
        df.pop(k)

    for k in ('latitude', 'longitude', 'listing_id'):
        df.pop(k)
    return df


def encode(x):
  if x == 'low':
    return 0
  elif x == 'medium':
    return 1
  elif x == 'high':
    return 2

def get_data(df):
  target = df.pop('interest_level').apply(encode)
  df = FeatureEngineer().fit_transform(df)
  return df, target

def statistical():
  x_data_generated, y_data_generated = make_classification()
  print(x_data_generated.shape)
  print(VarianceThreshold(0.7).fit_transform(x_data_generated).shape)
  print(VarianceThreshold(0.8).fit_transform(x_data_generated).shape)
  print(VarianceThreshold(0.9).fit_transform(x_data_generated).shape)

  x_data_kbest = SelectKBest(f_classif, k=5).fit_transform(x_data_generated, y_data_generated)
  x_data_varth = VarianceThreshold(0.9).fit_transform(x_data_generated)

  score = cross_val_score(LogisticRegression(), x_data_generated, y_data_generated, scoring='neg_log_loss').mean()
  print(score)

  score = cross_val_score(LogisticRegression(), x_data_kbest, y_data_generated, scoring='neg_log_loss').mean()
  print(score)

  score = cross_val_score(LogisticRegression(), x_data_varth, y_data_generated, scoring='neg_log_loss').mean()
  print(score)

  return

def baseline_modelling(df):
  x_data_generated, y_data_generated = make_classification()

  pipe = make_pipeline(SelectFromModel(estimator=RandomForestClassifier()), LogisticRegression())

  lr = LogisticRegression()
  rf = RandomForestClassifier()

  score = cross_val_score(lr, x_data_generated, y_data_generated, scoring='neg_log_loss').mean()
  print(score)

  score = cross_val_score(rf, x_data_generated, y_data_generated, scoring='neg_log_loss').mean()
  print(score)

  score = cross_val_score(pipe, x_data_generated, y_data_generated, scoring='neg_log_loss').mean()
  print(score)

  # -----------------

  x_data, y_data = get_data(df)
  x_data = x_data.values

  pipe1 = make_pipeline(StandardScaler(),
                        SelectFromModel(estimator=RandomForestClassifier()),
                        LogisticRegression())

  pipe2 = make_pipeline(StandardScaler(),
                        LogisticRegression())

  rf = RandomForestClassifier()

  score = cross_val_score(pipe1, x_data, y_data, scoring='neg_log_loss').mean()
  print('LR + selection', score)

  score = cross_val_score(pipe2, x_data, y_data, scoring='neg_log_loss').mean()
  print('LR', score)

  score = cross_val_score(rf, x_data, y_data, scoring='neg_log_loss').mean()
  print('RF', score)

  return

def selector(df):
  x_data, y_data = get_data(df)
  x_data_scaled = StandardScaler().fit_transform(x_data)

  selector = SequentialFeatureSelector(LogisticRegression(), scoring='neg_log_loss', verbose=2, k_features=3, forward=False, n_jobs=-1)
  selector.fit(x_data_scaled, y_data)
  # no output :(

  return