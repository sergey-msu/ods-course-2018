import os
import re
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from l8_script import preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import fetch_20newsgroups, load_files
from scipy.sparse import csr_matrix

def header(): return 'LECTURE 7: Bigdata - Gradient Descent and Vowpal Wabbit https://habrahabr.ru/company/ods/blog/326418/';

def run():

  #sgd()
  #categorical()
  #vowpal()
  homework()

  return

def sgd():
  data_demo = pd.read_csv(utils.PATH.COURSE_FILE('weights_heights.csv'))

  plt.scatter(data_demo['Weight'], data_demo['Height'])
  plt.xlabel('Weight in pounds')
  plt.ylabel('Height in inches')
  #plt.show()

  # see SGDClassifier, SGDRegressor from sklearn.linear_model
  return

def categorical():
  df     = pd.read_csv(utils.PATH.COURSE_FILE('bank_train.csv'))
  labels = pd.read_csv(utils.PATH.COURSE_FILE('bank_train_target.csv'), header=None)
  print(df.head())

  #label_encoding(df, labels)
  #one_hot_encoding(df, labels)
  hashing_trick(df, labels)

  return

def logistic_regression_accuracy_on(dataframe, labels):
  features = dataframe.as_matrix()
  train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels)

  logit = LogisticRegression()
  logit.fit(train_features, train_labels)

  return classification_report(test_labels, logit.predict(test_features))

def label_encoding(df, labels):
  df['education'].value_counts().plot.barh()
  #plt.show()

  label_encoder = LabelEncoder()
  mapped_edication = pd.Series(label_encoder.fit_transform(df['education']))
  mapped_edication.value_counts().plot.barh()
  #plt.show()

  print(dict(enumerate(label_encoder.classes_)))

  categorical_columns = df.columns[df.dtypes=='object'].union(['education'])
  for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
  print(df.head())

  print(logistic_regression_accuracy_on(df[categorical_columns], labels))

  return

def one_hot_encoding(df, labels):

  label_encoder = LabelEncoder()
  categorical_columns = df.columns[df.dtypes=='object'].union(['education'])
  for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

  onehot_encoder = OneHotEncoder(sparse=False)
  encoded_categorical_columns = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_columns]))
  print(encoded_categorical_columns.head())

  print(logistic_regression_accuracy_on(encoded_categorical_columns, labels))

  return

def hashing_trick(df, labels):
  hash_module = 25
  for s in ('feature_1', 'feature_2', 'feature_3'):
    print(s, '->', hash(s) % hash_module)

  #example
  hashing_example = pd.DataFrame([{i: 0.0 for i in range(hash_module)}])
  for s in ('job=student', 'marital=single', 'day_of_week=mon'):
    h = hash(s) % hash_module
    print(s, '->', h)
    hashing_example.loc[0, h] = 1
  print(hashing_example)

  return

def vowpal():
  #vowpal_binclass()
  #vowpal_multiclass()
  #vowpal_movies()
  vowpal_stackoverflow()

  return

def to_vw_format(document, label=None):
  return (str(label or '') + ' |text ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n')

def vowpal_binclass():
  newsgroups = fetch_20newsgroups(utils.PATH.COURSE_PATH('news_data'))
  text = newsgroups['data'][0]
  target = newsgroups['target_names'][newsgroups['target'][0]]

  print('----')
  print(target)
  print('----')
  print(text.strip())
  print('----')

  txt = to_vw_format(text, 1 if target == 'rec.autos' else -1)

  print(txt)

  print('---------------------------------------------')

  all_documents = newsgroups['data']
  all_targets   = [1 if newsgroups['target_names'][target] == 'rec.autos' else -1
                   for target in newsgroups['target']]
  train_documents, test_documents, train_labels, test_labels = train_test_split(all_documents, all_targets, random_state=7)

  with codecs.open(utils.PATH.COURSE_FILE('20news_train.vw', 'news_data'), 'w', 'utf-8') as vw_train_data:
    for text, target in zip(train_documents, train_labels):
      vw_train_data.write(to_vw_format(text, target))
  with codecs.open(utils.PATH.COURSE_FILE('20news_test.vw', 'news_data'), 'w', 'utf-8') as vw_test_data:
    for text in test_documents:
      vw_test_data.write(to_vw_format(text))

  print('DONE')

  print('C:\Program Files\VowpalWabbit>vw F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_train.vw --loss_function hinge -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_model.vw')
  print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_model.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_test_predictions.txt')

  with open(utils.PATH.COURSE_FILE('20news_test_predictions.txt', 'news_data')) as pred_file:
    test_predictions = [ float(label) for label in pred_file.readlines() ]

  auc = roc_auc_score(test_labels, test_predictions)
  rc = roc_curve(test_labels, test_predictions)

  with plt.xkcd():
    plt.plot(rc[0], rc[1])
    plt.plot([0, 1], [0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('test AUC = %f' % (auc))
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.show()

  return

def vowpal_multiclass():
  newsgroups = fetch_20newsgroups(utils.PATH.COURSE_PATH('news_data'))
  all_documents = newsgroups['data']
  topic_encoder = LabelEncoder()
  all_targets_mult = topic_encoder.fit_transform(newsgroups['target']) + 1

  train_documents, test_documents, train_labels_mult, test_labels_mult = train_test_split(all_documents, all_targets_mult, random_state=7)

  with codecs.open(utils.PATH.COURSE_FILE('20news_train_mult.vw', 'news_data'), 'w', 'utf-8') as vw_train_data:
    for text, target in zip(train_documents, train_labels_mult):
      vw_train_data.write(to_vw_format(text, target))
  with codecs.open(utils.PATH.COURSE_FILE('20news_test_mult.vw', 'news_data'), 'w', 'utf-8') as vw_test_data:
    for text in test_documents:
      vw_test_data.write(to_vw_format(text))

  print('vw --oaa 20 F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_train_mult.vw --loss_function hinge -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_model_mult.vw')
  print('vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_model_mult.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_test_mult.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\news_data\20news_test_predictions_mult.txt')

  with open(utils.PATH.COURSE_FILE('20news_test_predictions_mult.txt', 'news_data')) as pred_file:
    test_prediction_mult = [float(label) for label in pred_file.readlines()]

  acc = accuracy_score(test_labels_mult, test_prediction_mult)
  print(acc) # 87%

  # atheism confuse
  M = confusion_matrix(test_labels_mult, test_prediction_mult)
  for i in np.where(M[0, :] > 0)[0][1:]:
    print(newsgroups['target_names'][i], M[0, i])

  return

def vowpal_movies():
  path_to_movies = utils.PATH.COURSE_PATH('aclImdb')

  reviews_train = load_files(os.path.join(path_to_movies, 'train'))
  text_train, y_train = reviews_train.data, reviews_train.target

  print('Training data #:', len(text_train))
  print(np.bincount(y_train))

  reviews_test = load_files(os.path.join(path_to_movies, 'test'))
  text_test, y_test = reviews_test.data, reviews_train.target

  print("Test data #:", len(text_test))
  print(np.bincount(y_test))

  train_share = int(0.7 * len(text_train))
  train, valid = text_train[:train_share], text_train[train_share:]
  train_labels, valid_labels = y_train[:train_share], y_train[train_share:]

  #with codecs.open(utils.PATH.COURSE_FILE('movie_reviews_train.vw', 'aclImdb'), 'w', 'utf-8') as vw_train_data:
  #  for text, target in zip(train, train_labels):
  #    vw_train_data.write(to_vw_format(str(text), 1 if target == 1 else -1))
  #with codecs.open(utils.PATH.COURSE_FILE('movie_reviews_valid.vw', 'aclImdb'), 'w', 'utf-8') as vw_train_data:
  #  for text, target in zip(valid, valid_labels):
  #    vw_train_data.write(to_vw_format(str(text), 1 if target == 1 else -1))
  #with codecs.open(utils.PATH.COURSE_FILE('movie_reviews_test.vw', 'aclImdb'), 'w', 'utf-8') as vw_test_data:
  #  for text in text_test:
  #    vw_test_data.write(to_vw_format(str(text)))

  print('C:\Program Files\VowpalWabbit>vw -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_train.vw --loss_function hinge -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_model.vw')
  print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_model.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_valid_pred.txt --quiet')

  with open(utils.PATH.COURSE_FILE('movie_valid_pred.txt', 'aclImdb')) as pred_file:
    valid_prediction = [float(label) for label in pred_file.readlines()]
  print("Accuracy: {}".format(round(accuracy_score(valid_labels, [int(pred_prob > 0) for pred_prob in valid_prediction]), 3)))
  print("AUC: {}".format(round(roc_auc_score(valid_labels, valid_prediction), 3)))

  print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_model.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_test_pred.txt --quiet')

  with open(utils.PATH.COURSE_FILE('movie_test_pred.txt', 'aclImdb')) as pred_file:
    test_prediction = [float(label) for label in pred_file.readlines()]
  print("Accuracy: {}".format(round(accuracy_score(y_test, [int(pred_prob > 0) for pred_prob in test_prediction]), 3)))
  print("AUC: {}".format(round(roc_auc_score(y_test, test_prediction), 3)))

  # bigramms

  print('C:\Program Files\VowpalWabbit>vw -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_train.vw --loss_function hinge --ngram 2 -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_model2.vw')
  print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_model2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_valid_pred2.txt --quiet')

  with open(utils.PATH.COURSE_FILE('movie_valid_pred2.txt', 'aclImdb')) as pred_file:
    valid_prediction = [float(label) for label in pred_file.readlines()]
  print("Accuracy: {}".format(round(accuracy_score(valid_labels, [int(pred_prob > 0) for pred_prob in valid_prediction]), 3)))
  print("AUC: {}".format(round(roc_auc_score(valid_labels, valid_prediction), 3)))

  print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_model2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_reviews_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\aclImdb\movie_test_pred2.txt --quiet')

  with open(utils.PATH.COURSE_FILE('movie_test_pred2.txt', 'aclImdb')) as pred_file:
    test_prediction = [float(label) for label in pred_file.readlines()]
  print("Accuracy: {}".format(round(accuracy_score(y_test, [int(pred_prob > 0) for pred_prob in test_prediction]), 3)))
  print("AUC: {}".format(round(roc_auc_score(y_test, test_prediction), 3)))


  return

def vowpal_stackoverflow():
  with open(utils.PATH.COURSE_FILE('stackoverflow.10kk.tsv', 'stackoverflow_big')) as pred_file:
    for line in pred_file.readline():
      x = line

  return

def homework():

  def data_preprocessing():
    print('DATA PREPROCESSING')
    preprocess.run(utils.PATH.COURSE_FILE('stackoverflow.10kk.tsv', 'stackoverflow_big'),
                   utils.PATH.COURSE_FILE('stackoverflow.vw', 'stackoverflow_big'))
    return
  data_preprocessing()

  def training():
    print('TRAINING')

    print('C:\Program Files\VowpalWabbit>vw --passes 1 --ngram 1 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n1.vw')
    print('C:\Program Files\VowpalWabbit>vw --passes 1 --ngram 2 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n2.vw')
    print('C:\Program Files\VowpalWabbit>vw --passes 1 --ngram 3 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n3.vw')
    print('C:\Program Files\VowpalWabbit>vw --passes 3 --ngram 1 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -c -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n1.vw')
    print('C:\Program Files\VowpalWabbit>vw --passes 3 --ngram 2 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -c -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n2.vw')
    print('C:\Program Files\VowpalWabbit>vw --passes 3 --ngram 3 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -c -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n3.vw')
    print('C:\Program Files\VowpalWabbit>vw --passes 5 --ngram 1 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -c -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n1.vw')
    print('C:\Program Files\VowpalWabbit>vw --passes 5 --ngram 2 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -c -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n2.vw')
    print('C:\Program Files\VowpalWabbit>vw --passes 5 --ngram 3 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -c -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n3.vw')
    return
  training()

  def validation():
    print('VALIDATION')

    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n1.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p1_n1.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p1_n2.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n3.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p1_n3.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n1.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p3_n1.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p3_n2.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n3.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p3_n3.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n1.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p5_n1.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p5_n2.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n3.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_valid.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\validation\stackoverflow_valid_pred_p5_n3.txt --quiet')

    with open(utils.PATH.COURSE_FILE('stackoverflow_valid_labels.txt', 'stackoverflow_big')) as labels_file:
      valid_labels = [float(label) for label in labels_file.readlines()]

    for p in [1, 3, 5]:
      for n in [1, 2, 3]:
        with open(utils.PATH.COURSE_FILE('stackoverflow_valid_pred_p{0}_n{1}.txt'.format(p, n), 'stackoverflow_big\\validation')) as pred_file:
          valid_prediction = [float(label) for label in pred_file.readlines()]
          print("validation accuracy (p={0} n={1}): {2}".format(p, n, round(accuracy_score(valid_labels, valid_prediction), 6)))

    return
  validation()

  def testing():
    print('TESTING')

    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n1.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p1_n1.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p1_n2.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p1_n3.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p1_n3.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n1.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p3_n1.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p3_n2.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p3_n3.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p3_n3.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n1.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p5_n1.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p5_n2.txt --quiet')
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_model_p5_n3.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_test_pred_p5_n3.txt --quiet')

    with open(utils.PATH.COURSE_FILE('stackoverflow_test_labels.txt', 'stackoverflow_big')) as labels_file:
      test_labels = [float(label) for label in labels_file.readlines()]

    for p in [1, 3, 5]:
      for n in [1, 2, 3]:
        with open(utils.PATH.COURSE_FILE('stackoverflow_test_pred_p{0}_n{1}.txt'.format(p, n), 'stackoverflow_big\\testing')) as pred_file:
          test_prediction = [float(label) for label in pred_file.readlines()]
          print("testing accuracy (p={0} n={1}): {2}".format(p, n, round(accuracy_score(test_labels, test_prediction), 6)))

    return
  testing()

  def full_model():
    print('FULL MODEL')

    #training with best parameters
    print('C:\Program Files\VowpalWabbit>vw --passes 1 --ngram 2 --loss_function hinge --bit_precision 28 --random_seed 17 --oaa 10 -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_full_train.vw -f F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_full_model_p1_n2.vw')

    #testing
    print('C:\Program Files\VowpalWabbit>vw -i F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\models\stackoverflow_full_model_p1_n2.vw -t -d F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\stackoverflow_test.vw -p F:\Work\My\Python\ods\course2018\_src\mlcourse_open-master\data\stackoverflow_big\testing\stackoverflow_full_test_pred_p1_n2.txt --quiet')

    with open(utils.PATH.COURSE_FILE('stackoverflow_test_labels.txt', 'stackoverflow_big')) as labels_file:
      test_labels = [float(label) for label in labels_file.readlines()]

    with open(utils.PATH.COURSE_FILE('stackoverflow_full_test_pred_p1_n2.txt', 'stackoverflow_big\\testing')) as pred_file:
      test_prediction = [float(label) for label in pred_file.readlines()]
      print("testing accuracy (p=1 n=2): {0}".format(round(accuracy_score(test_labels, test_prediction), 6)))

    return
  full_model()

  return