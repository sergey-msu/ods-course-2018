import sys
import os
import re

def main(args=None):
  args = args or sys.argv[1:]

  if args is None or len(args) != 2:
    print('incorrect number of agruments: expected 2')
    return

  input_fname = args[0]
  if not os.path.exists(input_fname):
    print('input file does not exist')
    return

  output_fname = args[1]

  run(input_fname, output_fname)

  return

def run(input_fname, output_fname):
  print('INPUT:',  input_fname)
  print('OUTPUT:', output_fname)

  build_data(input_fname, output_fname)
  split_data(output_fname)

  return

def build_data(input_fname, output_fname):
  if os.path.exists(output_fname):
    return

  print('build data: start')

  labels = set([ 'javascript', 'java', 'python', 'ruby', 'php', 'c++', 'c#', 'go', 'scala', 'swift' ])
  labels_dict = { l:i+1 for (i,l) in enumerate(labels) }
  line_splitter = '\t'
  tags_splitter  = ' '
  spec_sym1 = ':'
  spec_sym2 = '|'
  spec_sym_replace = ' '
  tot_lines = 0
  prc_lines = 0

  def to_vw_format(doc, label):
    return '{0} |text {1}\n'.format(label, tags_splitter.join(re.findall('\w{3,}', doc.lower())))

  with open(input_fname,  'r') as input_file, \
       open(output_fname, 'w') as output_file:
    for line in input_file:
      tot_lines += 1
      if tot_lines%100000==0:
        print('processed/total lines: {0}/{1}'.format(prc_lines, tot_lines))

      segs = line.split(line_splitter)
      if (len(segs) != 2):
        continue

      doc  = segs[0].replace(spec_sym1, spec_sym_replace).replace(spec_sym2, spec_sym_replace)
      tags = set(segs[1].rstrip().split(tags_splitter))
      common_tags = list(labels.intersection(tags))
      if (len(common_tags) != 1):
        continue

      output_file.write(to_vw_format(doc, labels_dict[common_tags[0]]))
      prc_lines += 1

  print('build data: done')
  return

def split_data(data_fname):

  dir_name = os.path.dirname(data_fname)
  train_data_fname      = os.path.join(dir_name, 'stackoverflow_train.vw')
  valid_data_fname      = os.path.join(dir_name, 'stackoverflow_valid.vw')
  full_train_data_fname = os.path.join(dir_name, 'stackoverflow_full_train.vw')
  test_data_fname       = os.path.join(dir_name, 'stackoverflow_test.vw')

  train_labels_fname      = os.path.join(dir_name, 'stackoverflow_train_labels.txt')
  valid_labels_fname      = os.path.join(dir_name, 'stackoverflow_valid_labels.txt')
  full_train_labels_fname = os.path.join(dir_name, 'stackoverflow_full_train_labels.txt')
  test_labels_fname       = os.path.join(dir_name, 'stackoverflow_test_labels.txt')

  if os.path.exists(train_data_fname)   and \
     os.path.exists(test_data_fname)    and \
     os.path.exists(valid_data_fname)   and \
     os.path.exists(full_train_data_fname)   and \
     os.path.exists(train_labels_fname) and \
     os.path.exists(full_train_labels_fname)  and \
     os.path.exists(test_labels_fname)  and \
     os.path.exists(valid_labels_fname):
    return

  print('split data: start')
  vw_splitter = '|'
  eol = '\n'
  tot_cnt   = 4389057
  valid_cnt = tot_cnt//3
  test_cnt  = 2*tot_cnt//3
  line_cnt  = 0

  with open(data_fname, 'r')              as data_file, \
       open(train_data_fname, 'w')        as train_data_file, \
       open(test_data_fname, 'w')         as test_data_file, \
       open(valid_data_fname, 'w')        as valid_data_file, \
       open(full_train_data_fname, 'w')   as full_train_data_file, \
       open(train_labels_fname, 'w')      as train_labels_file, \
       open(test_labels_fname, 'w')       as test_labels_file, \
       open(full_train_labels_fname, 'w') as full_train_labels_file, \
       open(valid_labels_fname, 'w')      as valid_labels_file:
    for line in data_file:

      if line_cnt < valid_cnt:
        df, lf = train_data_file, train_labels_file
      elif line_cnt < test_cnt:
        df, lf = valid_data_file, valid_labels_file
      else:
        df, lf = test_data_file, test_labels_file

      df.write(line)
      label = line.split(vw_splitter)[0].strip() + eol
      lf.write(label)

      if line_cnt < test_cnt:
        full_train_data_file.write(line)
        full_train_labels_file.write(label)

      line_cnt += 1
      if line_cnt%100000 == 0:
        print('processed: ', line_cnt)

  print('split data: done')
  return


if __name__=='__main__':
  main()