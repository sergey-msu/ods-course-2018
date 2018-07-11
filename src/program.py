import time as t
import datetime as dt
import math
import utils
#from l1_pandas          import header, run
#from l2_visualization   import header, run
#from l3_classification  import header, run
#from l4_linear          import header, run
#from l5_composition     import header, run
#from l6_features        import header, run
#from l6_homework.alice_competition_bench1 import header, run
from l6_homework.alice_competition_bench2 import header, run
#from l7_unsupervised      import header, run
#from l8_bigdata import header, run
#from l9_timeseries import header, run
#from l10_gboosting import header, run

#from alice.alice_main import header, run


def main(args=None):

  utils.PRINT.HEADER(header())
  print('STARTED ', dt.datetime.now())
  start = t.time()
  run()
  end = t.time()
  utils.PRINT.HEADER('DONE in {}s ({}m)'.format(round(end-start, 2), round((end-start)/60, 2)))

  return


if __name__=='__main__':
  main()