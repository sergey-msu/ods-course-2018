import time as t
import utils
#from alice.week1_prepare_dataset import header as week_header, run as week_run
#from alice.week2_init_analysis   import header as week_header, run as week_run
#from alice.week3_visualization   import header as week_header, run as week_run
#from alice.week4_model_fit       import header as week_header, run as week_run
#from alice.week5_competition     import header as week_header, run as week_run
from alice.week6_bigdata         import header as week_header, run as week_run

def header(): return 'PROJECT "ALICE"'

def run():

  utils.PRINT.HEADER(week_header())
  start = t.time()
  week_run()
  end = t.time()
  utils.PRINT.HEADER('DONE in {}s'.format(round(end-start, 2)))

  return

if __name__=='__main__':
  run()