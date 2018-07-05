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
#from l6_homework.alice_competition_bench2 import header, run
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



def delete():

  n = int(input())
  d = list(map(int, input().split()))

  r = n - 1
  for i in range(1, n-1):
    l = n - 1 - i
    di = d[i]
    r += (l if di==0 else l + (l + 1)*di)

  for j in range(1, n-1):
    dj = d[j]
    if dj == 0:
      r += j
      for i in range(1, j):
        l = j - i
        di = d[i]
        r += (l if di==0 else l + (l + 1)*di)
    elif dj==1:
      r += (2*j + 2)
      for i in range(1, j):
        l = j - i
        di = d[i]
        r += (2*l + 1 if di == 0 else l + (l + 1)*(di + 1) + (l + 2)*di)
    elif dj==2:
      r += (4 + j + (j + 1)*2)
      for i in range(1, j):
        l = j - i
        di = d[i]
        r += (l + (l + 1)*2  if di ==0 else l + (l + 1)*(di + 2) + (l + 2)*di*2)
    else:
      r += (j + (dj + j + 1)*dj)
      for i in range(1, j):
        l = j - i
        di = d[i]
        r += (l + (l + 1)*dj  if di ==0 else l + (l + 1)*(di + dj) + (l + 2)*di*dj)

  print(r % 1000000007)

  return







  n = int(input())
  a = list(map(int, input().split()))

  d = set()
  c = 0
  for x in a:
    if x in d:
      c += 1
    else:
      d.add(x)

  print(c)




  return







  import math

  def gcd(a, b):
    while b:
        a, b = b, a%b
    return a

  def binomial(n, k):
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

  y1, x1 = map(int, input().split())
  y2, x2 = map(int, input().split())

  c1 = binomial(x1, y1)
  c2 = binomial(x2, y2)
  print(gcd(c1, c2))

  return