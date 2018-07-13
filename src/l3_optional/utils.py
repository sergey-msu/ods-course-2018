def hist(y):
  hs = {}

  for item in y:
    if item in hs:
      hs[item] += 1
    else:
      hs[item] = 1

  return hs

def freqs(y):
  n = len(y)
  fs = hist(y)

  for cls in fs:
    fs[cls] /= n

  return fs
