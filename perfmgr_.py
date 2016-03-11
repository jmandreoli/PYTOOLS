# Only python2 syntax here

def _main():
  import pickle, sys
  stdin = sys.stdin if sys.version_info[0]==2 else sys.stdin.buffer
  stdout = sys.stdout if sys.version_info[0]==2 else sys.stdout.buffer
  from time import clock
  testbed,name,kargs = pickle.load(stdin)
  r = None
  try: exec(testbed,globals()); test = globals()[name]
  except: r = sys.exc_info()[1]
  pickle.dump(r,stdout)
  while True:
    try:
      try: sz = pickle.load(stdin)
      except EOFError: return
      e = test(sz,**kargs)
      next(e)
      t = clock()
      next(e)
      t = clock()-t
    except: t = sys.exc_info()[1]
    pickle.dump(t,stdout)

if __name__ == '__main__': _main()
