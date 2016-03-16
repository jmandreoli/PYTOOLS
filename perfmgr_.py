# Only python2 syntax here

def _main():
  import pickle, sys, gc
  stdin = sys.stdin if sys.version_info[0]==2 else sys.stdin.buffer
  stdout = sys.stdout if sys.version_info[0]==2 else sys.stdout.buffer
  testbed,name,kargs = pickle.load(stdin)
  r = None
  G = globals()
  try: exec(testbed,G); test = G[name]
  except: r = sys.exc_info()[1]
  pickle.dump(r,stdout)
  while True:
    gc.collect()
    try:
      try: x = pickle.load(stdin)
      except EOFError: return
      r = test(x,**kargs)
    except: r = sys.exc_info()[1]
    pickle.dump(r,stdout)

if __name__ == '__main__': _main()
