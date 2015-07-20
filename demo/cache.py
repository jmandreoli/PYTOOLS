# File:                 demo/cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module

if __name__=='__main__':
  import sys
  from myutil.demo.cache import demo, demo1
  if len(sys.argv) == 1: demo()
  else: demo1(*sys.argv[1:])
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

from pathlib import Path; DIR = Path(__file__).resolve().parent/'cache.dir'
from ..cache import lru_persistent_cache, lru_persistent_process_cache, ARG
automatic = False

@lru_persistent_cache(db=DIR,ignore=('z',))
def simplefunc(x,y=3,z=8): return x,y

@lru_persistent_cache(db=DIR,ignore=('delay',))
def longfunc(x,delay=10):
  from time import sleep
  sleep(delay)
  return x

def stepA(state,a,b):
  state.a = a
  state.b = b
  state.u = a+b
  return state

def stepB(state,c):
  state.v = c*state.u
  return state

process = lru_persistent_process_cache((stepA,dict(db=DIR)),(stepB,dict(db=DIR)))

def demo1(t,*L):
  import time, logging
  logging.basicConfig(level=logging.INFO,format='[proc %(process)d @ %(asctime)s] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger()
  time.sleep(float(t))
  for x in L:
    logger.info('Computing: %s',x)
    logger.info('Result: %s = %s',x,eval(x))

def demo():
  import subprocess
  from sys import executable as python
  DEMOS = (
      (simplefunc,('simplefunc(1,2)','simplefunc(1,y=2,z=3)')),
      (process,('process(ARG(1,b=2),ARG(3)).v','process(ARG(1,2),ARG(4)).v')),
      (longfunc,('longfunc(42,6)',)),
  )
  def cmd(t,L): return (python,__file__,str(t))+L
  for f,L in DEMOS:
    print(80*'-'); print('Clearing',f); f.clear()
    for w in [subprocess.Popen(cmd(t,L)) for t in (0.,2.)]: w.wait()
    if not automatic:
      try: input('RET: continue; Ctrl-C: stop')
      except: print(); break

