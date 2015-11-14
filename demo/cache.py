# File:                 demo/cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module

if __name__=='__main__':
  import sys
  from myutil.demo.cache import demo, demo_
  if len(sys.argv) == 1: demo()
  else: demo_(*sys.argv[1:])
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

from pathlib import Path; DIR = Path(__file__).resolve().parent/'cache.dir'
from ..cache import lru_persistent_cache, make_cache, ARG, State
automatic = False

@lru_persistent_cache(db=DIR,ignore=('z',))
def simplefunc(x,y=3,z=8): return x,y

@lru_persistent_cache(db=DIR,ignore=('delay',))
def longfunc(x,delay=10):
  from time import sleep
  sleep(delay)
  return x

def stepA(a,b):
  state = State()
  state.a = a
  state.b = b
  state.u = a+b
  return state

def stepB(state,c):
  state.v = c*state.u
  return state

process = make_cache(stepA,db=DIR)
process = make_cache(stepB,base=process,db=DIR)

#--------------------------------------------------------------------------------------------------

def demo_(t,*L):
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
      ((simplefunc.cache,),('simplefunc(1,2)','simplefunc(1,y=2,z=3)')),
      ((process,process.parent,),('process(ARG(ARG(1,b=2),3)).v','process(ARG(ARG(1,2),4)).v')),
      ((longfunc.cache,),('longfunc(42,6)',)),
  )
  def cmd(t,L): return (python,__file__,str(t))+L
  for cs,L in DEMOS:
    print(80*'-')
    for c in cs: print('Clearing',c); c.clear()
    for w in [subprocess.Popen(cmd(t,L)) for t in (0.,2.)]: w.wait()
    if not automatic:
      try: input('RET: continue; ^-C: stop')
      except: print(); break

