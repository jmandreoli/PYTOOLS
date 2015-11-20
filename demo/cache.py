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
from ..cache import lru_persistent_cache, make_process, ARG, State
automatic = False

@lru_persistent_cache(db=DIR,ignore=('z',))
def simplefunc(x,y=3,z=8): return x,y

@lru_persistent_cache(db=DIR,ignore=('delay',))
def longfunc(x,delay=10):
  from time import sleep
  sleep(delay)
  return x

def stepA(a,b,z=None):
  state = State()
  state.a = a
  state.b = b
  state.u = a+b
  return state

def stepB(state,c,z=0):
  state.v = c*state.u+z
  return state

proc = make_process(ARG('s_A',stepA),ARG('s_B',stepB),ignore=('z',),db=DIR,s_B=dict(ignore=()))

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
      ((simplefunc.cache,),'simplefunc(1,2) ; simplefunc(1,y=2,z=3)'),
      ((proc.cache,proc.cache.base,),'proc(s_A=ARG(1,b=2,z=36),s_B=ARG(3)).v ; proc(s_A=ARG(1,2),s_B=ARG(3,1)).v'),
      ((longfunc.cache,),'longfunc(42,6)'),
  )
  for caches,tests in DEMOS:
    print(80*'-')
    for c in caches: print('Clearing',c); c.clear()
    L = tests.split(' ; ')
    for w in [subprocess.Popen([python,__file__,str(t)]+L) for t in (0.,2.)]: w.wait()
    if not automatic:
      try: input('RET: continue; ^-C: stop')
      except: print(); break

